#!/usr/bin/env python3
"""Train a TorchRec DLRM click-through-rate model on Criteo click logs with sharded embeddings."""

import contextlib
import itertools
import json
import os
import time
from pathlib import Path

import mlflow
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
import torchmetrics
import yaml
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.models.dlrm import DLRM, DLRMTrain
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.optim.rowwise_adagrad import RowWiseAdagrad

import data


def to_batch(dense, sparse, labels):
    """One id per feature per row; KJT values are grouped by feature, so transpose first."""
    return Batch(
        dense_features=dense,
        sparse_features=KeyedJaggedTensor.from_lengths_sync(
            keys=data.SPARSE_COLUMNS,
            values=sparse.T.reshape(-1),
            lengths=torch.ones(sparse.numel(), dtype=torch.int32),
        ),
        labels=labels,
    )


def batches(split, batch_size, steps, generator=None):
    dense, sparse, labels = split
    order = torch.randperm(len(labels), generator=generator) if generator is not None else torch.arange(len(labels))
    for step in range(steps):
        rows = order[step * batch_size:(step + 1) * batch_size]
        yield to_batch(dense[rows], sparse[rows], labels[rows])


def min_across_ranks(value, device):
    """Embedding lookups are collective, so every rank must run the same number of batches."""
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    return int(tensor.item())


def log(rank, metrics, step=None):
    if rank == 0:
        mlflow.log_metrics(metrics, step=step)
        print(f"step {step}: {metrics}", flush=True)


@torch.no_grad()
def evaluate(model, split, p, ctr, device):
    model.eval()
    steps = data.num_steps(min_across_ranks(len(split[2]), device), p["batch_size"])
    auroc = torchmetrics.AUROC(task="binary").to(device)
    totals = torch.zeros(3, dtype=torch.float64, device=device)  # model loss, baseline loss, rows
    baseline_logit = torch.logit(torch.tensor(ctr, device=device))
    for batch in batches(split, p["batch_size"], steps):
        _, (_, logits, labels) = model(batch.to(device))
        auroc.update(torch.sigmoid(logits), labels.int())
        totals[0] += F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        totals[1] += F.binary_cross_entropy_with_logits(baseline_logit.expand_as(labels), labels, reduction="sum")
        totals[2] += len(labels)
    dist.all_reduce(totals)
    model.train()
    return {"auroc": auroc.compute().item(), "log_loss": (totals[0] / totals[2]).item(),
            "baseline_log_loss": (totals[1] / totals[2]).item()}


def prefixed(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    # The CLI can serialize small numbers as 1e-08, which PyYAML reads as a string.
    for key in ("learning_rate", "eps"):
        p[key] = float(p[key])
    if p["dense_arch_layer_sizes"][-1] != p["embedding_dim"]:
        raise ValueError("The last dense_arch_layer_sizes entry must equal embedding_dim")
    if p["hash_size"] < 2 or min(p["batch_size"], p["epochs"], p["log_every_steps"], p["eval_every_steps"]) < 1:
        raise ValueError("hash_size must be at least 2; batch_size, epochs, and step intervals must be positive")

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(p["seed"])
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Each rank downloads its own files to node-local storage.
    train_names, validation_names, test_names = data.split_files(
        data.list_day_files(p["dataset_repo"], p["train_day"]),
        data.list_day_files(p["dataset_repo"], p["eval_day"]), p, rank, world)
    fetch = lambda names: [data.download(p["dataset_repo"], name, p["data_root"]) for name in names]
    tensors = lambda arrays: tuple(torch.from_numpy(array) for array in arrays)
    train = tensors(data.load_split(fetch(train_names), p["hash_size"]))
    validation = tensors(data.load_split(fetch(validation_names), p["hash_size"], rank, world))
    test = tensors(data.load_split(fetch(test_names), p["hash_size"], rank, world))
    label_totals = torch.tensor([train[2].sum().item(), len(train[2])], dtype=torch.float64, device=device)
    dist.all_reduce(label_totals)
    ctr = (label_totals[0] / label_totals[1]).item()

    tables = [EmbeddingBagConfig(name=f"t_{name}", embedding_dim=p["embedding_dim"],
                                 num_embeddings=p["hash_size"], feature_names=[name])
              for name in data.SPARSE_COLUMNS]
    # Tables start on the meta device; the sharding plan materializes each shard on its GPU.
    model = DLRMTrain(DLRM(
        embedding_bag_collection=EmbeddingBagCollection(tables=tables, device=torch.device("meta")),
        dense_in_features=len(data.DENSE_COLUMNS),
        dense_arch_layer_sizes=p["dense_arch_layer_sizes"],
        over_arch_layer_sizes=p["over_arch_layer_sizes"],
        dense_device=device,
    ))
    apply_optimizer_in_backward(RowWiseAdagrad, model.model.sparse_arch.parameters(),
                                {"lr": p["learning_rate"], "eps": p["eps"]})
    topology = Topology(world_size=world, local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
                        compute_device="cuda", hbm_cap=torch.cuda.get_device_properties(device).total_memory)
    plan = EmbeddingShardingPlanner(topology=topology, batch_size=p["batch_size"]).collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD)
    model = DistributedModelParallel(module=model, device=device, plan=plan)
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.Adagrad(params, lr=p["learning_rate"], eps=p["eps"]),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    # Reduce row counts before checking them so a short rank fails on every rank instead of hanging.
    steps_per_epoch = data.num_steps(min_across_ranks(len(train[2]), device), p["batch_size"])
    total_steps = steps_per_epoch * p["epochs"]
    if p["max_train_steps"] > 0:
        total_steps = min(total_steps, p["max_train_steps"])
    generator = torch.Generator().manual_seed(p["seed"] + rank)
    train_batches = itertools.islice(itertools.chain.from_iterable(
        batches(train, p["batch_size"], steps_per_epoch, generator) for _ in range(p["epochs"])), total_steps)

    with mlflow.start_run(run_id=run_id) if rank == 0 else contextlib.nullcontext():
        if rank == 0:
            mlflow.log_params({"dataset": p["dataset_repo"], **p, "world_size": world,
                               "train_rows_rank0": len(train[2])})
            mlflow.log_text(str(plan), "sharding_plan.txt")
            print(plan, flush=True)
        started = window_started = time.perf_counter()
        running = torch.zeros(2, dtype=torch.float64, device=device)  # loss sum, steps
        window_steps = 0
        for step, batch in enumerate(train_batches, 1):
            loss, _ = model(batch.to(device, non_blocking=True))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running[0] += loss.detach()
            running[1] += 1
            window_steps += 1
            if step % p["log_every_steps"] == 0 or step == total_steps:
                dist.all_reduce(running)
                log(rank, {"train_log_loss": (running[0] / running[1]).item(),
                           "samples_per_second": window_steps * p["batch_size"] * world
                           / (time.perf_counter() - window_started)}, step)
                running.zero_()
                window_steps, window_started = 0, time.perf_counter()
            if step % p["eval_every_steps"] == 0 and step != total_steps:
                eval_started = time.perf_counter()
                log(rank, prefixed("validation", evaluate(model, validation, p, ctr, device)), step)
                # Leave evaluation time out of the training throughput window.
                window_started += time.perf_counter() - eval_started
        train_seconds = time.perf_counter() - started

        metrics = {**prefixed("validation", evaluate(model, validation, p, ctr, device)),
                   **prefixed("test", evaluate(model, test, p, ctr, device)),
                   "train_ctr": ctr, "train_steps": total_steps, "train_seconds": train_seconds}
        log(rank, metrics, total_steps)

        # Each rank writes only its own shards; gathering every table onto one GPU would not fit.
        dcp.save(model.state_dict(), checkpoint_id=str(output_dir / "checkpoint"))
        if rank == 0:
            config = {key: p[key] for key in
                      ("hash_size", "embedding_dim", "dense_arch_layer_sizes", "over_arch_layer_sizes")}
            config.update(dense_columns=data.DENSE_COLUMNS, sparse_columns=data.SPARSE_COLUMNS, world_size=world)
            for name, value in (("model_config", config), ("metrics", metrics)):
                (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
                mlflow.log_artifact(str(output_dir / f"{name}.json"))
            print(f"Test metrics: {metrics}\nArtifacts: {output_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
