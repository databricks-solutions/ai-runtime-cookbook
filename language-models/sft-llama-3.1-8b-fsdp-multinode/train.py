#!/usr/bin/env python3
"""Multi-node FSDP supervised fine-tuning of Llama-3.1-8B.

Launched via ``torchrun`` from the workload YAML ``command`` across 2 nodes x 8 H100 (16 ranks). Each rank
owns one GPU. The model is sharded with PyTorch FSDP (full shard + bf16), trained on
an instruction dataset, and the consolidated checkpoint is written to a Unity Catalog
Volume by rank 0. Metrics are logged to MLflow.

Hyperparameters are read from the YAML block passed by ``air`` via HYPERPARAMETERS_PATH.
"""

import functools
import os

import mlflow
import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

LOG_EVERY = 10


def load_params() -> dict:
    """Read the hyperparameters block that `air` materializes from the YAML `parameters:`."""
    with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
        return yaml.safe_load(f)


def build_dataset(tokenizer, dataset_name: str, max_seq_len: int):
    """Tokenize an instruction dataset into fixed-length causal-LM examples."""
    raw = load_dataset(dataset_name, split="train")

    def format_example(row):
        instruction = row.get("instruction", "")
        context = row.get("input", "")
        response = row.get("output", "")
        prompt = f"### Instruction:\n{instruction}\n\n"
        if context:
            prompt += f"### Input:\n{context}\n\n"
        text = f"{prompt}### Response:\n{response}{tokenizer.eos_token}"
        out = tokenizer(text, truncation=True, max_length=max_seq_len, padding="max_length")
        # Ignore padding tokens in the causal language modeling loss.
        out["labels"] = [
            token_id if mask else -100
            for token_id, mask in zip(out["input_ids"], out["attention_mask"])
        ]
        return out

    cols = raw.column_names
    tokenized = raw.map(format_example, remove_columns=cols)
    # Emit torch tensors so the default DataLoader collate stacks them into [B, L] batches.
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def main():
    # AIR sets MLFLOW_RUN_ID. Check it on every rank before any collective call, so a missing run
    # fails at startup instead of hanging the other ranks.
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError("MLFLOW_RUN_ID is not set; AIR provides it for each run.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    p = load_params()
    model_name = p["model_name"]
    dataset_name = p["dataset_name"]
    max_seq_len = int(p["max_seq_len"])
    batch_size = int(p["per_device_batch_size"])
    grad_accum = int(p["gradient_accumulation_steps"])
    lr = float(p["learning_rate"])
    max_steps = int(p["max_steps"])
    # Keyed by the MLflow run, so a rerun does not overwrite the previous checkpoint.
    output_dir = os.path.join(p["output_dir"], run_id)

    if rank == 0:
        print(f"World size={world_size} | model={model_name} | dataset={dataset_name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer})
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=local_rank,
        use_orig_params=True,
    )

    dataset = build_dataset(tokenizer, dataset_name, max_seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if rank == 0:
        mlflow.start_run(run_id=run_id)
        mlflow.log_params({"model_name": model_name, "lr": lr, "batch_size": batch_size, "world_size": world_size})

    model.train()
    sampler.set_epoch(0)
    step = 0
    optimizer.zero_grad()
    for micro_step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        (out.loss / grad_accum).backward()

        if (micro_step + 1) % grad_accum == 0:
            model.clip_grad_norm_(1.0)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            # Reading the loss syncs the GPU, so rank 0 reads it once per LOG_EVERY steps.
            if rank == 0 and (step % LOG_EVERY == 0 or step >= max_steps):
                loss = out.loss.item()
                print(f"step={step}/{max_steps} loss={loss:.4f}", flush=True)
                mlflow.log_metric("train_loss", loss, step=step)
            if step >= max_steps:
                break

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        model.module.save_pretrained(output_dir, state_dict=cpu_state)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved checkpoint to {output_dir}", flush=True)
        mlflow.end_run()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
