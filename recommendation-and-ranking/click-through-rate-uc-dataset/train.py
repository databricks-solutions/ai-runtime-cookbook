#!/usr/bin/env python3
"""Train a click-through-rate model on a Unity Catalog table with UCDataset.

torchrun starts one process per GPU on every node. UCDataset gives each process its own share of the
rows that match the filter, so the script has no sampler or sharding code.
"""

import math
import os

import mlflow
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from databricks.air.data import DataLoader
from databricks.uc_dataset.beta import UCDataset
from databricks.uc_dataset.beta.dataframe import col, table
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP

FEATURES = ["user_age", "user_tenure_days", "item_price", "user_ctr_7d", "item_views_7d"]
# Rough scale of each feature, so the model's inputs land in similar ranges.
SCALE = torch.tensor([100.0, 3650.0, 200.0, 1.0, 100000.0])


def to_model_inputs(batch):
    # Each batch is a dict with one tensor per column.
    features = torch.stack([batch[c].to(torch.float32) for c in FEATURES], dim=1) / SCALE
    label = batch["label"].to(torch.float32)
    return features, label


def make_loader(table_name, split, batch_size):
    # The filter runs as UCDataset reads the table's files, so rows from the other split never reach this process.
    df = (
        table(table_name)
        .filter((col("split") == split) & col("label").isNotNull())
        .select(*FEATURES, "label")
    )
    dataset = UCDataset(df=df, batch_size=batch_size)
    # UCDataset already yields whole batches, so the DataLoader must not batch again.
    return dataset, DataLoader(dataset, batch_size=None, collate_fn=to_model_inputs)


def evaluate(model, loader, device):
    """Return the summed log loss, rows, and clicks in the test split, combined across processes."""
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    totals = torch.zeros(3, device=device)
    model.eval()
    with torch.no_grad():
        for features, labels in loader:
            labels = labels.to(device)
            loss = criterion(model(features.to(device)).squeeze(-1), labels)
            totals += torch.stack([loss, torch.tensor(float(len(labels)), device=device), labels.sum()])
    model.train()
    dist.all_reduce(totals)
    return totals.tolist()


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)

    # Start the process group before creating the datasets, so rank 0 plans each read once and shares the plan.
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)

    train_data, train_loader = make_loader(p["table"], "train", p["batch_size"])
    _, test_loader = make_loader(p["table"], "test", p["batch_size"])
    train_rows = train_data.num_rows()

    model = DDP(nn.Sequential(nn.Linear(len(FEATURES), 64), nn.ReLU(), nn.Linear(64, 1)).to(device), device_ids=[device.index])
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    if rank == 0:
        mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])
        mlflow.log_params({**p, "world_size": dist.get_world_size(), "train_rows_matching": train_rows})

    for epoch in range(1, p["epochs"] + 1):
        totals = torch.zeros(2, device=device)
        # Processes can finish an epoch one batch apart. Join lets a process that runs out of batches
        # first finish cleanly, instead of waiting on a gradient all-reduce the others never join.
        with Join([model]):
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(features).squeeze(-1), labels)
                loss.backward()
                optimizer.step()
                totals += torch.stack([loss.detach() * len(labels), torch.tensor(float(len(labels)), device=device)])
        dist.all_reduce(totals)
        loss_sum, rows = totals.tolist()
        test_loss_sum, test_rows, test_clicks = evaluate(model.module, test_loader, device)
        if rank == 0:
            metrics = {"train_loss": loss_sum / rows, "train_rows": rows, "test_loss": test_loss_sum / test_rows}
            mlflow.log_metrics(metrics, step=epoch)
            print(
                f"Epoch {epoch}: read {int(rows):,} of {train_rows:,} training rows, "
                f"train loss {metrics['train_loss']:.4f}, test loss {metrics['test_loss']:.4f}",
                flush=True,
            )

    if rank == 0:
        # Log loss of always predicting the test split's click rate, for comparison.
        rate = test_clicks / test_rows
        baseline = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
        mlflow.log_metric("test_loss_baseline", baseline)
        print(f"Test loss {metrics['test_loss']:.4f}. Always predicting the click rate gives {baseline:.4f}.", flush=True)
        mlflow.end_run()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
