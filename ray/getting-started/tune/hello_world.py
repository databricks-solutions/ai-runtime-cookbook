"""Ray Tune hyperparameter search example on AI Runtime.

Runs 8 trials across all available GPUs in the cluster (one GPU per trial).
Uses ASHA scheduler to prune unpromising trials early.
"""

import os
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.tune.schedulers import ASHAScheduler

ray.init(address="auto")

num_nodes = int(os.environ.get("NUM_NODES", 1))
total_gpus = int(ray.cluster_resources().get("GPU", 0))
if total_gpus < 1:
    raise SystemExit("No GPUs registered with Ray; check GPU discovery on the cluster.")
print(f"Cluster ready: {num_nodes} node(s), {total_gpus} GPU(s) available")
print(f"Running 8 trials with up to {total_gpus} in parallel\n")


def train_fn(config):
    """Single trial: trains a small MLP on synthetic data for one GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(128, config["hidden_size"]),
        nn.ReLU(),
        nn.Linear(config["hidden_size"], 10),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        x = torch.randn(config["batch_size"], 128, device=device)
        y = torch.randint(0, 10, (config["batch_size"],), device=device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        tune.report({"loss": loss.item(), "epoch": epoch})


tuner = tune.Tuner(
    tune.with_resources(train_fn, resources={"gpu": 1}),
    param_space={
        "lr": tune.loguniform(1e-4, 1e-1),
        "hidden_size": tune.choice([64, 128, 256]),
        "batch_size": tune.choice([32, 64, 128]),
    },
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=ASHAScheduler(max_t=20, grace_period=3, reduction_factor=2),
        num_samples=8,
    ),
)

results = tuner.fit()
best = results.get_best_result("loss", "min")
print(f"\nBest config: {best.config}")
print(f"Best loss:   {best.metrics['loss']:.4f}")

ray.shutdown()
