"""Ray Train distributed training example on AI Runtime.

Trains a small MLP on synthetic data with one training worker per GPU using
Ray Train's TorchTrainer. Ray Train places the workers across the cluster
(one per GPU) and wires up torch.distributed; the per-worker train loop just
uses `ray.train.torch` helpers to move the model/data to the right device.
"""

import os

import ray
import torch
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from torch.utils.data import DataLoader, TensorDataset

# Connect to the cluster started by ray_bootstrap.sh.
ray.init(address="auto")

num_nodes = int(os.environ.get("NUM_NODES", 1))
total_gpus = int(ray.cluster_resources().get("GPU", 0))
if total_gpus < 1:
    raise SystemExit("No GPUs registered with Ray; check GPU discovery on the cluster.")
print(f"Cluster ready: {num_nodes} node(s), {total_gpus} GPU(s) available")
print(f"Launching a Ray Train run with {total_gpus} worker(s), one per GPU\n")


def train_loop_per_worker(config):
    """Runs on each Ray Train worker; one worker is pinned to one GPU."""
    # prepare_model wraps the model in DDP and moves it to this worker's GPU.
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    model = prepare_model(model)

    x = torch.randn(1024, 128)
    y = torch.randint(0, 10, (1024,))
    loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)
    # prepare_data_loader shards the data across workers and moves batches to the GPU.
    loader = prepare_data_loader(loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # ray.train.report surfaces metrics back to the driver.
        ray.train.report({"epoch": epoch, "loss": epoch_loss / len(loader)})


trainer = TorchTrainer(
    train_loop_per_worker,
    train_loop_config={"lr": 1e-3, "epochs": 5},
    scaling_config=ScalingConfig(num_workers=total_gpus, use_gpu=True),
)

result = trainer.fit()
# result.metrics holds the last reported dict (may be None if nothing was
# reported on the final iteration); fall back to a plain message.
print(f"\nTraining finished. Final metrics: {result.metrics or 'see per-worker logs above'}")

ray.shutdown()
