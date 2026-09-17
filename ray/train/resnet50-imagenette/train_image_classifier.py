#!/usr/bin/env python3
"""Distributed ResNet-50 fine-tuning on Imagenette with Ray Train.

The workload starts a two-node Ray cluster with one A10 GPU per node.
TorchTrainer launches one worker per GPU, wraps the model in DDP, shards the
dataset across workers, and reports metrics to Ray and MLflow.

The dataset and model weights are public. Each node downloads and caches them
locally, so the workload does not require credentials or shared storage.
"""

import os
import time

import mlflow
import ray
import ray.train
import torch
import torch.distributed as dist
import torch.nn as nn
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, get_device, prepare_data_loader, prepare_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import ResNet50_Weights, resnet50

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_ROOT = "/tmp/imagenette2-320"
NUM_EPOCHS = 5
PER_WORKER_BATCH_SIZE = 64
LEARNING_RATE = 1e-3

NORMALIZE = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    ),
}


def prepare_local_assets() -> None:
    """Cache the dataset and pretrained weights once on each node."""
    context = ray.train.get_context()
    if context.get_local_rank() == 0:
        if not os.path.isdir(os.path.join(IMAGENETTE_ROOT, "train")):
            download_and_extract_archive(IMAGENETTE_URL, download_root="/tmp")
        ResNet50_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    dist.barrier()


def create_data_loaders(batch_size: int) -> dict[str, DataLoader]:
    datasets_by_split = {
        split: datasets.ImageFolder(os.path.join(IMAGENETTE_ROOT, split), TRANSFORMS[split])
        for split in ("train", "val")
    }
    if datasets_by_split["train"].class_to_idx != datasets_by_split["val"].class_to_idx:
        raise ValueError("Imagenette train and validation class mappings do not match")

    loaders = {
        "train": DataLoader(
            datasets_by_split["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets_by_split["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        ),
    }
    return {split: prepare_data_loader(loader) for split, loader in loaders.items()}


def run_epoch(model, loader, loss_fn, optimizer=None) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    loss_sum = 0.0
    correct = 0
    samples = 0

    for images, labels in loader:
        if is_training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        samples += labels.size(0)

    totals = torch.tensor([loss_sum, correct, samples], dtype=torch.float64, device=get_device())
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals[0].item() / totals[2].item(), totals[1].item() / totals[2].item()


def train_loop_per_worker(config: dict) -> None:
    prepare_local_assets()
    loaders = create_data_loaders(config["batch_size"])

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = prepare_model(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()
    context = ray.train.get_context()
    world_rank = context.get_world_rank()
    use_mlflow = world_rank == 0 and bool(os.environ.get("MLFLOW_RUN_ID"))

    if use_mlflow:
        mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"])
        mlflow.log_params(
            {
                "model": "resnet50",
                "dataset": "imagenette2-320",
                "epochs": config["epochs"],
                "learning_rate": config["learning_rate"],
                "per_worker_batch_size": config["batch_size"],
                "num_workers": context.get_world_size(),
            }
        )

    for epoch in range(config["epochs"]):
        if hasattr(loaders["train"].sampler, "set_epoch"):
            loaders["train"].sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, loaders["train"], loss_fn, optimizer)
        validation_loss, validation_accuracy = run_epoch(model, loaders["val"], loss_fn)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        ray.train.report(metrics)
        if use_mlflow:
            mlflow.log_metrics({key: value for key, value in metrics.items() if key != "epoch"}, step=epoch + 1)
        if world_rank == 0:
            print(
                f"Epoch {epoch + 1}/{config['epochs']}: "
                f"train_loss={train_loss:.4f}, validation_accuracy={validation_accuracy:.4f}",
                flush=True,
            )

    if use_mlflow:
        mlflow.end_run()


def main() -> None:
    ray.init(address="auto")
    expected_gpus = int(os.environ["NUM_GPUS"])
    for _ in range(60):
        if int(ray.cluster_resources().get("GPU", 0)) >= expected_gpus:
            break
        time.sleep(5)

    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < expected_gpus:
        raise SystemExit(f"Expected {expected_gpus} GPU(s) but Ray only sees {total_gpus}")
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config={
            "epochs": NUM_EPOCHS,
            "batch_size": PER_WORKER_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        scaling_config=ScalingConfig(num_workers=total_gpus, use_gpu=True),
    )
    trainer.fit()
    print("Training finished.", flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
