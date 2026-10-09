#!/usr/bin/env python3
"""Fine-tune a DINOv3 ViT-Small backbone and a new Flowers-102 classifier."""

import json
import math
import os
import random
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import timm
import torch
import yaml
from mlflow.models import infer_signature
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(p, data_config):
    train_transform = timm.data.create_transform(
        **data_config, is_training=True, scale=(0.6, 1.0), hflip=0.5,
    )
    eval_transform = timm.data.create_transform(**data_config, is_training=False)
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = Flowers102(
            p["data_root"], split=split, download=True,
            transform=train_transform if split == "train" else eval_transform,
        )
        loaders[split] = DataLoader(
            dataset, batch_size=p["batch_size"], shuffle=split == "train",
            num_workers=p["num_workers"], pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(p["seed"]),
        )
    return loaders


def run_epoch(model, loader, device, loss_fn, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss during fine-tuning")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)
    return {"loss": total_loss / count, "accuracy": correct / count}


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    if not 0 <= p["warmup_epochs"] < p["epochs"]:
        raise ValueError("Require 0 <= warmup_epochs < epochs")
    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    device = torch.device("cuda")
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # All backbone parameters are trainable; a fresh 102-class head replaces the encoder head.
    model = timm.create_model(p["model_name"], pretrained=True, num_classes=102).to(device)
    data_config = timm.data.resolve_model_data_config(model)
    loaders = make_loaders(p, data_config)
    head = list(model.get_classifier().parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [parameter for parameter in model.parameters() if id(parameter) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": p["backbone_learning_rate"]},
         {"params": head, "lr": p["head_learning_rate"]}],
        weight_decay=p["weight_decay"],
    )
    total_steps = p["epochs"] * len(loaders["train"])
    warmup_steps = p["warmup_epochs"] * len(loaders["train"])

    def lr_multiplier(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    train_loss = torch.nn.CrossEntropyLoss(label_smoothing=p["label_smoothing"])
    eval_loss = torch.nn.CrossEntropyLoss()
    best_accuracy, best_epoch = -1.0, 0
    checkpoint_path = output_dir / "best_model.pt"

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"dataset": "Oxford/Flowers102", "precision": "bfloat16", **p})
        mlflow.log_params({f"{split}_images": len(loader.dataset) for split, loader in loaders.items()})
        started = time.perf_counter()
        for epoch in range(1, p["epochs"] + 1):
            train_metrics = run_epoch(model, loaders["train"], device, train_loss, optimizer, scheduler)
            val_metrics = run_epoch(model, loaders["val"], device, eval_loss)
            metrics = {
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                "backbone_learning_rate": optimizer.param_groups[0]["lr"],
                "head_learning_rate": optimizer.param_groups[1]["lr"],
            }
            mlflow.log_metrics(metrics, step=epoch)
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy, best_epoch = val_metrics["accuracy"], epoch
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Epoch {epoch}/{p['epochs']}: {metrics}", flush=True)

        # Restore the validation-selected model before touching the official test set.
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        test_metrics = run_epoch(model, loaders["test"], device, eval_loss)
        final_metrics = {
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_accuracy,
            "train_and_eval_seconds": time.perf_counter() - started,
        }
        metadata = {
            "model_name": p["model_name"], "num_classes": 102, "timm_version": timm.__version__,
            "data_config": data_config,
            "class_labels": {str(i): i + 1 for i in range(102)},
        }
        (output_dir / "model_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2) + "\n")
        mlflow.log_metrics(final_metrics)
        mlflow.log_artifact(str(output_dir / "model_config.json"))
        mlflow.log_artifact(str(output_dir / "metrics.json"))
        # Package the selected model. Its input is normalized NCHW float32 images.
        model = model.cpu().eval()
        # Two examples let torch.export retain a dynamic batch dimension in newer MLflow versions.
        example = torch.stack([loaders["val"].dataset[i][0] for i in range(2)])
        with torch.inference_mode():
            logits = model(example).numpy()
        info = mlflow.pytorch.log_model(
            pytorch_model=model, name="model", input_example=example.numpy(),
            signature=infer_signature(example.numpy(), logits),
            extra_pip_requirements=[f"timm=={timm.__version__}"],
        )
        print(f"Test metrics: {final_metrics}\nArtifacts: {output_dir}\nModel: {info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
