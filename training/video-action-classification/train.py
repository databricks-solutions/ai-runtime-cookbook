#!/usr/bin/env python3
"""Train a temporal action classifier over frozen ResNet-18 video-frame features."""

import json
import os
import random
import re
import tarfile
import time
from pathlib import Path

import av
import mlflow
import numpy as np
import pandas as pd
import timm
import torch
import yaml
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset


def split_videos(paths, seed, validation_fraction, test_fraction):
    """Keep every (action, original-video-group) in a single partition."""
    if min(validation_fraction, test_fraction) <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be positive and sum to less than one")
    rows = []
    for path in sorted(paths):
        match = re.fullmatch(r"v_(.+)_g(\d+)_c\d+\.avi", path.name)
        if match is None:
            raise ValueError(f"Unexpected UCF101 filename: {path.name}")
        label, group = match.groups()
        rows.append({"path": str(path), "video": path.name, "label": label, "group": f"{label}_g{group}"})
    if not rows or len({row["video"] for row in rows}) != len(rows):
        raise ValueError("Expected nonempty videos with unique filenames")

    rng = random.Random(seed)
    assignment = {}
    for label in sorted({row["label"] for row in rows}):
        groups = sorted({row["group"] for row in rows if row["label"] == label})
        rng.shuffle(groups)
        n_validation = max(1, int(len(groups) * validation_fraction))
        n_test = max(1, int(len(groups) * test_fraction))
        if n_validation + n_test >= len(groups):
            raise ValueError(f"Not enough independent video groups for {label}")
        for i, group in enumerate(groups):
            assignment[group] = "validation" if i < n_validation else (
                "test" if i < n_validation + n_test else "train"
            )
    for row in rows:
        row["split"] = assignment[row["group"]]
    return pd.DataFrame(rows)


def load_videos(p, run_id):
    root = Path(p["data_root"])
    archive = hf_hub_download(
        repo_id=p["dataset_name"], filename=p["dataset_file"], repo_type="dataset",
        cache_dir=str(root / "hub"),
    )
    extracted = root / run_id
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extracted, filter="data")
    # Ignore the archive's clip-level partitions and rebuild splits by original video group.
    return split_videos(
        extracted.rglob("*.avi"), p["seed"], p["validation_fraction"], p["test_fraction"]
    )


def sample_frames(path, count):
    """Count decoded frames before sampling; AVI metadata can overreport their number."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        total = sum(1 for _ in container.decode(stream))
    if total < 1:
        raise ValueError(f"Empty video: {path}")
    indices = torch.linspace(0, total - 1, count).long().tolist()
    wanted = set(indices)
    frames = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames[index] = frame.to_image()
            if index == indices[-1]:
                break
    if frames.keys() != wanted:
        raise ValueError(f"Video frame count changed between decoding passes: {path}")
    return [frames[index] for index in indices]


def encode_videos(manifest, p, device):
    encoder = timm.create_model(p["model_name"], pretrained=True, num_classes=0)
    encoder = encoder.to(device).eval().requires_grad_(False)
    preprocess = timm.data.create_transform(**timm.data.resolve_model_data_config(encoder), is_training=False)
    features = []
    # The frozen encoder has no statistics fitted on this dataset. Cache its features in RAM.
    with torch.inference_mode():
        for i, path in enumerate(manifest["path"]):
            frames = sample_frames(path, p["frames_per_video"])
            batch = torch.stack([preprocess(frame) for frame in frames])
            features.append(encoder(batch.to(device)).cpu())
            if (i + 1) % 50 == 0 or i + 1 == len(manifest):
                print(f"Encoded {i + 1}/{len(manifest)} videos", flush=True)
    return torch.stack(features)


class TemporalClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, features):
        _, hidden = self.gru(features)
        return self.classifier(hidden[-1])


def predict(model, loader, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for features, _ in loader:
            outputs.append(model(features.to(device)).softmax(dim=1).cpu())
    return torch.cat(outputs).numpy()


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    for name in ("frames_per_video", "hidden_size", "epochs", "batch_size", "cpu_threads"):
        if not isinstance(p[name], int) or p[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU")
    torch.manual_seed(p["seed"])
    torch.set_num_threads(p["cpu_threads"])
    device = torch.device("cuda")
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_videos(p, run_id)
    class_names = sorted(manifest["label"].unique().tolist())
    label_to_id = {name: i for i, name in enumerate(class_names)}
    labels = torch.tensor(manifest["label"].map(label_to_id).to_numpy())
    manifest.drop(columns="path").to_csv(output_dir / "split_manifest.csv", index=False)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(p)
        mlflow.log_params({f"{split}_videos": int((manifest["split"] == split).sum())
                           for split in ("train", "validation", "test")})
        started = time.perf_counter()
        features = encode_videos(manifest, p, device)
        feature_seconds = time.perf_counter() - started
        if not torch.isfinite(features).all():
            raise ValueError("Frame encoder returned non-finite features")
        loaders = {}
        for split in ("train", "validation", "test"):
            indices = torch.tensor(np.flatnonzero(manifest["split"].eq(split)))
            loaders[split] = DataLoader(
                TensorDataset(features[indices], labels[indices]), batch_size=p["batch_size"],
                shuffle=split == "train", generator=torch.Generator().manual_seed(p["seed"]),
            )
        config = {"input_size": features.shape[-1], "hidden_size": p["hidden_size"],
                  "num_classes": len(class_names)}
        model = TemporalClassifier(**config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"]
        )
        checkpoint = output_dir / "classifier.pt"
        best_accuracy, best_epoch = -1.0, 0
        validation_labels = labels[manifest["split"].eq("validation").to_numpy()].numpy()
        started = time.perf_counter()
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            loss_sum = 0.0
            for batch, target in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(batch.to(device)), target.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(target)
            validation_predictions = predict(model, loaders["validation"], device).argmax(axis=1)
            accuracy = float(accuracy_score(validation_labels, validation_predictions))
            metrics = {"train_loss": loss_sum / len(loaders["train"].dataset),
                       "validation_accuracy": accuracy}
            mlflow.log_metrics(metrics, step=epoch)
            if accuracy > best_accuracy:
                best_accuracy, best_epoch = accuracy, epoch
                torch.save({"state_dict": model.state_dict(), "model_config": config,
                            "class_names": class_names, "parameters": p}, checkpoint)
            print(f"Epoch {epoch}/{p['epochs']}: {metrics}", flush=True)

        # Select on validation accuracy; evaluate the chosen checkpoint on test data once.
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        probabilities = predict(model, loaders["test"], device)
        predictions = probabilities.argmax(axis=1)
        test = manifest.loc[manifest["split"].eq("test"), ["video", "label", "group"]].copy()
        test_labels = test["label"].map(label_to_id).to_numpy()
        metrics = {
            "test_accuracy": float(accuracy_score(test_labels, predictions)),
            "test_f1_macro": float(f1_score(test_labels, predictions, labels=range(len(class_names)),
                                           average="macro", zero_division=0)),
            "best_validation_accuracy": best_accuracy, "best_epoch": best_epoch,
            "feature_extraction_seconds": feature_seconds,
            "train_and_eval_seconds": time.perf_counter() - started,
        }
        test = test.rename(columns={"label": "true_label"})
        test["predicted_label"] = [class_names[i] for i in predictions]
        test["confidence"] = probabilities.max(axis=1)
        test.to_csv(output_dir / "test_predictions.csv", index=False)
        pd.DataFrame(
            confusion_matrix(test_labels, predictions, labels=range(len(class_names))),
            index=class_names, columns=class_names,
        ).rename_axis("true_label").to_csv(output_dir / "confusion_matrix.csv")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir), artifact_path="video_classifier")
        print(f"Test metrics: {metrics}\nArtifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
