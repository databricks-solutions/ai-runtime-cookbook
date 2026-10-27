#!/usr/bin/env python3
"""Classify images with ViT on one GPU and write an annotated JPEG for each one."""

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from datasets import Image as ImageFeature
from datasets import load_dataset
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


class EncodedImages(Dataset):
    """Decode and preprocess one image per item so loader workers run in parallel."""

    def __init__(self, encoded, processor):
        self.encoded = encoded
        self.processor = processor

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.encoded[index])).convert("RGB")
        return self.processor(image, return_tensors="pt")["pixel_values"][0], index


def annotate(encoded, label, score, destination, quality):
    """Draw the prediction onto the original image and save it as a JPEG."""
    image = Image.open(io.BytesIO(encoded)).convert("RGB")
    text = f"{label} ({score:.1%})"
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((0, 0), text)
    draw.rectangle([(2, 2), (box[2] - box[0] + 10, box[3] - box[1] + 10)], fill="black")
    draw.text((6, 4), text, fill="white")
    image.save(destination, "JPEG", quality=quality)


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    for name in ("batch_size", "loader_workers", "write_workers", "jpeg_quality"):
        if not isinstance(p[name], int) or p[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(p["max_images"], int) or p["max_images"] < 0:
        raise ValueError("max_images must be 0 (the whole split) or a positive integer")
    if p["jpeg_quality"] > 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU")

    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    annotated_dir = output_dir / "annotated"
    # Each attempt gets its own run ID, so a rerun never overwrites earlier output.
    annotated_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(p)

        started = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(p["model_name"])
        model = AutoModelForImageClassification.from_pretrained(p["model_name"])
        model = model.to("cuda").eval()
        id2label = model.config.id2label
        model_load_seconds = time.perf_counter() - started

        # Keep the encoded bytes so the write phase annotates the original image.
        started = time.perf_counter()
        split = load_dataset(p["dataset_name"], split=p["dataset_split"])
        if p["max_images"]:
            split = split.select(range(min(p["max_images"], len(split))))
        split = split.cast_column("image", ImageFeature(decode=False))
        encoded = [item["bytes"] for item in split["image"]]
        image_ids = [str(value) for value in split["image_id"]]
        num_images = len(encoded)
        if not num_images:
            raise ValueError("The split contains no images")
        data_load_seconds = time.perf_counter() - started

        loader = DataLoader(
            EncodedImages(encoded, processor),
            batch_size=p["batch_size"],
            shuffle=False,
            num_workers=p["loader_workers"],
            pin_memory=True,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        batched_logits, batched_indexes = [], []
        with torch.no_grad():
            for pixel_values, indexes in loader:
                pixel_values = pixel_values.to("cuda", non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(pixel_values=pixel_values).logits
                batched_logits.append(logits.float().cpu())
                batched_indexes.append(indexes)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started

        logits = torch.cat(batched_logits)[torch.cat(batched_indexes).argsort()]
        if not torch.isfinite(logits).all():
            raise ValueError("The model returned non-finite logits")
        scores, predictions = torch.softmax(logits, dim=-1).max(dim=-1)
        labels = [id2label[index] for index in predictions.tolist()]
        scores = scores.tolist()

        # The GPU is idle for this whole phase, so its duration is billed GPU time
        # spent on object-store writes. Threads overlap those writes.
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=p["write_workers"]) as pool:
            list(
                pool.map(
                    annotate,
                    encoded,
                    labels,
                    scores,
                    [annotated_dir / f"{index:06d}.jpg" for index in range(num_images)],
                    [p["jpeg_quality"]] * num_images,
                )
            )
        write_seconds = time.perf_counter() - started

        written = sum(1 for _ in annotated_dir.glob("*.jpg"))
        if written != num_images:
            raise RuntimeError(f"Wrote {written} annotated images, expected {num_images}")

        pq.write_table(
            pa.table({
                "index": list(range(num_images)),
                "image_id": image_ids,
                "label": labels,
                "score": scores,
                "annotated_path": [f"annotated/{index:06d}.jpg" for index in range(num_images)],
            }),
            output_dir / "predictions.parquet",
        )
        total_seconds = model_load_seconds + data_load_seconds + inference_seconds + write_seconds
        metrics = {
            "images": num_images,
            "inference_seconds": inference_seconds,
            "write_seconds": write_seconds,
            "total_seconds": total_seconds,
            "images_per_second": num_images / total_seconds,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (output_dir / "parameters.json").write_text(json.dumps(p, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        for name in ("metrics.json", "parameters.json", "predictions.parquet"):
            mlflow.log_artifact(str(output_dir / name), artifact_path="classification")
        # A few annotated images make the output reviewable from the MLflow run.
        for index in range(min(8, num_images)):
            mlflow.log_artifact(str(annotated_dir / f"{index:06d}.jpg"), artifact_path="samples")
        print(f"Metrics: {metrics}\nAnnotated images: {annotated_dir}", flush=True)


if __name__ == "__main__":
    main()
