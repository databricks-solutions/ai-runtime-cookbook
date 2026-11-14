#!/usr/bin/env python3
"""Research-only chest X-ray inference over DICOM files with Ray Data.

The input directory must contain single-frame, grayscale DX or CR DICOM files.
Patient names and IDs are never copied to the output. The model scores are
uncalibrated research outputs, not diagnoses or triage decisions.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice

import cv2
import numpy as np
import pydicom
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchxrayvision as xrv
from databricks.air.data import DataLoader, UCVolumeDataset
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from torch.nn import functional as F
from torch.utils.data import IterableDataset

MODEL_WEIGHTS = "densenet121-res224-all"
MODEL_CACHE_DIR = os.environ.get("XRV_CACHE_DIR", "/opt/torchxrayvision")
ALLOWED_MODALITIES = {"CR", "DX"}
ALLOWED_PHOTOMETRIC_INTERPRETATIONS = {"MONOCHROME1", "MONOCHROME2"}
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
DATALOADER_WORKERS = int(os.environ.get("DATALOADER_WORKERS", "12"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "32"))
USE_AMP = os.environ.get("USE_AMP", "true").lower() == "true"
WARMUP_PASS = os.environ.get("WARMUP_PASS", "false").lower() == "true"
INFERENCE_REPEATS = int(os.environ.get("INFERENCE_REPEATS", "1"))
INFERENCE_WORKERS = int(os.environ.get("INFERENCE_WORKERS", "2"))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "true").lower() == "true"
CHANNELS_LAST = os.environ.get("CHANNELS_LAST", "true").lower() == "true"
GPU_METRICS_MARKERS = os.environ.get("GPU_METRICS_MARKERS")
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "0"))
INPUT_PATH = os.environ.get(
    "INPUT_PATH",
    "/Volumes/main/default/air_examples/dicom-batch-inference-torchxrayvision/input",
)
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    "/Volumes/main/default/air_examples/dicom-batch-inference-torchxrayvision/predictions",
)
RUN_OUTPUT_PATH = os.path.join(OUTPUT_ROOT, os.environ.get("MLFLOW_RUN_ID") or "local")
METADATA_FIELDS = [
    "sop_instance_uid",
    "study_instance_uid",
    "series_instance_uid",
    "modality",
    "manufacturer",
    "manufacturer_model_name",
    "original_rows",
    "original_columns",
]
CENTER_CROP = xrv.datasets.XRayCenterCrop()
RESIZER = xrv.datasets.XRayResizer(224, engine="cv2")
# Let Ray provide the parallelism; a thread pool per task would oversubscribe CPUs.
cv2.setNumThreads(1)


def score_column(pathology):
    """Convert a TorchXRayVision pathology name to a stable output column."""
    return "score_" + re.sub(r"[^a-z0-9]+", "_", pathology.lower()).strip("_")


def prepare_dicom(row, resize=True, gpu_preprocess=False):
    """Decode one DICOM file and apply the model's expected preprocessing."""
    try:
        with BytesIO(row["bytes"]) as stream:
            dataset = pydicom.dcmread(stream, force=False)
            modality = str(dataset.Modality).upper()
            if modality not in ALLOWED_MODALITIES:
                raise ValueError(f"expected DX or CR modality, found {modality}")
            if int(getattr(dataset, "NumberOfFrames", 1)) != 1:
                raise ValueError("multiframe DICOM files are not supported")
            if int(getattr(dataset, "SamplesPerPixel", 1)) != 1:
                raise ValueError("color DICOM files are not supported")

            photometric = str(dataset.PhotometricInterpretation).upper()
            if photometric not in ALLOWED_PHOTOMETRIC_INTERPRETATIONS:
                raise ValueError(f"unsupported photometric interpretation: {photometric}")

            preprocess = None
            if gpu_preprocess:
                voi_function = str(getattr(dataset, "VOILUTFunction", "")).upper()
                use_cpu_lut = "VOILUTSequence" in dataset or voi_function not in ("", "LINEAR")
                try:
                    preprocess = {
                        "rescale_slope": float(getattr(dataset, "RescaleSlope", 1.0)),
                        "rescale_intercept": float(getattr(dataset, "RescaleIntercept", 0.0)),
                        "window_center": float(dataset.WindowCenter) if "WindowCenter" in dataset else None,
                        "window_width": float(dataset.WindowWidth) if "WindowWidth" in dataset else None,
                        "monochrome1": photometric == "MONOCHROME1",
                    }
                except (TypeError, ValueError):
                    use_cpu_lut = True
                if use_cpu_lut:
                    pixels = apply_modality_lut(dataset.pixel_array, dataset)
                    pixels = apply_voi_lut(pixels, dataset)
                    preprocess = {
                        "rescale_slope": 1.0,
                        "rescale_intercept": 0.0,
                        "window_center": None,
                        "window_width": None,
                        "monochrome1": photometric == "MONOCHROME1",
                    }
                else:
                    pixels = dataset.pixel_array
            else:
                pixels = apply_modality_lut(dataset.pixel_array, dataset)
                pixels = apply_voi_lut(pixels, dataset)
            pixels = np.asarray(pixels, dtype=np.float32)
            if pixels.ndim != 2 or not np.isfinite(pixels).all():
                raise ValueError("pixel data must be a finite single-channel image")

            if gpu_preprocess:
                image = CENTER_CROP(pixels[None, ...])
            else:
                # Respect the DICOM VOI/window, then map the remaining display
                # range to the 8-bit range expected by TorchXRayVision.
                minimum = float(pixels.min())
                maximum = float(pixels.max())
                if maximum <= minimum:
                    raise ValueError("pixel data has no dynamic range")
                pixels = (pixels - minimum) * (255.0 / (maximum - minimum))
                if photometric == "MONOCHROME1":
                    pixels = 255.0 - pixels
                # VOI LUT interpolation and float32 rounding can slightly exceed
                # the 8-bit bounds that TorchXRayVision validates.
                pixels = np.clip(pixels, 0.0, 255.0)
                pixels = xrv.datasets.normalize(pixels, 255)
                image = CENTER_CROP(pixels[None, ...])
            if resize and not gpu_preprocess:
                image = RESIZER(image).astype(np.float32)
            metadata = {
                "sop_instance_uid": str(dataset.SOPInstanceUID),
                "study_instance_uid": str(dataset.StudyInstanceUID),
                "series_instance_uid": str(dataset.SeriesInstanceUID),
                "modality": modality,
                "manufacturer": str(getattr(dataset, "Manufacturer", "")),
                "manufacturer_model_name": str(getattr(dataset, "ManufacturerModelName", "")),
                "original_rows": int(dataset.Rows),
                "original_columns": int(dataset.Columns),
            }
            result = {**metadata, "image": image}
            if preprocess is not None:
                result["preprocess"] = preprocess
            return result
    except Exception as error:
        raise ValueError(f"Failed to prepare DICOM file {row.get('path', '<unknown>')}: {error}") from error


def preprocess_cpu(image, preprocess):
    """Finish one raw DICOM image on CPU for mixed-size fallback batches."""
    pixels = image.astype(np.float32, copy=False)
    pixels = pixels * preprocess["rescale_slope"] + preprocess["rescale_intercept"]
    if preprocess["window_center"] is not None:
        lower = preprocess["window_center"] - (preprocess["window_width"] - 1) / 2
        upper = preprocess["window_center"] + (preprocess["window_width"] - 1) / 2
        pixels = np.clip(pixels, lower, upper)
    minimum = float(pixels.min())
    maximum = float(pixels.max())
    if maximum <= minimum:
        raise ValueError("pixel data has no dynamic range")
    pixels = (pixels - minimum) * (255.0 / (maximum - minimum))
    if preprocess["monochrome1"]:
        pixels = 255.0 - pixels
    pixels = np.clip(pixels, 0.0, 255.0)
    pixels = xrv.datasets.normalize(pixels, 255)
    return RESIZER(pixels).astype(np.float32)


def preprocess_gpu(inputs, preprocess):
    """Apply standard DICOM scaling and normalization as a GPU batch."""
    def column(key):
        values = [float("nan") if item[key] is None else item[key] for item in preprocess]
        return torch.tensor(values, dtype=torch.float32, device=inputs.device).view(-1, 1, 1, 1)

    inputs = inputs.to(dtype=torch.float32)
    inputs = inputs * column("rescale_slope") + column("rescale_intercept")
    center = column("window_center")
    width = column("window_width")
    has_window = ~torch.isnan(center)
    lower = center - (width - 1) / 2
    upper = center + (width - 1) / 2
    lower = torch.where(has_window, lower, torch.full_like(lower, -torch.inf))
    upper = torch.where(has_window, upper, torch.full_like(upper, torch.inf))
    inputs = torch.minimum(torch.maximum(inputs, lower), upper)

    minimum = inputs.amin(dim=(1, 2, 3), keepdim=True)
    maximum = inputs.amax(dim=(1, 2, 3), keepdim=True)
    dynamic_range = maximum - minimum
    if torch.any(dynamic_range <= 0):
        raise ValueError("pixel data has no dynamic range")
    inputs = (inputs - minimum) * (255.0 / dynamic_range)
    monochrome1 = torch.tensor(
        [item["monochrome1"] for item in preprocess],
        dtype=torch.bool,
        device=inputs.device,
    ).view(-1, 1, 1, 1)
    inputs = torch.where(monochrome1, 255.0 - inputs, inputs)
    inputs = torch.clamp(inputs, 0.0, 255.0)
    return (inputs / 127.5 - 1.0) * 1024.0


class DicomPathDataset(IterableDataset):
    """Decode each locally cached path produced by UCVolumeDataset."""

    def __init__(self, path_dataset, max_images=0):
        self.path_dataset = path_dataset
        self.max_images = max_images

    def __iter__(self):
        paths = self.path_dataset
        if self.max_images > 0:
            paths = islice(paths, self.max_images)
        for local_path in paths:
            with open(local_path, "rb") as handle:
                yield prepare_dicom(
                    {"path": local_path, "bytes": handle.read()},
                    resize=False,
                    gpu_preprocess=True,
                )


def collate(rows):
    """Build a pinned-memory-compatible tensor batch plus metadata columns."""
    batch = {field: [row[field] for row in rows] for field in METADATA_FIELDS}
    shapes = {row["image"].shape for row in rows}
    batch["resize_on_gpu"] = len(shapes) == 1
    if batch["resize_on_gpu"]:
        images = [row["image"] for row in rows]
    else:
        images = [preprocess_cpu(row["image"], row["preprocess"]) for row in rows]
    batch["image"] = torch.from_numpy(np.stack(images))
    batch["preprocess"] = [row["preprocess"] for row in rows]
    return batch


class ChestXRayPredictor:
    """Load one model copy on the GPU and run batched inference."""

    def __init__(self):
        self.device = torch.device("cuda")
        self.stream = torch.cuda.Stream()
        model = xrv.models.DenseNet(
            weights=MODEL_WEIGHTS,
            cache_dir=MODEL_CACHE_DIR,
        ).to(self.device).eval()
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        self.pathologies = list(model.pathologies)
        self.model = model

    def __call__(self, batch):
        with torch.cuda.stream(self.stream):
            if isinstance(batch["image"], torch.Tensor):
                images = batch["image"]
                inputs = images.to(self.device, non_blocking=True)
            else:
                images = np.stack(batch["image"]).astype(np.float32, copy=False)
                inputs = torch.from_numpy(images).to(self.device, non_blocking=True)
            if batch.get("preprocess") is not None:
                inputs = preprocess_gpu(inputs, batch["preprocess"])
            if batch.get("resize_on_gpu", False):
                inputs = F.interpolate(inputs, size=(224, 224), mode="area")
            if CHANNELS_LAST:
                inputs = inputs.to(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=USE_AMP
            ):
                scores = self.model(inputs).float().cpu().numpy()

            output = {field: batch[field] for field in METADATA_FIELDS}
            output["model_weights"] = np.asarray([MODEL_WEIGHTS] * len(images))
            for index, pathology in enumerate(self.pathologies):
                output[score_column(pathology)] = scores[:, index]
            return output


def check_volume(path, setting):
    """Fail early when a Unity Catalog volume is missing or inaccessible."""
    if not path.startswith("/Volumes/"):
        return
    parts = path.rstrip("/").split("/")
    if len(parts) < 5:
        raise SystemExit(f"{setting} is not a valid Unity Catalog volume path: {path}")
    volume_path = "/".join(parts[:5])
    if not os.path.isdir(volume_path):
        volume_name = ".".join(parts[2:5])
        raise SystemExit(
            f"{setting} uses volume {volume_name}, which does not exist or is not accessible. "
            "Choose an existing accessible volume in workload.yaml."
        )


def log_to_mlflow(params, metrics):
    """Attach workload metadata and throughput to AIR's existing MLflow run."""
    run_id = os.environ.get("MLFLOW_RUN_ID")
    if not run_id:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    for key, value in params.items():
        client.log_param(run_id, key, value)
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)


def main():
    check_volume(INPUT_PATH, "INPUT_PATH")
    check_volume(OUTPUT_ROOT, "OUTPUT_ROOT")
    if not os.path.isdir(INPUT_PATH):
        raise SystemExit(
            f"INPUT_PATH is not an accessible directory: {INPUT_PATH}. "
            "Upload de-identified DX/CR DICOM files or update workload.yaml."
        )
    os.makedirs(RUN_OUTPUT_PATH, exist_ok=True)

    started = time.time()
    dataset = DicomPathDataset(UCVolumeDataset(INPUT_PATH), max_images=MAX_IMAGES)
    workers = 0 if MAX_IMAGES > 0 else DATALOADER_WORKERS

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate,
        num_workers=workers,
        prefetch_factor=PREFETCH_FACTOR if workers > 0 else None,
        pin_memory=PIN_MEMORY,
    )
    predictors = [ChestXRayPredictor() for _ in range(INFERENCE_WORKERS)]

    def mark_phase(name):
        if GPU_METRICS_MARKERS:
            with open(GPU_METRICS_MARKERS, "a") as handle:
                handle.write(json.dumps({"phase": name, "time": time.time()}) + "\n")

    def run_pass(label, output_path=None, started=None):
        started = started or time.time()
        mark_phase(label.lower() + "_start")
        writer = None
        num_images = 0
        batches = 0
        try:
            with ThreadPoolExecutor(max_workers=len(predictors)) as inference_executor, ThreadPoolExecutor(
                max_workers=1
            ) as output_executor:
                inference_pending = []
                output_pending = []

                def write_result(result):
                    nonlocal writer
                    table = pa.Table.from_pydict(result)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                    writer.write_table(table)

                def submit_output(result):
                    if output_path is not None:
                        output_pending.append(output_executor.submit(write_result, result))
                        if len(output_pending) >= 2:
                            output_pending.pop(0).result()

                for batch in loader:
                    image_count = len(batch["image"])
                    if len(predictors) == 1:
                        submit_output(predictors[0](batch))
                    else:
                        future = inference_executor.submit(
                            predictors[len(inference_pending) % len(predictors)], batch
                        )
                        inference_pending.append((future, image_count))
                        if len(inference_pending) >= len(predictors):
                            result_future, count = inference_pending.pop(0)
                            submit_output(result_future.result())
                            num_images += count
                            batches += 1
                            if batches % 10 == 0:
                                rate = num_images / (time.time() - started)
                                print(f"{label}: {num_images} images at {rate:.3f} images/s", flush=True)
                    if len(predictors) == 1:
                        num_images += image_count
                        batches += 1
                        if batches % 10 == 0:
                            rate = num_images / (time.time() - started)
                            print(f"{label}: {num_images} images at {rate:.3f} images/s", flush=True)
                for result_future, count in inference_pending:
                    submit_output(result_future.result())
                    num_images += count
                    batches += 1
                for future in output_pending:
                    future.result()
        finally:
            if writer is not None:
                writer.close()
            mark_phase(label.lower() + "_end")
        return num_images, time.time() - started

    warmup_images = 0
    warmup_elapsed = 0.0
    if WARMUP_PASS:
        warmup_images, warmup_elapsed = run_pass("Warmup", started=started)
    output_path = os.path.join(RUN_OUTPUT_PATH, "predictions.parquet")
    for _ in range(INFERENCE_REPEATS):
        num_images, elapsed = run_pass("Inference", output_path)
    if num_images == 0:
        raise RuntimeError(f"No .dcm or .DCM files found under {INPUT_PATH}")

    metrics = {
        "images": num_images,
        "inference_wall_time_s": round(elapsed, 3),
        "inference_images_per_sec": round(num_images / elapsed, 3),
    }
    if WARMUP_PASS:
        metrics.update(
            {
                "warmup_images": warmup_images,
                "warmup_wall_time_s": round(warmup_elapsed, 3),
                "warmup_images_per_sec": round(warmup_images / warmup_elapsed, 3),
            }
        )
    log_to_mlflow(
        params={
            "model_weights": MODEL_WEIGHTS,
            "batch_size": BATCH_SIZE,
            "dataloader_workers": workers,
            "prefetch_factor": PREFETCH_FACTOR,
            "use_amp": USE_AMP,
            "channels_last": CHANNELS_LAST,
            "inference_workers": INFERENCE_WORKERS,
            "warmup_pass": WARMUP_PASS,
            "inference_repeats": INFERENCE_REPEATS,
            "num_gpus": 1,
            "output_path": RUN_OUTPUT_PATH,
        },
        metrics=metrics,
    )
    print(f"Wrote {num_images} predictions to {RUN_OUTPUT_PATH}", flush=True)
    print("METRICS " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
