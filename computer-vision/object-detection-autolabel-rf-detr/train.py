#!/usr/bin/env python3
"""Auto-label unlabeled training images with an open-vocabulary detector, then fine-tune RF-DETR on the pseudo-labels.

The human-labeled validation split chooses the auto-labeler's confidence threshold and RF-DETR's best checkpoint.
The human-labeled test split is used only for the reported numbers.
"""

import json
import os
import shutil
import time
from pathlib import Path

import mlflow
import torch
import yaml
from faster_coco_eval import COCO
from faster_coco_eval.core.faster_eval_api import COCOeval
from huggingface_hub import snapshot_download
from PIL import Image
from rfdetr import RFDETR, RFDETRNano, RFDETRSmall
from torchvision.ops import batched_nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

RFDETR_VARIANTS = {"nano": RFDETRNano, "small": RFDETRSmall}
# Hugging Face split name -> RF-DETR split directory name.
HF_SPLITS = ("train", "validation", "test")


def load_parameters():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    # Small floats may arrive as "1e-05", which PyYAML parses as a string.
    p["learning_rate"] = float(p["learning_rate"])
    p["nms_iou"] = float(p["nms_iou"])
    p["threshold_grid"] = [float(value) for value in p["threshold_grid"]]
    if p["rfdetr_model"] not in RFDETR_VARIANTS:
        raise ValueError(f"rfdetr_model must be one of {sorted(RFDETR_VARIANTS)}")
    if not all(0.0 < value < 1.0 for value in p["threshold_grid"]):
        raise ValueError("threshold_grid values must be in (0, 1)")
    return p


def read_split(root, split):
    """Read one Hugging Face split. Category index c maps to COCO category id c + 1."""
    records = []
    with open(root / split / "metadata.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            objects = row.get("objects") or {}
            records.append(
                {
                    "file_name": row["file_name"],
                    "width": row["width"],
                    "height": row["height"],
                    "boxes": objects.get("bbox", []),
                    "categories": [int(c) + 1 for c in objects.get("category", [])],
                }
            )
    return records


def coco_ground_truth(records, class_names):
    images, annotations = [], []
    for image_id, record in enumerate(records, start=1):
        images.append(
            {"id": image_id, "file_name": record["file_name"], "width": record["width"], "height": record["height"]}
        )
        for box, category_id in zip(record["boxes"], record["categories"]):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": box,
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                }
            )
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": categories}


def run_labeler(records, image_dir, p, detector, processor, device):
    """Return raw detections per image above the lowest candidate threshold, before NMS."""
    class_names = p["class_names"]
    floor = min(p["threshold_grid"])
    raw = {}
    batch_size = p["detector_batch_size"]
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [Image.open(image_dir / record["file_name"]).convert("RGB") for record in batch]
        inputs = processor(text=[class_names] * len(images), images=images, return_tensors="pt").to(device)
        # bf16 halves inference time on the A10; box decoding below runs on float32 copies.
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = detector(**inputs)
        target_sizes = torch.tensor([[image.height, image.width] for image in images], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=floor,
            target_sizes=target_sizes,
            text_labels=[class_names] * len(images),
        )
        for record, result in zip(batch, results):
            raw[record["file_name"]] = (
                result["boxes"].float().cpu().numpy(),
                result["scores"].float().cpu().numpy(),
                result["labels"].cpu().numpy(),
            )
        for image in images:
            image.close()
    return raw


def select_detections(raw_entry, threshold, nms_iou):
    """Apply a confidence threshold, then per-class NMS. Boxes stay in xyxy pixel coordinates."""
    boxes, scores, labels = raw_entry
    keep = scores >= threshold
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    if len(scores):
        order = batched_nms(
            torch.from_numpy(boxes), torch.from_numpy(scores), torch.from_numpy(labels), nms_iou
        ).numpy()
        boxes, scores, labels = boxes[order], scores[order], labels[order]
    return boxes, scores, labels


def labeler_predictions(records, raw, threshold, nms_iou):
    """Convert the detector's output to COCO result dicts. Image ids follow the order of `records`."""
    predictions = []
    for image_id, record in enumerate(records, start=1):
        boxes, scores, labels = select_detections(raw[record["file_name"]], threshold, nms_iou)
        for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(score),
                }
            )
    return predictions


def rfdetr_predictions(model, records, image_dir, threshold=0.01):
    predictions = []
    for image_id, record in enumerate(records, start=1):
        image = Image.open(image_dir / record["file_name"]).convert("RGB")
        detections = model.predict(image, threshold=threshold)
        image.close()
        for (x0, y0, x1, y1), score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(score),
                }
            )
    return predictions


def coco_map(ground_truth_path, predictions, image_ids):
    """COCO box mAP@[.50:.95] and mAP@.50 using faster-coco-eval, the backend RF-DETR uses for its own metrics."""
    if not predictions:
        return {"map50_95": 0.0, "map50": 0.0}
    ground_truth = COCO(str(ground_truth_path))
    evaluator = COCOeval(ground_truth, iouType="bbox")
    evaluator.cocoDt = ground_truth.loadRes(predictions)
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {"map50_95": float(evaluator.stats[0]), "map50": float(evaluator.stats[1])}


def stage_dataset(dataset_dir, splits):
    """Lay out an RF-DETR COCO dataset. `splits` maps RF-DETR split name to (coco dict, source image dir).

    Images are symlinked from the Hugging Face snapshot rather than copied.
    """
    for split_name, (coco, source_dir) in splits.items():
        split_dir = dataset_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for image in coco["images"]:
            os.symlink(source_dir / image["file_name"], split_dir / image["file_name"])
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco))
    return dataset_dir


def train_rfdetr(p, dataset_dir, output_dir):
    model = RFDETR_VARIANTS[p["rfdetr_model"]]()
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        grad_accum_steps=1,
        lr=p["learning_rate"],
        seed=p["seed"],
        class_names=p["class_names"],
        tensorboard=False,
        run_test=False,
    )
    checkpoint = output_dir / "checkpoint_best_total.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"RF-DETR did not write {checkpoint}")
    return checkpoint


def log_to_air_run(run_id, p, metrics, artifact_paths):
    """Log metrics and artifacts to the AIR-created MLflow run."""
    flat = {
        "labeler_threshold": metrics["labeler_threshold"],
        "labeler_test_map50_95": metrics["labeler_test"]["map50_95"],
        "labeler_test_map50": metrics["labeler_test"]["map50"],
        "labeler_validation_map50_95": metrics["labeler_threshold_sweep"][str(metrics["labeler_threshold"])][
            "map50_95"
        ],
        "pseudo_label_count": metrics["pseudo_label_count"],
        "duration_seconds": metrics["duration_seconds"],
    }
    for name, result in metrics["rfdetr_test"].items():
        flat[f"{name}_test_map50_95"] = result["map50_95"]
        flat[f"{name}_test_map50"] = result["map50"]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({key: str(value) for key, value in p.items()})
        mlflow.log_metrics(flat)
        for path in artifact_paths:
            mlflow.log_artifact(str(path))


def main():
    p = load_parameters()
    run_id = os.environ["MLFLOW_RUN_ID"]
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.manual_seed(p["seed"])
    class_names = p["class_names"]

    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(p["data_root"]) / run_id
    scratch.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(
        snapshot_download(
            repo_id=p["dataset_repo"],
            repo_type="dataset",
            revision=p["dataset_revision"],
            allow_patterns=[f"{split}/*" for split in HF_SPLITS],
            local_dir=str(scratch / "hf_dataset"),
        )
    )
    splits = {split: read_split(dataset_root, split) for split in HF_SPLITS}
    image_dirs = {split: dataset_root / split for split in HF_SPLITS}

    # Human labels are read only for validation and test, plus train when the upper-bound option is on.
    ground_truth_paths = {}
    for split in ("validation", "test"):
        path = scratch / f"gt_{split}.json"
        path.write_text(json.dumps(coco_ground_truth(splits[split], class_names)))
        ground_truth_paths[split] = path
    validation_image_ids = list(range(1, len(splits["validation"]) + 1))
    test_image_ids = list(range(1, len(splits["test"]) + 1))

    # Run the auto-labeler once per split, then select the threshold on validation only.
    processor = Owlv2Processor.from_pretrained(p["detector_model"], revision=p["detector_revision"])
    detector = (
        Owlv2ForObjectDetection.from_pretrained(p["detector_model"], revision=p["detector_revision"])
        .to(device)
        .eval()
    )
    raw = {
        split: run_labeler(splits[split], image_dirs[split], p, detector, processor, device)
        for split in ("validation", "test", "train")
    }
    del detector
    torch.cuda.empty_cache()

    threshold_sweep = {}
    for threshold in p["threshold_grid"]:
        predictions = labeler_predictions(splits["validation"], raw["validation"], threshold, p["nms_iou"])
        threshold_sweep[str(threshold)] = coco_map(
            ground_truth_paths["validation"], predictions, validation_image_ids
        )
    best_threshold = max(p["threshold_grid"], key=lambda value: threshold_sweep[str(value)]["map50_95"])
    print(f"Validation sweep: {threshold_sweep}\nSelected threshold: {best_threshold}", flush=True)

    labeler_test_predictions = labeler_predictions(splits["test"], raw["test"], best_threshold, p["nms_iou"])
    labeler_test = coco_map(ground_truth_paths["test"], labeler_test_predictions, test_image_ids)

    # Pseudo-labels for the training split. Human train labels are not read here.
    pseudo_coco = coco_ground_truth([], class_names)
    for image_id, record in enumerate(splits["train"], start=1):
        boxes, scores, labels = select_detections(raw["train"][record["file_name"]], best_threshold, p["nms_iou"])
        pseudo_coco["images"].append(
            {"id": image_id, "file_name": record["file_name"], "width": record["width"], "height": record["height"]}
        )
        for (x0, y0, x1, y1), score, label in zip(boxes, scores, labels):
            pseudo_coco["annotations"].append(
                {
                    "id": len(pseudo_coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "iscrowd": 0,
                    "score": float(score),
                }
            )
    pseudo_label_count = len(pseudo_coco["annotations"])
    pseudo_path = output_dir / "pseudo_labels_train.json"
    pseudo_path.write_text(json.dumps(pseudo_coco, indent=2) + "\n")
    # RF-DETR trains on standard COCO annotations, so drop the score field used only for the saved file.
    training_coco = json.loads(json.dumps(pseudo_coco))
    for annotation in training_coco["annotations"]:
        annotation.pop("score")

    # Validation is the human-labeled split; RF-DETR uses it for best-checkpoint selection only.
    validation_coco = coco_ground_truth(splits["validation"], class_names)
    runs = {"pseudo_label": training_coco}
    if p["train_on_human_labels"]:
        runs["human_label"] = coco_ground_truth(splits["train"], class_names)

    rfdetr_test = {}
    for mode, train_coco in runs.items():
        dataset_dir = stage_dataset(
            scratch / "rfdetr" / mode,
            {
                "train": (train_coco, image_dirs["train"]),
                "valid": (validation_coco, image_dirs["validation"]),
            },
        )
        checkpoint = train_rfdetr(p, dataset_dir, scratch / "runs" / mode)
        model = RFDETR.from_checkpoint(checkpoint)
        test_predictions = rfdetr_predictions(model, splits["test"], image_dirs["test"])
        rfdetr_test[mode] = coco_map(ground_truth_paths["test"], test_predictions, test_image_ids)
        shutil.copy2(checkpoint, output_dir / f"rfdetr_{mode}_best.pth")
        del model
        torch.cuda.empty_cache()

    metrics = {
        "labeler_model": p["detector_model"],
        "labeler_threshold": best_threshold,
        "labeler_threshold_sweep": threshold_sweep,
        "labeler_test": labeler_test,
        "pseudo_label_count": pseudo_label_count,
        "pseudo_label_images": len(splits["train"]),
        "rfdetr_model": f"RFDETR{p['rfdetr_model'].capitalize()}",
        "rfdetr_test": rfdetr_test,
        "duration_seconds": time.perf_counter() - started,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Metrics: {json.dumps(metrics, indent=2)}\nArtifacts: {output_dir}", flush=True)

    log_to_air_run(run_id, p, metrics, [metrics_path, pseudo_path] + [output_dir / f"rfdetr_{m}_best.pth" for m in runs])


if __name__ == "__main__":
    main()
