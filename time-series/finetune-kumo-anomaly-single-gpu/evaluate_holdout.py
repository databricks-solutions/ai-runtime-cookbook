"""Score the held-out test set with the base and fine-tuned checkpoints.

The last 20% of the Kaggle series is never used for training or validation. It
keeps its original labels, so it works as a labeled test set. This script runs
anomaly detection on it with each checkpoint and reports average precision (a
threshold-free ranking metric) plus precision, recall, and F1. Scoring the same
rows with the base and fine-tuned weights shows the before/after effect of
fine-tuning.

Run from the upstream Kumo-Anomaly directory so that `sdk` is importable:
    uv run --frozen --no-dev python /path/to/evaluate_holdout.py \
        --source <dir with TimeSeries.csv and labelsTimeSeries.csv> \
        --manifest <work>/data/manifest.json \
        --finetuned-model <work>/checkpoints/best_finetuned_model.pth \
        --finetuned-config <work>/checkpoints/finetune_config.yaml \
        --base-model <work>/weights/final_model.pth \
        --base-config <work>/weights/curriculum_medium.yaml \
        --output <work>/evaluation.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# The upstream anomaly-detection SDK lives in the Kumo-Anomaly working directory.
sys.path.insert(0, os.getcwd())
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion  # noqa: E402

FEATURE_COLUMNS = [f"v{i}" for i in range(1, 12)]


def load_holdout(source, manifest_path, eval_rows):
    """Return the held-out features and labels after the 80% boundary."""
    manifest = json.loads(Path(manifest_path).read_text())
    # temporal_boundaries is [0, train_end (60%), val_end (80%)]; rows after the
    # last boundary are the held-out test set.
    boundary = int(manifest["temporal_boundaries"][2])
    x = pd.read_csv(Path(source) / "TimeSeries.csv")
    y = pd.read_csv(Path(source) / "labelsTimeSeries.csv")
    if list(x.columns) != FEATURE_COLUMNS or list(y.columns) != ["label"]:
        raise ValueError(
            "Expected the Kaggle TimeSeries.csv (v1..v11) and labelsTimeSeries.csv (label)."
        )
    features = x.iloc[boundary:].reset_index(drop=True)
    labels = y.label.to_numpy()[boundary:]
    if eval_rows:
        features = features.iloc[:eval_rows].reset_index(drop=True)
        labels = labels[:eval_rows]
    return features, labels


def evaluate(features, labels, model_path, config_path, nsample):
    """Score one checkpoint and return ranking and threshold metrics."""
    results = perform_anomaly_analysis_with_diffusion(
        df=features,
        threshold_strategy="scs",
        model_path=model_path,
        model_config_path=config_path,
        nsample=nsample,
    )
    scores = results["MAE"].to_numpy()
    predictions = results["Anomaly"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--finetuned-model", required=True, type=Path)
    parser.add_argument("--finetuned-config", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, help="Optional base checkpoint for a before/after comparison.")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--eval-rows", type=int, default=0, help="Held-out rows to score; 0 uses the whole test set.")
    parser.add_argument("--nsample", type=int, default=15)
    args = parser.parse_args()
    if args.eval_rows < 0:
        parser.error("--eval-rows must be nonnegative; 0 uses the whole test set.")
    if bool(args.base_model) != bool(args.base_config):
        parser.error("Pass --base-model and --base-config together, or neither.")

    features, labels = load_holdout(args.source, args.manifest, args.eval_rows)
    if labels.sum() == 0:
        raise ValueError(
            "The held-out evaluation slice has no labeled anomalies; increase --eval-rows."
        )

    checkpoints = {"finetuned": (args.finetuned_model, args.finetuned_config)}
    if args.base_model:
        checkpoints["base"] = (args.base_model, args.base_config)

    summary = {
        "eval_rows": int(len(labels)),
        "positives": int(labels.sum()),
        "nsample": args.nsample,
        "models": {},
    }
    for name, (model_path, config_path) in checkpoints.items():
        print(f"Scoring {name} checkpoint: {model_path}", flush=True)
        summary["models"][name] = evaluate(
            features, labels, str(model_path), str(config_path), args.nsample
        )

    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
