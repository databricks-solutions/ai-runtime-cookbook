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

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# The upstream anomaly-detection SDK lives in the Kumo-Anomaly working directory.
sys.path.insert(0, os.getcwd())
from sdk.anomaly_analysis import perform_anomaly_analysis_with_diffusion  # noqa: E402

FEATURE_COLUMNS = [f"v{i}" for i in range(1, 12)]


def load_holdout(source, manifest_path):
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
    return features, labels


# Neighboring rows are correlated, so the bootstrap resamples 100-row blocks, not single rows.
BOOTSTRAP_BLOCK_ROWS = 100
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0


def evaluate(features, model_path, config_path, nsample):
    """Return each held-out row's anomaly score and predicted label for one checkpoint."""
    results = perform_anomaly_analysis_with_diffusion(
        df=features,
        threshold_strategy="scs",
        model_path=model_path,
        model_config_path=config_path,
        nsample=nsample,
    )
    return results["MAE"].to_numpy(), results["Anomaly"].to_numpy()


def score_metrics(labels, scores, predictions):
    """Average precision (threshold-free) and precision, recall and F1 at the predicted labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def block_bootstrap_rows(num_rows, rng):
    """Row indices for one resample, built from whole blocks drawn with replacement."""
    starts = np.arange(0, num_rows, BOOTSTRAP_BLOCK_ROWS)
    picked = rng.integers(len(starts), size=len(starts))
    return np.concatenate(
        [np.arange(starts[block], min(starts[block] + BOOTSTRAP_BLOCK_ROWS, num_rows)) for block in picked]
    )


def _spread(values):
    """Sample standard deviation over resamples of (average precision, F1)."""
    return {
        "average_precision_se": float(values[:, 0].std(ddof=1)),
        "f1_se": float(values[:, 1].std(ddof=1)),
    }


def bootstrap_standard_errors(labels, runs):
    """Block-bootstrap standard errors for average precision and F1.

    runs maps each checkpoint name to its (scores, predictions). Every resample uses the
    same rows for all checkpoints, so the base-to-fine-tuned difference is paired.
    Resamples with no anomaly are skipped because average precision is undefined for them.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(runs)
    draws = []
    while len(draws) < BOOTSTRAP_RESAMPLES:
        rows = block_bootstrap_rows(len(labels), rng)
        if labels[rows].sum() == 0:
            continue
        draws.append([
            [metrics["average_precision"], metrics["f1"]]
            for metrics in (
                score_metrics(labels[rows], runs[name][0][rows], runs[name][1][rows]) for name in names
            )
        ])
    draws = np.array(draws)  # shape: (resamples, checkpoints, [average precision, F1])
    errors = {name: _spread(draws[:, index]) for index, name in enumerate(names)}
    if {"base", "finetuned"} <= set(names):
        difference = draws[:, names.index("finetuned")] - draws[:, names.index("base")]
        errors["finetuned_minus_base"] = _spread(difference)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--finetuned-model", required=True, type=Path)
    parser.add_argument("--finetuned-config", required=True, type=Path)
    parser.add_argument("--base-model", type=Path, help="Optional base checkpoint for a before/after comparison.")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--nsample", type=int, default=15)
    args = parser.parse_args()
    if bool(args.base_model) != bool(args.base_config):
        parser.error("Pass --base-model and --base-config together, or neither.")

    features, labels = load_holdout(args.source, args.manifest)
    if labels.sum() == 0:
        raise ValueError("The held-out test set has no labeled anomalies.")

    checkpoints = {"finetuned": (args.finetuned_model, args.finetuned_config)}
    if args.base_model:
        checkpoints["base"] = (args.base_model, args.base_config)

    runs = {}
    for name, (model_path, config_path) in checkpoints.items():
        print(f"Scoring {name} checkpoint: {model_path}", flush=True)
        runs[name] = evaluate(features, str(model_path), str(config_path), args.nsample)
    errors = bootstrap_standard_errors(labels, runs)

    summary = {
        "eval_rows": int(len(labels)),
        "positives": int(labels.sum()),
        "nsample": args.nsample,
        "models": {
            name: {**score_metrics(labels, scores, predictions), **errors[name]}
            for name, (scores, predictions) in runs.items()
        },
    }
    if "finetuned_minus_base" in errors:
        base, finetuned = summary["models"]["base"], summary["models"]["finetuned"]
        summary["finetuned_minus_base"] = {
            "average_precision": finetuned["average_precision"] - base["average_precision"],
            "f1": finetuned["f1"] - base["f1"],
            **errors["finetuned_minus_base"],
        }

    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
