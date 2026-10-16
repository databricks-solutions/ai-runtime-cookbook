"""Prepare normal-only training and validation windows from the Kaggle dataset."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/drscarlat/time-series"
CHECKSUMS = {
    "TimeSeries.csv": "3b62ba9d302ff42f3a17c9bbd52186220e26e032b6c81f14c763a84fec14e4a2",
    "labelsTimeSeries.csv": "51ab67863258714efd37c94d90938b9f3d841d2328367cfc62f0e06e9552a177",
}


def normal_windows(features, labels, start, stop, window, max_rows):
    starts = np.arange(start, stop - window + 1, window)
    starts = starts[np.array([not labels[i : i + window].any() for i in starts])]
    if max_rows:
        starts = starts[: max_rows // window]
    if len(starts) < 10:
        raise ValueError(
            "Need at least ten complete normal windows in each training/normal-validation partition."
        )
    indices = (starts[:, None] + np.arange(window)).reshape(-1)
    return features.iloc[indices].reset_index(drop=True), indices


def prepare(source, output, window=100, train_rows=50000, val_rows=10000):
    if window != 100:
        raise ValueError("This recipe uses 100-row windows.")
    output.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(source / "TimeSeries.csv")
    y_frame = pd.read_csv(source / "labelsTimeSeries.csv")
    if list(x.columns) != [f"v{i}" for i in range(1, 12)] or list(y_frame.columns) != [
        "label"
    ]:
        raise ValueError(
            "Expected the Kaggle TimeSeries.csv (v1..v11) and labelsTimeSeries.csv (label)."
        )
    y = y_frame.label.to_numpy()
    if (
        len(x) != len(y)
        or not np.isfinite(x.to_numpy()).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError(
            "Features/labels must align, contain finite numeric values, and have binary labels."
        )
    a, b = int(len(x) * 0.6) // window * window, int(len(x) * 0.8) // window * window
    train, train_indices = normal_windows(x, y, 0, a, window, train_rows)
    val, val_indices = normal_windows(x, y, a, b, window, val_rows)
    train.to_csv(output / "train.csv", index=False)
    val.to_csv(output / "normal_val.csv", index=False)
    pd.DataFrame({"row_id": train_indices}).to_csv(
        output / "train_rows.csv", index=False
    )
    pd.DataFrame({"row_id": val_indices}).to_csv(
        output / "normal_val_rows.csv", index=False
    )
    manifest = {
        "dataset": "drscarlat/time-series",
        "source_url": DATASET_URL,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in CHECKSUMS
        },
        "source_rows": len(x),
        "feature_columns": list(x),
        "window_length": window,
        "training_stride": window,
        "normal_train_windows": len(train) // window,
        "normal_val_windows": len(val) // window,
        "temporal_boundaries": [0, a, b],
        "note": "Training/normal-validation use only whole normal blocks. Stride equals block length, so concatenated blocks never create windows across anomaly gaps. Training uses the first 60% source partition, normal validation the next 20%; remaining rows are unused. Row IDs retain original observation indices.",
    }
    manifest["sha256"] = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*.csv"))
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def download(source):
    source.mkdir(parents=True, exist_ok=True)
    if not all((source / name).exists() for name in CHECKSUMS):
        archive = source / "dataset.zip"
        urllib.request.urlretrieve(DATASET_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            for name in CHECKSUMS:
                (source / name).write_bytes(bundle.read(name))
    for name, expected in CHECKSUMS.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Dataset checksum mismatch: {name}; refusing a silently changed benchmark."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-rows", type=int, default=50000)
    parser.add_argument("--val-rows", type=int, default=10000)
    args = parser.parse_args()
    if min(args.train_rows, args.val_rows) < 0:
        parser.error("Row limits must be nonnegative; 0 uses all eligible rows.")
    download(args.source)
    prepare(
        args.source, args.output, train_rows=args.train_rows, val_rows=args.val_rows
    )
