#!/usr/bin/env python3
"""Batch forecast electricity prices with Chronos-2, without fine-tuning."""

import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from chronos import Chronos2Pipeline


def split_history(data, horizon, context_length, seasonal_period, num_windows):
    """Build rolling forecast tasks; each context ends before its own holdout."""
    if min(horizon, context_length, seasonal_period, num_windows) < 1 or context_length < seasonal_period:
        raise ValueError("Lengths must be positive and context_length >= seasonal_period")
    data = data[["id", "timestamp", "target"]].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if data.duplicated(["id", "timestamp"]).any() or data.isna().any().any():
        raise ValueError("Expected non-null targets and unique (id, timestamp) pairs")
    contexts, holdouts = [], []
    for series_id, series in data.groupby("id", sort=True):
        series = series.sort_values("timestamp").tail(context_length + horizon * num_windows)
        if len(series) < horizon * num_windows + seasonal_period:
            raise ValueError("Each series needs all forecast windows plus one seasonal period")
        if not series["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError("This electricity recipe expects regular hourly observations")
        for window in range(num_windows):
            end = len(series) - horizon * (num_windows - window - 1)
            cutoff = end - horizon
            history = series.iloc[max(0, cutoff - context_length):cutoff].copy()
            actual = series.iloc[cutoff:end].copy()
            task_id = f"{series_id}:{window:03d}"
            history["id"], actual["id"] = task_id, task_id
            actual["source_id"] = series_id
            actual["seasonal_naive"] = np.resize(history["target"].to_numpy()[-seasonal_period:], horizon)
            contexts.append(history)
            holdouts.append(actual)
    if not contexts:
        raise ValueError("No time series found")
    return pd.concat(contexts, ignore_index=True), pd.concat(holdouts, ignore_index=True)


def evaluate(predictions, actual):
    scored = actual.merge(predictions, on=["id", "timestamp"], how="outer", validate="one_to_one", indicator=True)
    if not scored["_merge"].eq("both").all():
        raise ValueError("Forecast timestamps must match every held-out timestamp exactly")
    scored = scored.drop(columns="_merge")
    values = scored[["target", "0.1", "0.5", "0.9", "seasonal_naive"]].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Targets and forecasts must be finite")
    metrics = {}
    for name, column in (("chronos", "0.5"), ("seasonal_naive", "seasonal_naive")):
        error = scored["target"].to_numpy() - scored[column].to_numpy()
        metrics[f"{name}_mae"] = float(np.abs(error).mean())
        metrics[f"{name}_rmse"] = float(np.sqrt(np.square(error).mean()))
    metrics["coverage_80"] = float(scored["target"].between(scored["0.1"], scored["0.9"]).mean())
    for q in (0.1, 0.5, 0.9):
        error = scored["target"].to_numpy() - scored[str(q)].to_numpy()
        metrics[f"pinball_{q}"] = float(np.maximum(q * error, (q - 1) * error).mean())
    return scored, metrics


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    context, actual = split_history(
        pd.read_parquet(p["data_url"]), p["prediction_length"], p["context_length"], p["seasonal_period"], p["num_windows"]
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({**p, "forecast_tasks": context["id"].nunique(), "cross_learning": False})
        pipeline = Chronos2Pipeline.from_pretrained(p["model_name"], device_map="cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        # Only historical target values enter the model. No held-out values or covariates.
        predictions = pipeline.predict_df(
            context,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
            prediction_length=p["prediction_length"],
            context_length=p["context_length"],
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=p["batch_size"],
            cross_learning=False,
        )
        torch.cuda.synchronize()
        prediction_seconds = time.perf_counter() - started
        scored, metrics = evaluate(predictions, actual)
        metrics["prediction_seconds"] = prediction_seconds
        scored.to_parquet(output_dir / "forecasts.parquet", index=False)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir), artifact_path="forecast")
        print(f"Metrics: {metrics}\nForecasts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
