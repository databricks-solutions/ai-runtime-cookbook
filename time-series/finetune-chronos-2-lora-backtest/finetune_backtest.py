#!/usr/bin/env python3
"""Compare zero-shot, LoRA and full fine-tuned Chronos-2 on the same rolling daily backtest."""

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
from transformers import TrainerCallback

QUANTILE_LEVELS = [0.1, 0.5, 0.9]


class MLflowLossCallback(TrainerCallback):
    """Forwards fit() loss logs to the open AIR run, because fit() sets report_to="none"."""

    def __init__(self, prefix):
        self.prefix = prefix

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "eval_loss", "learning_rate"):
            if logs and key in logs:
                mlflow.log_metric(f"{self.prefix}_{key}", float(logs[key]), step=state.global_step)


def build_backtest(series_list, ids, horizon, context_length, seasonal_period, num_windows):
    """Split every series into training, validation and test regions, in that order.

    The test set is the last num_windows * horizon points, cut into consecutive daily forecast
    origins. The validation set is the same kind of windows just before the test set. Training
    inputs end where validation starts, so no test or validation target reaches fine-tuning.
    """
    span = num_windows * horizon
    kept = []
    for index, values in enumerate(series_list):
        if not np.isfinite(values).all():
            raise ValueError(f"Series {ids[index]} has missing values")
        test_start = len(values) - span
        val_start = test_start - span
        if val_start < seasonal_period + horizon:
            raise ValueError(f"Series {ids[index]} is too short for {num_windows} validation and test windows")
        # MASE needs a non-zero seasonal-naive scale at every origin. Exclusion uses only the inputs
        # (history up to each origin), never the targets, and it applies to every method alike.
        scales = [
            np.abs(context[seasonal_period:] - context[:-seasonal_period]).mean()
            for context in (
                values[max(0, test_start + w * horizon - context_length) : test_start + w * horizon]
                for w in range(num_windows)
            )
        ]
        if min(scales) == 0:
            continue
        kept.append(index)

    train_inputs, validation_inputs = [], []
    test_contexts, test_actuals, test_scales, test_seasonal, test_task_series = [], [], [], [], []
    for index in kept:
        values = series_list[index]
        test_start = len(values) - span
        val_start = test_start - span
        train_inputs.append(values[:val_start])
        for window in range(num_windows):
            cut = val_start + window * horizon
            # Validation inputs end at the last validation target, which is before any test point.
            validation_inputs.append(values[max(0, cut - context_length) : cut + horizon])
        for window in range(num_windows):
            cut = test_start + window * horizon
            context = values[max(0, cut - context_length) : cut]
            test_contexts.append(context)
            test_actuals.append(values[cut : cut + horizon])
            test_scales.append(np.abs(context[seasonal_period:] - context[:-seasonal_period]).mean())
            test_seasonal.append(np.resize(context[-seasonal_period:], horizon))
            test_task_series.append(index)
    return {
        "kept": kept,
        "train_inputs": train_inputs,
        "validation_inputs": validation_inputs,
        "test_contexts": test_contexts,
        "test_actuals": np.stack(test_actuals),
        "test_scales": np.asarray(test_scales),
        "test_seasonal": np.stack(test_seasonal),
        "test_task_series": np.asarray(test_task_series),
    }


def forecast_quantiles(pipeline, contexts, horizon, context_length, batch_size):
    quantiles, _ = pipeline.predict_quantiles(
        contexts,
        prediction_length=horizon,
        quantile_levels=QUANTILE_LEVELS,
        batch_size=batch_size,
        context_length=context_length,
        cross_learning=False,
    )
    # predict_quantiles returns one (n_variates, horizon, levels) tensor per task. Chronos-2 is univariate here.
    stacked = np.stack([q[0].float().cpu().numpy() for q in quantiles])
    expected = (len(contexts), horizon, len(QUANTILE_LEVELS))
    if stacked.shape != expected:
        raise RuntimeError(f"Quantile forecast shape {stacked.shape}, expected {expected}")
    return stacked


def per_task_scores(actual, forecast, scale):
    """Scores for (tasks, horizon, levels) quantile forecasts. The median is the 0.5 level."""
    levels = np.asarray(QUANTILE_LEVELS, dtype=np.float32)
    if forecast.shape != (*actual.shape, len(QUANTILE_LEVELS)):
        raise ValueError(f"Forecast shape {forecast.shape} does not match actuals {actual.shape}")
    error = actual[:, :, None] - forecast
    pinball = np.maximum(levels * error, (levels - 1) * error).sum(axis=(1, 2))
    mase = np.abs(actual - forecast[:, :, 1]).mean(axis=1) / scale
    return mase, pinball, np.abs(actual).sum(axis=1)


def per_series_scores(task_series, mase, pinball, abs_actual, num_series):
    frame = pd.DataFrame({"series": task_series, "mase": mase, "pinball": pinball, "abs_actual": abs_actual})
    grouped = frame.groupby("series")
    # Weighted quantile loss per series: 2 * summed pinball / summed |actual|, averaged over levels.
    wql = 2 * grouped["pinball"].sum() / (len(QUANTILE_LEVELS) * grouped["abs_actual"].sum())
    scores = pd.DataFrame({"mase": grouped["mase"].mean(), "wql": wql.replace([np.inf, -np.inf], np.nan)})
    return scores.reindex(range(num_series))


def paired_difference(a, b, rng, resamples=2000):
    """Mean of a - b over series, with its standard error and a paired bootstrap 95% interval."""
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = (a - b)[both]
    draws = rng.integers(0, len(diff), size=(resamples, len(diff)))
    boot = diff[draws].mean(axis=1)
    return {
        "mean": float(diff.mean()),
        "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "n_series": int(len(diff)),
    }


def mean_and_se(values):
    values = values[~np.isnan(values)]
    return {"mean": float(values.mean()), "se": float(values.std(ddof=1) / np.sqrt(len(values))), "n_series": int(len(values))}


def finetune(base, splits, mode, learning_rate, p, output_dir):
    pipeline = base.fit(
        inputs=splits["train_inputs"],
        prediction_length=int(p["prediction_length"]),
        validation_inputs=splits["validation_inputs"],
        finetune_mode=mode,
        context_length=int(p["context_length"]),
        learning_rate=float(learning_rate),
        num_steps=int(p["finetune_steps"]),
        batch_size=int(p["finetune_batch_size"]),
        output_dir=output_dir,
        callbacks=[MLflowLossCallback(mode)],
        logging_steps=50,
        disable_tqdm=True,
    )
    if mode == "lora":
        # fit() silently falls back to full fine-tuning when peft is missing. Fail instead.
        if not hasattr(pipeline.model, "merge_and_unload"):
            raise RuntimeError("finetune_mode='lora' did not produce a LoRA model")
        pipeline = Chronos2Pipeline(model=pipeline.model.merge_and_unload())
    return pipeline


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    run_id = os.environ["MLFLOW_RUN_ID"]
    horizon = int(p["prediction_length"])
    context_length = int(p["context_length"])
    seasonal_period = int(p["seasonal_period"])
    num_windows = int(p["num_windows"])
    max_series = int(p["max_series"])
    predict_batch_size = int(p["predict_batch_size"])
    seed = int(p["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(p["data_url"])
    if 0 < max_series < len(frame):
        frame = frame.iloc[np.linspace(0, len(frame) - 1, max_series).astype(int)]
    ids = frame["id"].tolist()
    series_list = [np.asarray(values, dtype=np.float32) for values in frame["target"]]
    splits = build_backtest(series_list, ids, horizon, context_length, seasonal_period, num_windows)
    num_series = len(splits["kept"])
    print(
        f"Series: {len(series_list)} loaded, {num_series} kept, {len(series_list) - num_series} dropped "
        f"(constant at the seasonal lag). Test tasks: {len(splits['test_contexts'])}",
        flush=True,
    )

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(
            {
                **p,
                "series_kept": num_series,
                "series_dropped": len(series_list) - num_series,
                "test_tasks": len(splits["test_contexts"]),
            }
        )
        base = Chronos2Pipeline.from_pretrained(p["model_name"], device_map="cuda")

        forecasts, fit_seconds = {}, {}
        forecasts["seasonal_naive"] = splits["test_seasonal"]
        forecasts["zero_shot"] = forecast_quantiles(
            base, splits["test_contexts"], horizon, context_length, predict_batch_size
        )
        for mode, learning_rate in (("lora", p["lora_learning_rate"]), ("full", p["full_learning_rate"])):
            started = time.perf_counter()
            tuned = finetune(base, splits, mode, learning_rate, p, str(output_dir / mode))
            fit_seconds[mode] = time.perf_counter() - started
            forecasts[mode] = forecast_quantiles(
                tuned, splits["test_contexts"], horizon, context_length, predict_batch_size
            )

        per_method = {}
        for method, forecast in forecasts.items():
            if method == "seasonal_naive":
                # A point forecast has no quantiles, so only MASE is defined for it.
                mase = np.abs(splits["test_actuals"] - forecast).mean(axis=1) / splits["test_scales"]
                per_method[method] = pd.DataFrame(
                    {"mase": pd.Series(mase).groupby(splits["test_task_series"]).mean()}
                ).reindex(range(len(ids)))
                continue
            mase, pinball, abs_actual = per_task_scores(
                splits["test_actuals"], forecast, splits["test_scales"]
            )
            per_method[method] = per_series_scores(splits["test_task_series"], mase, pinball, abs_actual, len(ids))
        per_method = {m: s.reindex(splits["kept"]).reset_index(drop=True) for m, s in per_method.items()}

        rng = np.random.default_rng(seed)
        table = {}
        for method in per_method:
            table[method] = {"mase": mean_and_se(per_method[method]["mase"].to_numpy())}
            if method != "seasonal_naive":
                table[method]["wql"] = mean_and_se(per_method[method]["wql"].to_numpy())
        differences = {}
        for a, b in (("lora", "zero_shot"), ("full", "zero_shot"), ("lora", "full")):
            differences[f"{a}_minus_{b}"] = {
                "mase": paired_difference(per_method[a]["mase"].to_numpy(), per_method[b]["mase"].to_numpy(), rng),
                "wql": paired_difference(per_method[a]["wql"].to_numpy(), per_method[b]["wql"].to_numpy(), rng),
            }

        metrics = {}
        for method, values in table.items():
            for name, stats in values.items():
                metrics[f"{method}_{name}_mean"] = stats["mean"]
                metrics[f"{method}_{name}_se"] = stats["se"]
        for name, values in differences.items():
            for metric, stats in values.items():
                metrics[f"{name}_{metric}_diff"] = stats["mean"]
                metrics[f"{name}_{metric}_diff_se"] = stats["se"]
                metrics[f"{name}_{metric}_diff_ci_low"] = stats["ci95"][0]
                metrics[f"{name}_{metric}_diff_ci_high"] = stats["ci95"][1]
        for mode, seconds in fit_seconds.items():
            metrics[f"{mode}_fit_seconds"] = seconds
        mlflow.log_metrics(metrics)

        per_series = pd.DataFrame({"series_id": [ids[i] for i in splits["kept"]]})
        for method, scores in per_method.items():
            per_series[f"{method}_mase"] = scores["mase"].to_numpy()
            if method != "seasonal_naive":
                per_series[f"{method}_wql"] = scores["wql"].to_numpy()
        per_series.to_csv(output_dir / "per_series_metrics.csv", index=False)
        report = {"table": table, "paired_differences": differences, "fit_seconds": fit_seconds}
        (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        mlflow.log_artifact(str(output_dir / "per_series_metrics.csv"), artifact_path="backtest")
        mlflow.log_artifact(str(output_dir / "comparison.json"), artifact_path="backtest")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
