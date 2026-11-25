#!/usr/bin/env python3
"""Load each saved fine-tuned checkpoint in a new process and forecast one series with it.

Run after finetune_backtest.py, so nothing from that process is in memory. A LoRA checkpoint
is an adapter, and from_pretrained applies it to the base model.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from chronos import Chronos2Pipeline


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    run_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    horizon = int(p["prediction_length"])
    context_length = int(p["context_length"])
    values = np.asarray(pd.read_parquet(p["data_url"]).iloc[0]["target"], dtype=np.float32)
    context = values[-(context_length + horizon) : -horizon]
    for mode in ("lora", "full"):
        ckpt = str(run_dir / mode / "finetuned-ckpt")
        pipeline = Chronos2Pipeline.from_pretrained(ckpt, device_map="cuda")
        quantiles, _ = pipeline.predict_quantiles(
            [context], prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
        )
        forecast = quantiles[0][0].float().cpu().numpy()
        if forecast.shape != (horizon, 3) or not np.isfinite(forecast).all():
            raise RuntimeError(f"{mode} checkpoint forecast has shape {forecast.shape} or non-finite values")
        print(f"{mode}: loaded {ckpt}, forecast shape {forecast.shape}", flush=True)


if __name__ == "__main__":
    main()
