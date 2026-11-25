# Fine-tuning Chronos-2 with LoRA and full fine-tuning, on a rolling backtest

This recipe asks whether fine-tuning a time-series foundation model beats zero-shot forecasting on one dataset. It compares [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) with a LoRA fine-tune and a full fine-tune, using `Chronos2Pipeline.fit()`. All three are scored on the same 14 daily origins for 320 hourly electricity series, on one A10 GPU.

The existing [zero-shot Chronos-2 recipe](../time-series-forecasting-chronos-2/) covers forecasting without fine-tuning. This recipe covers the decision after that: whether to fine-tune, and which mode to use.

Data comes from the `monash_electricity_hourly` config of [autogluon/chronos_datasets](https://huggingface.co/datasets/autogluon/chronos_datasets), licensed CC BY 4.0. It has 321 hourly client series of 26,304 points each. One series is dropped because its 24-hour seasonal-naive scale is zero at a test origin. The rule uses inputs only, never targets. That leaves 320 series.

The last 14 days (336 hours) are the test set. It is cut into 14 consecutive daily origins of 24 hours each. The 14 days before it are the validation set. Training inputs end where validation starts. Validation inputs end at their last validation target, which is before the test set. Test targets are never used for fine-tuning, checkpoint selection, or hyperparameters.

Methods:

- Seasonal naive: the previous day's values, as a floor.
- Zero-shot Chronos-2.
- LoRA fine-tuning with `learning_rate=1e-5`, the value the `fit()` docstring suggests for LoRA, and the default LoRA config.
- Full fine-tuning with `learning_rate=1e-6`, the `fit()` default.

Both fine-tunes run 500 steps at batch size 64, with a 2,048-point context. These settings were fixed before any run and not tuned. The learning rate is the main sensitivity. Read the comparison as "library-default settings", not "best possible fine-tune". `fit()` keeps the checkpoint with the lowest validation loss (`load_best_model_at_end`, metric `eval_loss`), and the validation loss is logged to the AIR run.

Metrics, per series and averaged over its 14 origins:

- MASE: mean absolute error of the median forecast, divided by the in-context seasonal-naive MAE at lag 24.
- WQL: 2 times the summed pinball loss at levels 0.1, 0.5 and 0.9, divided by the summed absolute actuals, averaged over those levels.

Uncertainty treats series as the independent unit. The 14 targets of one series are non-overlapping 24-hour blocks, but their 2,048-point contexts share most of their history. Each metric is reported as its mean across series, with its standard error across series. Differences are paired per series, with a paired bootstrap 95% interval over series (2,000 resamples). A fine-tune counts as better only if its interval for the difference excludes zero.

## Before you run

The model and dataset are public, and no Hugging Face token is needed. Set `parameters.output_root` to an existing writable Unity Catalog volume directory if you cannot use the shared default. The run writes the fine-tuned checkpoints there.

## Run

```bash
databricks air run -f workload.yaml --watch
```

A short smoke run, for debugging only:

```bash
databricks air run -f workload.yaml --watch \
  --override parameters.max_series=12 \
  --override parameters.num_windows=2 \
  --override parameters.finetune_steps=20
```

## Results

One default run, on 320 series and 4,480 test tasks (320 series times 14 origins). The fits took 259 s (LoRA) and 349 s (full). Container time at default settings was 11m 27s, excluding queueing.

Full fine-tuning is slightly better than zero-shot, and LoRA is slightly worse. Both gaps are small. The full fine-tune's MASE gain is statistically detectable but practically negligible, about 0.4% of zero-shot MASE. The standard errors and intervals are conditional on this single 14-day test period. Other periods could change the ranking.

| Method | MASE (mean ± SE over series) | WQL (mean ± SE over series) |
|---|---|---|
| Seasonal naive | 1.724 ± 0.033 | not reported |
| Zero-shot Chronos-2 | 1.168 ± 0.027 | 0.0830 ± 0.0043 |
| LoRA fine-tune | 1.170 ± 0.027 | 0.0831 ± 0.0043 |
| Full fine-tune | 1.163 ± 0.027 | 0.0826 ± 0.0043 |

Paired differences over series (negative is better), with paired bootstrap 95% intervals:

| Comparison | MASE diff (95% CI) | WQL diff (95% CI) |
|---|---|---|
| LoRA minus zero-shot | +0.0016 (+0.0011, +0.0021) | +0.00007 (+0.00001, +0.00012) |
| Full minus zero-shot | −0.0048 (−0.0063, −0.0034) | −0.00034 (−0.00059, −0.00010) |
| LoRA minus full | +0.0064 (+0.0051, +0.0077) | +0.00041 (+0.00022, +0.00060) |

Caveat: Chronos-2's pretraining data is not checked here, so this dataset may overlap with it. If it does, the zero-shot numbers may be optimistic. This null result cannot separate "fine-tuning does not help" from "zero-shot already saw this data". The paired comparison between methods still uses the same inputs.

`<output_root>/<MLFLOW_RUN_ID>/lora/finetuned-ckpt` holds the LoRA adapter. `<output_root>/<MLFLOW_RUN_ID>/full/finetuned-ckpt` holds the full model. `per_series_metrics.csv` and `comparison.json` hold the per-series scores and the comparison. The AIR MLflow run has the summary metrics, the training and validation loss curves, and these two files under `backtest/`.
