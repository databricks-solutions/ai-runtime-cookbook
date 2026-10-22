# Kumo-Anomaly fine-tuning for time-series anomaly detection

Fine-tune NVIDIA's [Kumo-Anomaly](https://huggingface.co/nvidia/Kumo-Anomaly) diffusion model on the public [Kaggle multivariate time-series dataset](https://www.kaggle.com/datasets/drscarlat/time-series) for 10 epochs on one A10 GPU. The recipe runs the fine-tuning example from NVIDIA's [Kumo-TS repository](https://github.com/NVIDIA/Kumo-TS/tree/main/Kumo-Anomaly) at a pinned commit, inside that project's locked `uv` environment.

Kumo-Anomaly learns what normal behavior looks like, so the recipe trains and validates on anomaly-free data only. The dataset has 11 numeric features and a 0/1 anomaly label for each row.

The recipe splits the series in time order into three parts: the first 60% for training, the next 20% for validation, and the last 20% held out as a test set. After training, the recipe scores that test set to measure detection performance before and after fine-tuning (see [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) below). Within the training and validation parts, the recipe scans the series in non-overlapping 100-row blocks and keeps only the blocks that contain no labeled anomalies. It keeps the earliest such blocks up to a fixed count: 500 blocks (50,000 rows) for training and 100 blocks (10,000 rows) for validation. Labels only decide which blocks to keep. The model never sees them.

The recipe reports the upstream training and validation losses. The model is trained to denoise normal data. It learns to predict the noise that was added to the data. The loss measures how well it does this. A lower loss means better denoising.

A low loss does not mean good anomaly detection. The two are different. To measure detection quality, you score labeled data that has both normal and anomalous rows, then compare the scores to the labels. A common metric is average precision. The recipe runs this step on the held-out test set and reports the result. See [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) below.

The validation loss is stochastic, so it changes from epoch to epoch. The epoch with the lowest validation loss becomes the best checkpoint. The workload takes about 10 minutes to run. Fine-tuning takes a few minutes, and scoring the held-out test set takes a few more.

## Before you run

The workload downloads the upstream code, Python packages, model weights, and dataset when it starts, so it needs outbound access to GitHub, PyPI, Hugging Face, and Kaggle. The weights are Apache 2.0 and download without a Hugging Face token. The dataset downloads without a Kaggle account, but Kaggle lists its license as unknown, so review the terms on the dataset page before you use it. To write the checkpoints elsewhere, set `env_variables.OUTPUT_ROOT` to an existing Unity Catalog volume you can write to.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run has per-epoch `train_loss` and `val_loss` and `best_val_loss`, plus the held-out metrics `base_average_precision`, `finetuned_average_precision`, `base_f1`, and `finetuned_f1`. These files are copied to `<OUTPUT_ROOT>/<MLFLOW_RUN_ID>/` and uploaded to the run under `kumo_finetune/`:

- `checkpoints/best_finetuned_model.pth` and `checkpoints/final_finetuned_model.pth`: the best-validation and last-epoch checkpoints
- `checkpoints/finetune_config.yaml`: the model configuration needed to load a checkpoint
- `checkpoints/epoch_metrics.json`: the losses for each epoch
- `training_summary.json`: training parameters, the upstream commit, and the final and best losses
- `evaluation.json`: the held-out average precision and F1 for the base and fine-tuned checkpoints

The MLflow run also keeps the prepared training and validation CSVs with their original row IDs and checksums (`kumo_finetune/data/`), plus the data-preparation, training, and evaluation logs. The [Evaluate on the held-out test set](#evaluate-on-the-held-out-test-set) section explains these metrics and how to rerun the evaluation.

## Evaluate on the held-out test set

Training and validation use normal data only, so the losses do not show how well the model detects anomalies. The last 20% of the series is held out for that. It keeps its labels and has both normal and anomalous rows, so it works as a labeled test set.

After training, the recipe scores the held-out rows with the base checkpoint and the fine-tuned checkpoint. For each one it reports average precision and F1 against the labels. Average precision suits this task because anomalies are rare and it needs no threshold. Comparing the base and fine-tuned numbers shows the effect of fine-tuning.

Anomalies are rare, so scoring is the slowest part of the recipe. `EVAL_ROWS` sets how many held-out rows to score. The default of 2500 keeps the run short. Set it to 0 to score the whole test set for a less noisy metric, and raise `timeout_minutes` to match.

To run the evaluation on its own, use `evaluate_holdout.py` from the upstream `Kumo-Anomaly` directory and its `uv` environment. It scores the held-out rows with the given checkpoints and writes the metrics to the output file:

```bash
uv run --frozen --no-dev python evaluate_holdout.py \
    --source <dir with TimeSeries.csv and labelsTimeSeries.csv> \
    --manifest <work_dir>/data/manifest.json \
    --finetuned-model <work_dir>/checkpoints/best_finetuned_model.pth \
    --finetuned-config <work_dir>/checkpoints/finetune_config.yaml \
    --base-model <work_dir>/weights/final_model.pth \
    --base-config <work_dir>/weights/curriculum_medium.yaml \
    --output evaluation.json
```

The script uses the upstream detection function `perform_anomaly_analysis_with_diffusion`. See [`examples/quick_example.py`](https://github.com/NVIDIA/Kumo-TS/blob/main/Kumo-Anomaly/examples/quick_example.py) for a standalone detection example.
