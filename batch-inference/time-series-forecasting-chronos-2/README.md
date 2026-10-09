# Time-series forecasting with Chronos-2

Use [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) to batch 16 daily forecasts of German electricity prices from the public [Chronos quickstart dataset](https://github.com/amazon-science/chronos-forecasting#quick-start) on one A10 GPU. Each forecast predicts 24 hourly prices from up to 2,048 earlier observations, without fine-tuning.

## Before you run

Set `parameters.output_root` to an existing writable Unity Catalog volume directory if you cannot use the shared default. The model (Apache 2.0) and sample data download automatically without a Hugging Face token.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

`<output_root>/<MLFLOW_RUN_ID>/forecasts.parquet` contains forecast-task IDs, original series IDs, timestamps, held-out actuals, the 10th/50th/90th percentile forecasts, and a seasonal-naive forecast that repeats the previous day's prices. `metrics.json` and the AIR MLflow run report median-forecast MAE/RMSE, baseline MAE/RMSE, quantile pinball loss, empirical 80% interval coverage, and inference time. These artifacts are also logged to MLflow.

The final 16 days form non-overlapping evaluation windows. Each day's context ends before its forecast starts; later forecast origins can use observations from earlier days, as in a rolling daily backtest. Cross-learning is disabled so separate forecast tasks cannot share later observations.

The sample demonstrates the API and a chronological backtest; it is not a claim of performance on data unseen during foundation-model pretraining. To adapt it, supply a Parquet file with `id`, hourly `timestamp`, and numeric `target` columns; other columns are ignored.
