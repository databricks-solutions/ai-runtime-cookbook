# LoRA hyperparameter search for Qwen2.5-0.5B

This recipe uses Ray Tune and ASHA to search LoRA configurations for `Qwen/Qwen2.5-0.5B` on Alpaca. It runs eight trials, up to four at once on A10 GPUs, evaluates each trial on a 200-example held-out split, and logs the best configuration and evaluation loss to the AIR MLflow run.

## Run

```bash
databricks air run -f workload.yaml
```
