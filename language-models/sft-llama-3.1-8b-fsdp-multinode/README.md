# Multi-node FSDP fine-tuning of Llama 3.1 8B

This recipe uses PyTorch FSDP to supervised-fine-tune `meta-llama/Llama-3.1-8B` on Alpaca for 100 steps. It launches 16 ranks across two 8x H100 nodes, uses bfloat16 full sharding, logs training loss to the AIR MLflow run, and saves a consolidated checkpoint.

## Before you run

Set `HF_TOKEN` in `workload.yaml` to a Databricks secret containing a Hugging Face token with access to `meta-llama/Llama-3.1-8B`. Set `parameters.output_dir` to an existing writable Unity Catalog volume if the default location is unsuitable.

## Run

[Select and verify your workspace profile](../../README.md#select-a-workspace-profile) before submitting. Keep that profile selected when [checking status or logs](../../README.md#submit-and-monitor).

```bash
databricks air run -f workload.yaml --watch
```

## Results

The model and tokenizer checkpoint are saved to `parameters.output_dir`.
