# Multi-node FSDP fine-tuning of Llama 3.1 8B

This recipe uses PyTorch FSDP to supervised-fine-tune `meta-llama/Llama-3.1-8B` on Alpaca for 100 steps. It launches 16 ranks across two 8x H100 nodes, uses bfloat16 full sharding, logs training loss to the AIR MLflow run, and saves a consolidated checkpoint.

## Before you run

Complete the [Hugging Face token and Databricks secret setup](../../docs/hugging-face-secrets.md) for your target workspace. Obtain access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) for the token owner's account, and configure `secrets.HF_TOKEN` in `workload.yaml` with the secret reference (default: `my_scope/hf_token`).

Set `parameters.output_dir` to an existing writable Unity Catalog volume if the default location is unsuitable.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The model and tokenizer checkpoint are saved to `parameters.output_dir`.
