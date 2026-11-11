# Multi-node FSDP fine-tuning of Llama 3.1 8B

This recipe uses PyTorch FSDP to supervised-fine-tune `meta-llama/Llama-3.1-8B` on Alpaca for 100 steps. It launches 16 ranks across two 8x H100 nodes, uses bfloat16 full sharding, logs training loss to the AIR MLflow run, and saves a consolidated checkpoint.

## Before you run

Request access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) with the token owner's account and wait for approval. Follow the [token, workspace secret, and preflight instructions](../../docs/hugging-face-preflight.md) before your first run in a workspace or with this model. Repeat the checks after changing the profile, token, model, or run-as identity. For an ungated model, use the separate [Qwen2.5 TorchTrainer recipe](../ray-train-sft-qwen2.5-3b-torchtrainer/).

Set `secrets.HF_TOKEN` in `workload.yaml` to the secret containing the checked token (default `my_scope/hf_token`). Set `parameters.output_dir` to an existing writable Unity Catalog volume if the default location is unsuitable.

## Run

```bash
python ../../scripts/hf_preflight.py --profile "<profile-name>" \
  --model meta-llama/Llama-3.1-8B
# Submit only after the checks pass and run-as secret access is confirmed.
databricks air run -f workload.yaml --watch --profile "<profile-name>"
```

## Results

The model and tokenizer checkpoint are saved to `parameters.output_dir`.
