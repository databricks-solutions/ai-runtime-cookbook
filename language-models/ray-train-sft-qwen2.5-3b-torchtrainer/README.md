# Supervised fine-tuning Qwen2.5-3B with Ray Train

This recipe uses Ray Train to supervised-fine-tune `Qwen/Qwen2.5-3B` on the first 8,000 Alpaca examples. It runs distributed data parallel training with one worker per GPU on an 8x H100 node for 100 steps and logs training loss to the AIR MLflow run.

## Run

```bash
databricks air run -f workload.yaml
```
