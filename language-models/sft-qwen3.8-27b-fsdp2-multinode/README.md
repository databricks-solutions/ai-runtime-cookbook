# Multi-node FSDP2 fine-tuning of Qwen3.8-27B

This recipe uses plain PyTorch and FSDP2 to supervised-fine-tune `Qwen/Qwen3.8-27B` on UltraChat 200k for one full pass (983 steps) across four 8x H100 nodes. It packs conversations into 8,192-token blocks. No example attends to another. Rank 0 logs the training loss every 100 steps. The run takes about 1 hour 45 minutes, including about 35 minutes of setup before training starts.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

Rank 0 logs the training loss to the run's MLflow experiment every 100 steps, and the time per step and final loss when training ends. The same numbers appear in the workload logs (`databricks air logs`). In our validation run, training took about 4.2 seconds per step and ended with a loss of 0.856.
