# Single-node FSDP2 fine-tuning of Qwen3.8-27B on 8x B300

This recipe uses plain PyTorch and FSDP2 to supervised-fine-tune `Qwen/Qwen3.8-27B` on UltraChat 200k on a single 8x B300 node. It makes one full pass over the dataset in 995 steps. It packs conversations into 32,768-token blocks. No conversation attends to another one in the same block. The run takes about 2 hours 25 minutes, including about 10 minutes of setup before training starts.

The recipe adapts [`sft-qwen3.8-27b-fsdp2-multinode`](../sft-qwen3.8-27b-fsdp2-multinode), which runs on four 8x H100 nodes. Each B300 GPU has 288 GB of HBM and fits 4x the H100 recipe's sequence length. As a result, one B300 node trains on as many tokens per step as the four H100 nodes. FlashAttention-3 runs only on Hopper GPUs such as the H100. This recipe uses the FlashAttention-2 build that ships in the `databricks_ai_v6` environment instead.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The metrics are in the workload logs. You can view them with `databricks air logs`. Rank 0 logs the training loss every 100 steps. At the end, it logs one summary line with the step count, the time per step, and the final loss. In our validation run, training took 8.06 seconds per step and ended with a loss of 0.735.

The recipe doesn't save checkpoints or a trained model.
