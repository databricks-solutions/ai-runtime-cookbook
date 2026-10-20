# Single-node FSDP2 fine-tuning of Qwen3.8-27B on 8x B300

This recipe uses plain PyTorch and FSDP2 to supervised-fine-tune `Qwen/Qwen3.8-27B` on UltraChat 200k for one full pass (995 steps) on a single 8x B300 node, packing conversations into 32,768-token blocks so that no example attends to another. Rank 0 logs the training loss every 100 steps, and the run takes about 2 hours 25 minutes, including about 10 minutes of setup before training starts.

It adapts [`sft-qwen3.8-27b-fsdp2-multinode`](../sft-qwen3.8-27b-fsdp2-multinode) (four 8x H100 nodes) to B300. Each B300 GPU has 288 GB of HBM, which fits 4x the sequence length, so one node trains on as many tokens per step as the four H100 nodes. FlashAttention-3 runs only on Hopper, so this recipe uses the FlashAttention-2 build that ships in the `databricks_ai_v6` environment.

## Run

```bash
databricks air run -f workload.yaml
```
