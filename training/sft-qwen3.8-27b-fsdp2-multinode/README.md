# Multi-node FSDP2 fine-tuning of Qwen3.8-27B

This recipe supervised-fine-tunes `Qwen/Qwen3.8-27B` on UltraChat 200k for one full pass (about 980 steps) with plain PyTorch and FSDP2 across four 8x H100 nodes. It packs conversations into 8,192-token blocks without attention across examples, and takes about 1 hour 45 minutes, including about 35 minutes of setup to build FlashAttention-3 and causal-conv1d from source.

## Run

```bash
databricks air run -f workload.yaml
```
