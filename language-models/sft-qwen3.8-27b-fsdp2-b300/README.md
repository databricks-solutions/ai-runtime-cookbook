# Single-node FSDP2 fine-tuning of Qwen3.8-27B on 8x B300

This recipe uses plain PyTorch and FSDP2 to supervised-fine-tune `Qwen/Qwen3.8-27B` on UltraChat 200k on a single 8x B300 node. It makes one full pass over the dataset in 995 steps. It packs conversations into 32,768-token blocks. No conversation attends to another one in the same block. The run takes about 2 hours 25 minutes, including about 10 minutes of setup before training starts.

The recipe adapts [`sft-qwen3.8-27b-fsdp2-multinode`](../sft-qwen3.8-27b-fsdp2-multinode), which runs on four 8x H100 nodes. Each B300 GPU has 288 GB of HBM and fits 4x the H100 recipe's sequence length. As a result, one B300 node trains on as many tokens per step as the four H100 nodes. FlashAttention-3 runs only on Hopper GPUs such as the H100. This recipe uses the FlashAttention-2 build that ships in the `databricks_ai_v6` environment instead.

## Before you run

The default `output_root` is the shared Unity Catalog volume `/Volumes/main/default/air_examples/sft-qwen3.8-27b-fsdp2-b300`. The volume must already exist and be writable. To use another volume, run:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

Rank 0 logs the training loss to the run's MLflow experiment every 100 steps, and the time per step and final loss when training ends. The same numbers appear in the workload logs (`databricks air logs`). In our validation run, training took 8.06 seconds per step and ended with a loss of 0.735.

By default, the recipe doesn't save checkpoints. To save resumable checkpoints and resume from one, set `checkpoint_every_steps` and `resume_from` in `workload.yaml`.
