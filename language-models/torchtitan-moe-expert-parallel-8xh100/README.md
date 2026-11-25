# Mixture-of-experts pretraining with expert parallelism using torchtitan

This recipe pretrains a mixture-of-experts (MoE) language model from scratch on C4 with [torchtitan](https://github.com/pytorch/torchtitan) on one 8xH100 node. The model uses the Qwen3-30B-A3B layer design: 128 experts per layer, 8 active per token. It keeps 12 of the 48 layers, for 8.1B total and 1.0B active parameters.

Experts are split across the 8 GPUs with expert parallelism (EP=8, 16 experts per GPU per layer), and tokens move between GPUs with all-to-all communication. The rest of the model is sharded with FSDP. The run trains for 100 steps (64K tokens per step) and takes about 3 minutes.

The config uses PyTorch FlexAttention, because torchtitan's default `varlen` backend needs FlashAttention 3, which the environment doesn't include.

## Before you run

The default `OUTPUT_ROOT` in `workload.yaml` is the shared Unity Catalog volume `/Volumes/main/default/air_examples/torchtitan-moe-expert-parallel-8xh100`. The volume must already exist and be writable. To use another volume, run:

```bash
databricks air run -f workload.yaml --override env_variables.OUTPUT_ROOT=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

`EXPERT_PARALLEL_DEGREE`, `TENSOR_PARALLEL_DEGREE` and `NUM_LAYERS` in `workload.yaml` set the layout. Only the default layout (EP=8, TP=1, 12 layers) has been validated. Raising `TENSOR_PARALLEL_DEGREE` leaves fewer data-parallel ranks, which lowers the tokens per step.

## Results

The run prints torchtitan's step metrics. It writes `summary.json` (final loss, tokens/s and TFLOPs per GPU, and all logged steps) to `<OUTPUT_ROOT>/<MLflow run ID>/`. It also logs the summary metrics to the MLflow run and uploads `summary.json` and the rank 0 training log (`logs/train.log`) as run artifacts.

A validation run on 8xH100 produced these results:

| Metric | Value |
|---|---|
| Loss, step 1 → step 100 | 12.38 → 6.74 |
| Throughput (step 100) | 16,953 tokens/s per GPU, 142 TFLOP/s per GPU |
| GPU memory | 21.2 GiB per GPU |
