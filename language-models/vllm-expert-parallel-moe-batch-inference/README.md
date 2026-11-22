# Batch inference with a frontier-scale MoE model: vLLM tensor parallel vs. data and expert parallel

This recipe runs offline batch generation with [vLLM](https://github.com/vllm-project/vllm) 0.34.0 on [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), a 321B-parameter mixture-of-experts (MoE) model with about 18B parameters active per token, on one 8x H100 node. It runs the same 1,000 prompts under two parallel layouts and compares them: tensor parallel 8 (TP8), and data parallel 8 with expert parallel 8 (DP8+EP8). Expert parallelism (EP) spreads each MoE layer's experts across GPUs, so no single GPU holds a full copy of the experts.

The safetensors weights are 328.4 GB (about 306 GiB). The full run takes about 26 minutes in the container, including the download; model startup takes the most time. The workload's timeout is 120 minutes.

## Before you run

- **Local disk.** The checkpoint needs about 336 GiB of free space on a local disk. The download stage prints the free space and stops before downloading if there is not enough. It uses the disk set by the `local_scratch_dir` parameter (`/tmp` on the tested node, which had about 28 TB free). Each run downloads the checkpoint again unless the scratch disk still holds it.
- **Output volume.** Results are written to `/Volumes/main/default/air_examples/vllm-expert-parallel-moe-batch-inference`. Replace `output_root` if you do not have access to that volume.
- **Dataset license.** The prompts come from [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca), which is licensed CC BY-NC 4.0: non-commercial use only. The model is MIT licensed.

## Run

```bash
databricks air run -f workload.yaml --watch
```

The workload runs four stages in order: `download`, `generate --layout tp8`, `generate --layout dp8_ep`, and `summary`. Both layouts use the same prompts, the same MoE kernel (`moe_backend: triton`), and the same batched-token budget, so the comparison is like for like.

## Results

For each layout, the run writes `generations.parquet` (the prompt, the output text, and token counts for every prompt) and `metrics.json` to `<output_root>/<MLflow run ID>/<layout>/`. The `summary` stage prints a throughput table and the fraction of greedy outputs that are identical across the two layouts. The same metrics are logged to the AI Runtime MLflow run.

`metrics.json` holds output tokens per second, generation time (measured from a shared start point after warm-up to the last finished generation), and startup time (model load, CUDA graph capture, and compilation).

**KV-cache size is not in `metrics.json`.** vLLM's startup log reports it, so read it from the run log. Each engine prints a line like `GPU KV cache size: <N> tokens, Maximum concurrency for 4,096 tokens per request: <x>x`. TP8 has one engine. DP8+EP8 has eight, one per GPU.

### Full-run results

Measured on one `GPU_8xH100` node with the default parameters: 1000 Alpaca prompts, up to 512 new tokens each, greedy decoding. Output tokens per layout were 226,902 (TP8) and 226,447 (DP8+EP8).

| Layout | Output tokens/s | Generation time (s) | Startup (s) | KV-cache tokens (log, per engine) |
|---|---|---|---|---|
| TP8 | 4,732 | 48.0 | 765 | 867,737 (one engine) |
| DP8+EP8 | 2,556 | 88.6 | 305 | 80,749 (eight engines) |

Under this configuration TP8 generated about 1.9x faster than DP8+EP8.

**Cross-layout output divergence.** Greedy decoding does not make the two layouts produce the same text: in the run above, 14.4% of prompts produced bit-identical outputs. Bit-identical outputs across parallel layouts are not expected — each layout splits matrix multiplications differently, so floating-point reduction orders differ. Compare throughput, not text, across layouts.

### Memory budget

These are estimates from the Hub checkpoint and the model config, with the measured log values from the run above in the last row. They assume 80 GB H100s and `gpu_memory_utilization` 0.90.

| | TP8 | DP8+EP8 |
|---|---|---|
| Weights per GPU (estimate) | about 38 GiB | about 58 GiB |
| Free for KV cache and recurrent state per GPU (estimate) | about 28 GiB | about 9 GiB |
| Measured KV tokens (run log) | 867,737 in one engine | 80,749 per engine |

The measured DP8+EP8 cache is far below the estimate. Under DP8+EP8, each engine holds the full dense attention weights and the full fp32 linear-attention (KDA) recurrent state for every sequence, and only about 9 GiB of each GPU is left for cache.

The checkpoint's main branch at the time of writing is commit `46a050bec13002ad29ee795c16e61cf4fa3c4a9c`.
