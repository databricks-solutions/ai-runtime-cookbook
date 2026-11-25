# Quantize to NVFP4 with Model Optimizer, then batch inference with vLLM on B300

This recipe quantizes `Qwen/Qwen3-32B` to NVFP4 (4-bit floating point) with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) post-training quantization (PTQ). Calibration uses 256 CNN/DailyMail articles. The recipe then runs vLLM offline batch inference on all 1,319 GSM8K test questions with the quantized model and the bf16 original, side by side on two B300 GPUs. It compares output throughput and accuracy.

The whole run takes about 11 minutes.

## Before you run

The run writes generations to `output_root` in `workload.yaml`, under a subdirectory named for the MLflow run ID. That Unity Catalog volume must already exist and be writable. To write elsewhere, pass another existing writable volume directory:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints throughput and GSM8K accuracy for both models. It writes `generations_bf16.jsonl` and `generations_quantized.jsonl` to `<output_root>/<MLflow run ID>/`. `summarize.py` logs the throughput, speedup and accuracy metrics to the MLflow run.

Validation runs on 8xB300 produced these results:

| Model | Weights | Output tokens/s | Speedup | GSM8K accuracy |
|---|---|---|---|---|
| Qwen3-32B bf16 | 65.5 GB | 9,628 | 1.00× | 93.4% |
| Qwen3-32B NVFP4 | 20.7 GB | 14,190 | 1.47× | 92.6% |

An earlier Qwen3-32B run measured a 1.55× speedup, so expect some run-to-run variation.

Notes on the environment:

- The command removes `databricks_ai_v6`'s preinstalled `flash-attn`, which is compiled for the original PyTorch that vLLM replaces.
- FlashInfer compiles some kernels when vLLM starts, and they need `curand.h`. That header ships only in the NVIDIA pip wheels in this environment. `run.sh` copies the `curand` headers to a directory on `CPATH`.
