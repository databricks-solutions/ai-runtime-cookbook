# Quantize to NVFP4 with Model Optimizer, then batch inference with vLLM on B300

This recipe quantizes `Qwen/Qwen3-32B` to NVFP4 (4-bit floating point) with [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) post-training quantization (PTQ). Calibration uses 256 CNN/DailyMail articles. The recipe then runs vLLM offline batch inference on all 1,319 GSM8K test questions with the quantized model and the bf16 original, side by side on two B300 GPUs. It compares output throughput and accuracy.

The workload takes about 11 minutes in the container. `quantize.py` also accepts `quant_format: fp8`, but only NVFP4 has been validated.

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
| Qwen3-8B bf16 | 16.4 GB | 32,557 | 1.00× | 90.9% |
| Qwen3-8B NVFP4 | 6.4 GB | 26,951 | 0.83× | 88.6% |

An earlier Qwen3-32B run measured a 1.55× speedup, so expect some run-to-run variation. NVFP4 was faster for Qwen3-32B but slower than bf16 for Qwen3-8B. For a small model on B300, expect NVFP4 to save memory rather than time. Set `model_name: Qwen/Qwen3-8B` to reproduce the smaller comparison.

Notes on the environment:

- The command removes `databricks_ai_v6`'s preinstalled `flash-attn`, which is compiled for the original PyTorch that vLLM replaces.
- FlashInfer compiles some kernels when vLLM starts, and they need `curand.h`. That header ships only in the NVIDIA pip wheels in this environment. `run.sh` copies the `curand` headers to a directory on `CPATH`.
