# Quantized GGUF inference with llama.cpp on one A10, versus bf16 vLLM

This recipe runs Qwen3-8B on one 24 GB A10 in bf16 with vLLM, and in three GGUF quantizations with llama.cpp (Q8_0, Q4_K_M and Q2_K). It measures GSM8K accuracy, output tokens per second and peak GPU memory for each, on the same card, with the same questions, chat template, greedy decoding and request concurrency.

The comparison covers the trade-off between accuracy, throughput and memory for k-quants on a small GPU. It does not claim that either engine is faster in general.

## Before you run

The run writes its results to the Unity Catalog volume directory `/Volumes/main/default/air_examples/llama-cpp-gguf-quantization-a10`, which must already exist and be writable. The models and the dataset are public and ungated, so no Hugging Face token is needed. To write results elsewhere, pass another existing writable volume directory:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

The recipe builds llama.cpp with CUDA for sm_86 from the source vendored in the pinned `llama-cpp-python==0.3.37` source distribution (llama.cpp commit `a956a852c`). The package index serves that source, so no GitHub clone is needed. The Python wheel is not installed.

Time in the container at default parameters: about 64 minutes, including the llama.cpp build, the model downloads and the Q2_K quantization. The build time is not measured separately. Queue time is not included.

## What the run does

1. Builds `llama-server` and `llama-quantize`.
2. Downloads `Qwen/Qwen3-8B` (bf16), and the official `Qwen/Qwen3-8B-GGUF` Q8_0 and Q4_K_M files. The revisions are `b968826d9c46dd6066d109eabc6255188de91218` (Qwen3-8B) and `7c41481f57cb95916b40956ab2f0b139b296d974` (Qwen3-8B-GGUF).
3. Creates Q2_K by requantizing the Q8_0 file with `llama-quantize --allow-requantize`. The official repository has no 2-bit file. Q8_0 is close to lossless, so the result is close to quantizing from bf16 but not identical. No importance matrix is used, so this is a plain Q2_K. This recipe does not measure a calibrated 2-bit quant.
4. For each configuration, starts a server, sends the same 500 GSM8K test questions with 16 requests in flight, and shuts the server down before the next one starts:
   - bf16 on `vllm serve` (`--max-model-len 2048`, `--gpu-memory-utilization 0.90`, `--max-num-seqs 16`).
   - Each GGUF on `llama-server` (`-np 16`, `-c 16384`, `-fa on`, `--jinja`).

Every request uses the same system prompt, `temperature 0`, `max_tokens 512`, and `enable_thinking: false`. The thinking switch is set explicitly because Qwen3 thinks by default. vLLM applies the chat template from the bf16 model's Hugging Face tokenizer. `llama-server` (`--jinja`) applies the template embedded in each GGUF file. The two rendered prompts were not compared directly.

The GGUF replies begin with an empty `<think></think>` block: 500 of 500 replies for each GGUF configuration, and none for bf16. The block is empty, so the answer parser is unaffected.

Accuracy compares the number after `####` (or the last number in the reply) with the gold answer as a number. Each configuration gets a paired difference from bf16 on the same questions. Its standard error is the standard error of the mean per-question difference.

The 500 questions are a seeded random sample (seed 0) of the 1,319-question test split. The sample keeps the container time for four configurations on one A10 within the recipe's time budget. The wall time of each pass in the default run was 335.8 s (bf16), 294.3 s (Q8_0), 238.0 s (Q4_K_M) and 250.5 s (Q2_K), so these four passes took about 19 minutes in total.

Output tokens per second is total completion tokens divided by the wall time of the whole pass. The pass includes the ramp-up and tail of each run.

Peak GPU memory is the maximum `memory.used` on GPU 0 sampled by `nvidia-smi` while the server was running. vLLM preallocates most of the card at startup (`gpu_memory_utilization 0.90`), so its figure reflects that setting more than the model's needs. In the run below, vLLM's log reports 15.27 GiB of weights and a KV cache of 24,160 tokens, which is 11.8x concurrency at 2,048 tokens per request. The GGUF figures include the llama-server KV cache allocated for its configured context (`-c 16384` split over 16 slots). The llama-server logs in this run do not report the KV size, so the split between weights and cache is not measured for the GGUF configurations.

## Results

The run writes to `/Volumes/main/default/air_examples/llama-cpp-gguf-quantization-a10/<MLFLOW_RUN_ID>/`:

- `results.csv`: one row per configuration, with accuracy, its standard error, the paired difference from bf16 and its standard error, discordant pair counts, truncated replies, output tokens per second, wall time, peak GPU memory and weight size.
- `predictions/<config>.jsonl`: every reply, its parsed number and whether it was correct.
- `servers/<config>.log`: each server's log.

The AIR MLflow run has the same figures as `gsm8k_accuracy_<config>`, `output_tokens_per_second_<config>`, `peak_gpu_memory_mib_<config>`, `weights_gb_<config>` and `accuracy_diff_vs_bf16_<config>`, plus the `results.csv` artifact.

| Config | Weights (GB) | GSM8K accuracy (± SE) | Diff. vs bf16 (± SE) | Output tokens/s | Peak GPU memory (MiB) |
|---|---|---|---|---|---|
| bf16 (vLLM) | 16.38 | 0.926 ± 0.012 | - | 394 | 19,781 |
| Q8_0 | 8.71 | 0.926 ± 0.012 | +0.000 ± 0.007 | 448 | 10,401 |
| Q4_K_M | 5.03 | 0.914 ± 0.013 | −0.012 ± 0.010 | 567 | 7,187 |
| Q2_K | 3.28 | 0.628 ± 0.022 | −0.298 ± 0.022 | 589 | 5,665 |

These are the measured values from one default run on a single A10, 500 questions each. Each configuration answered the same 500 questions.

The paired comparison uses discordant pair counts (bf16 right and config wrong, then the reverse): Q8_0 6 and 6, Q4_K_M 16 and 10, Q2_K 155 and 6. The Q8_0 and Q4_K_M differences are within about one standard error of zero. The Q2_K loss is large and clearly outside its standard error.

The Q2_K drop applies to this file only: a plain Q2_K requantized from the Q8_0 GGUF, with no importance matrix. It is not a measurement of 2-bit GGUF quantizations in general. Part of the loss is runaway generation: 78 of 500 Q2_K replies hit the 512-token limit, against 8 to 11 for the other configurations. A truncated reply has no `####` answer, so it is scored from the last number in the text. The results do not separate those replies from wrong final answers.

Throughput compares two engines with their own defaults at the same request concurrency, so the gap between configurations mixes engine differences with quantization effects. For example, Q8_0 at 448 tokens/s versus bf16 at 394 tokens/s is not a pure quantization effect. bf16 had a KV cache of 24,160 tokens. The tokens/s figures are not tuned best cases for either engine, and the three llama.cpp configurations were not run at other concurrency levels.
