# SGLang batch inference with EAGLE-3 speculative decoding

This recipe runs [SGLang](https://github.com/sgl-project/sglang) offline batch inference on 256 GSM8K questions with `Qwen/Qwen3-8B` on one H100, twice. The first pass is plain decoding. The second pass uses the `AngelSlim/Qwen3-8B_eagle3` EAGLE-3 draft head: the draft proposes several tokens, and the target model verifies them in one forward pass. The recipe compares throughput and reports the mean accept length (tokens produced per target-model forward pass).

The run limits concurrency to 8 requests (`max_running_requests`), the latency-bound regime where speculative decoding helps most. The whole run takes about 8 minutes.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints the speedup and accept length. It writes `results.json` to `<output_root>/<MLflow run ID>/` and logs the metrics to MLflow.

A validation run on an H100 produced these results:

| Mode | Output tokens/s | Mean accept length |
|---|---|---|
| Baseline | 1,136 | 1.00 |
| EAGLE-3 (3 steps, top-1, 4 draft tokens) | 2,007 | 2.24 |

EAGLE-3 was 1.77× faster. 48% of greedy outputs were identical to the baseline. Speculative decoding doesn't change the target model's output distribution. However, bf16 kernels round slightly differently at different batch shapes, and long greedy generations can drift apart after a small difference. Raise `max_running_requests` to see the speedup shrink as the GPU becomes compute-bound.

Notes on the environment:

- The command removes `databricks_ai_v6`'s preinstalled `flash-attn` (compiled for the original PyTorch) and legacy `pynvml` (which shadows `nvidia-ml-py`).
- SGLang's `tokenizers` pin requires `huggingface_hub` < 2.
