# vLLM prefix caching and CPU KV offload on shared-prefix requests

This recipe runs one request stream through vLLM on one H100 in four passes: with prefix caching off, with prefix caching on, with prefix caching plus vLLM's CPU KV offload, and a second prefix-caching-off pass as a noise control. The stream reuses 40 long documents, and the GPU KV cache holds only about a third of them. It reports output tokens per second, wall time, time to first token (TTFT), end-to-end latency, prefix-cache hit rates, and the fraction of greedy outputs that match the baseline pass. The match rate is reported, not treated as proof of equivalent outputs.

## Run

```bash
databricks air run -f workload.yaml --watch
```

The two standard-v6 validations took 12 minutes 5 seconds and 12 minutes 6 seconds in the container for the four passes. Queueing time is not included.

A cheap smoke run for debugging uses reduced settings (fewer documents and requests, shorter prompts, smaller budgets). It exercises the same code paths, but its numbers are not comparable to the default run:

```bash
databricks air run -f workload.yaml --watch \
  --override parameters.num_documents=8 \
  --override parameters.num_requests=64 \
  --override parameters.doc_tokens=2048 \
  --override parameters.gpu_kv_gib=1.0 \
  --override parameters.cpu_offload_cap_gib=4.0
```

## How the workload is set up

- **Model:** `Qwen/Qwen3-8B` (ungated, Apache-2.0), one H100, served by vLLM 0.34.0 on standard environment version `6`. vLLM installs the PyTorch and CUDA libraries it needs.
- **Documents:** 40 synthetic documents, each a fixed random sequence of 4,096 token IDs. Random token IDs keep the workload independent of any dataset, and prefix-cache behavior depends only on token identity and length.
- **Requests:** 800 requests. Each is one document followed by a unique 64-token question, with 64 greedy output tokens. The document for each request is drawn uniformly at random, with repeats, from a seeded generator. Each document is visited about 20 times on average (in the latest run, between 7 and 26 times). The run prints the visit counts.
- **Why the GPU cache is smaller than the workload:** the GPU KV budget is fixed at 8 GiB with `--kv-cache-memory-bytes`, which gives 58,240 tokens. Qwen3-8B stores about 144 KiB of KV per token (36 layers, 8 KV heads, head dim 128, bf16). The 40 document prefixes total 163,840 tokens, about 22.5 GiB, so the GPU holds about 36% of them. The budget is an absolute size, so it stays fixed across configurations. In vLLM 0.34.0, the extensible KV cache is enabled on CUDA "unless kv_cache_memory_bytes is set", so this flag also turns that cache off and sizes the cache from the budget.
- **CPU tier:** the offload configuration uses vLLM 0.34's `OffloadingConnector` with `kv_role: kv_both`. Its `cpu_bytes_to_use` is `min(32 GiB, 0.5 × usable host RAM)`, where usable RAM is the smaller of `MemAvailable` and the cgroup's free memory. The run logs the RAM it measured and the tier size it chose.
- **Configurations:** each pass runs in a fresh vLLM server, with engine startup and warm-up kept outside the timed region:
  - `prefix_off`: `--no-enable-prefix-caching`
  - `prefix_gpu`: `--enable-prefix-caching`
  - `prefix_cpu_offload`: `--enable-prefix-caching` plus the `OffloadingConnector` CPU tier
  - `prefix_off_repeat`: the same settings as `prefix_off`, run last as a control for run-to-run numerical noise
- **Client:** 16 requests in flight at a time. TTFT is measured from each request's send, after it gets a concurrency slot, so it includes server-side queueing.

## What is measured

- Output tokens per second and wall time, from the first send to the last completion.
- TTFT mean and median, and end-to-end latency median. Standard errors (SE) come from a bootstrap over requests, the unit of independence. They do not capture run-to-run variance.
- Prefix-cache hit rates from vLLM's Prometheus counters, taken as the difference between snapshots before and after the timed stream. The counters are in tokens. GPU hit rate is `vllm:prefix_cache_hits` divided by `vllm:prefix_cache_queries`. External hit rate is `vllm:external_prefix_cache_hits` divided by `vllm:external_prefix_cache_queries`. The recipe treats the external counters as CPU-tier hits. That is an inference: the offload configuration is the only one with a KV connector, and the GPU-only configuration reports zero external counters. The direct per-tier breakdown, `vllm:prompt_tokens_cached_by_source` with `source="host"`, is not recorded.
- Output match: the fraction of requests whose greedy output text is identical to the `prefix_off` output for the same request.

## Results

The recipe was validated on one H100 in both the managed Databricks AI environment and standard v6. Each configuration ran once per validation run. The first managed-environment run had three configurations; the others include the `prefix_off_repeat` control. Two earlier smoke runs at reduced settings also passed; in the second, the control matched `prefix_off` on 64 of 64 requests.

Standard-v6 validation run D (final pinned dependencies):

| Configuration | Output tok/s | Wall time (s) | TTFT median ms (SE) | TTFT mean ms (SE) | E2E median ms | GPU prefix hit rate | CPU (external) hit rate | Output match vs `prefix_off` |
|---|---|---|---|---|---|---|---|---|
| `prefix_off` | 444.5 | 115.2 | 543 (0.5) | 664 (8.8) | 2189 | n/a | n/a | baseline |
| `prefix_gpu` | 612.1 | 83.6 | 311 (5.1) | 345 (7.6) | 1633 | 33.7% | n/a | 76.4% (611/800) |
| `prefix_cpu_offload` | 1192.9 | 42.9 | 140 (4.3) | 187 (7.3) | 759 | 33.4% | 90.3% | 72.0% (576/800) |
| `prefix_off_repeat` (control) | 444.3 | 115.2 | 542 (0.8) | 664 (8.9) | 2190 | n/a | n/a | 99.9% (799/800) |

Standard-v6 validation run C (before explicitly pinning PyTorch and aiohttp, which resolved to the same versions):

| Configuration | Output tok/s | Wall time (s) | TTFT median ms (SE) | TTFT mean ms (SE) | E2E median ms | GPU prefix hit rate | CPU (external) hit rate | Output match vs `prefix_off` |
|---|---|---|---|---|---|---|---|---|
| `prefix_off` | 444.3 | 115.2 | 544 (0.7) | 664 (8.8) | 2189 | n/a | n/a | baseline |
| `prefix_gpu` | 609.5 | 84.0 | 324 (4.0) | 372 (8.4) | 1641 | 33.5% | n/a | 77.1% (617/800) |
| `prefix_cpu_offload` | 1198.0 | 42.7 | 139 (4.4) | 181 (7.0) | 756 | 33.3% | 90.3% | 75.4% (603/800) |
| `prefix_off_repeat` (control) | 443.9 | 115.3 | 544 (0.6) | 665 (8.9) | 2192 | n/a | n/a | 100% (800/800) |

Managed-environment validation run B (four passes):

| Configuration | Output tok/s | Wall time (s) | TTFT median ms (SE) | TTFT mean ms (SE) | E2E median ms | GPU prefix hit rate | CPU (external) hit rate | Output match vs `prefix_off` |
|---|---|---|---|---|---|---|---|---|
| `prefix_off` | 440.8 | 116.2 | 547 (0.6) | 669 (8.9) | 2209 | n/a | n/a | baseline |
| `prefix_gpu` | 605.1 | 84.6 | 327 (3.1) | 380 (8.4) | 1640 | 33.5% | n/a | 76.3% (610/800) |
| `prefix_cpu_offload` | 1193.7 | 42.9 | 140 (3.9) | 187 (7.4) | 754 | 33.1% | 90.3% | 72.6% (581/800) |
| `prefix_off_repeat` (control) | 440.9 | 116.1 | 547 (0.6) | 663 (8.8) | 2208 | n/a | n/a | 98.5% (788/800) |

Managed-environment validation run A (the earlier three-configuration run, before the control was added):

| Configuration | Output tok/s | Wall time (s) | TTFT median ms (SE) | TTFT mean ms (SE) | E2E median ms | GPU prefix hit rate | CPU (external) hit rate | Output match vs `prefix_off` |
|---|---|---|---|---|---|---|---|---|
| `prefix_off` | 442.6 | 115.7 | 545 (0.8) | 666 (8.9) | 2199 | n/a | n/a | baseline |
| `prefix_gpu` | 608.6 | 84.1 | 314 (4.4) | 356 (8.3) | 1644 | 33.5% | n/a | 77.5% (620/800) |
| `prefix_cpu_offload` | 1192.9 | 42.9 | 140 (4.5) | 188 (7.4) | 756 | 33.1% | 90.3% | 74.1% (593/800) |

Host RAM measured at startup was 215.8 GiB usable in runs C and D and 216.6 GiB in run B (the cgroup's free memory was the smaller bound). CPU tier used: 32.0 GiB. GPU KV cache reported by vLLM: 58,240 tokens, 35.6% of the 163,840 document-prefix tokens.

What the runs show:

- The offload configuration produced about 2.7 times the output tokens per second of `prefix_off` and about twice that of `prefix_gpu`, in all runs. Its median TTFT was 140 ms against 311 ms for `prefix_gpu` in run D. The gaps are far larger than the standard errors.
- The gain depends on the deliberately small GPU budget. The 8 GiB budget holds about 36% of the distinct prefixes, so the GPU-only cache is under pressure. If the budget held all 40 prefixes (about 22.5 GiB), the GPU-only cache would be expected to reuse most prefixes, and most of the offload gain would be expected to disappear. This is an untested expectation; this recipe does not measure that configuration.
- Output match: the control matched `prefix_off` on 98.5% of requests in run B, 100% in run C, and 99.9% in run D, so the same settings do not always reproduce greedy outputs exactly across server instances. The two caching configurations matched on 72–78%. That gap is much larger than the controls', which are three control pairs, not a measured distribution: with caching on, outputs differ from `prefix_off` for roughly a quarter of requests, with or without the CPU tier. These runs did not isolate the cause. The match rate is not evidence that outputs are equivalent or wrong.

Known log noise, checked in the logs:

- Every server log in runs C and D has a FlashInfer warning that precompiled kernels are missing. The server logs select the FLASH_ATTN backend for attention, and the FlashInfer sampler is disabled, so the warning is not on the attention path used here.
- All four server logs in run C and three in run D contain an `AsyncLLM output_handler failed` traceback ending in `EngineDeadError`. It follows the SIGTERM shutdown that stops each server, after the timed stream has finished, so it does not affect the measurements.

Outputs are written to the run's directory under `/Volumes/main/default/air_examples/vllm-prefix-caching-cpu-kv-offload-qwen3-8b/<MLFLOW_RUN_ID>/`: `config.json`, `results.json`, `results.csv`, `outputs_prefix_off.jsonl`, and one server log per configuration.
