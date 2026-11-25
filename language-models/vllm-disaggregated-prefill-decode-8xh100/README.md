# Disaggregated prefill and decode with vLLM NixlConnector on one 8x H100 node

This recipe runs the same open-loop request stream against two serving layouts on the same 8 GPUs. In the colocated layout, each GPU runs a full vLLM instance. In the disaggregated layout, prefill and decode run in separate instances, and each request's KV cache moves between them through vLLM's `NixlConnector` (UCX on the same host). In an earlier smoke run on `databricks_ai_v6`, KV reads used UCX's `cuda_ipc` transport (see Results). It reports time to first token (TTFT), inter-token latency (ITL), and output tokens per second for each layout.

The model is [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) (Apache-2.0, ungated). Each instance runs on one H100 with tensor parallel size 1. The request stream has Poisson arrivals at 24 requests per second for 150 seconds. 20% of prompts have 4,096 tokens and the rest have 256. Every request asks for 256 output tokens with greedy decoding and `ignore_eos`, so both layouts generate the same number of tokens.

## Before you run

- **Output volume.** Results go to `/Volumes/main/default/air_examples/vllm-disaggregated-prefill-decode-8xh100`. Replace `output_root` if you do not have access to that volume.
- **Layouts.** The default runs both layouts one after the other, because each needs all 8 GPUs. Run `--override parameters.layouts=disagg` to debug one layout.
- **Time.** The standard v6 default run took 10 minutes 32 seconds in the container, excluding queue time. It included both rounds of instance startup. The workload timeout is 90 minutes so that queue time fits.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

Each layout writes `requests.jsonl` (one line per request, with every token timestamp gap) and `summary.json`. The run also writes `comparison.json` and `config.json` to `<output_root>/<MLflow run ID>/`. The same summary metrics are logged to the AI Runtime MLflow run, prefixed with `colocated/` or `disagg/`. Differences are logged as `disagg_minus_colocated/...`.

Each latency metric has a bootstrap standard error (`*_se`, 200 resamples of requests). The standard error of a difference is the square root of the sum of the two squared standard errors. This treats the two layouts as independent samples. They are not, because they share one request stream. The approximation is conservative for that reason.

Measured on one `GPU_8xH100` node with the default parameters and standard v6. Values are p50 / p99 with bootstrap standard errors in parentheses. Both layouts served all 3,209 measured requests, with no failures.

| Metric | Colocated (8 x TP=1) | Disaggregated (3 prefill + 5 decode) |
|---|---|---|
| TTFT p50 (ms) | 27.03 (0.10) | 65.13 (0.13) |
| TTFT p99 (ms) | 123.48 (0.24) | 224.64 (6.08) |
| ITL p50 (ms) | 7.31 (0.005) | 7.40 (0.006) |
| ITL p99 (ms) | 14.74 (1.17) | 10.78 (0.02) |
| Output tokens/s | 6,234.9 | 6,234.6 |
| Achieved request rate (req/s, offered 24.0) | 24.68 | 24.68 |
| Failed requests | 0 | 0 |

Both layouts kept up with the offered rate (`keeps_up: true`). Short-prompt and long-prompt TTFT p99 were 33.2 and 125.5 ms for colocated, and 148.1 and 257.3 ms for disaggregated. Long-prompt ITL p99 was 24.6 ms for colocated and 10.9 ms for disaggregated.

Disaggregated minus colocated, with the standard error of the difference:

| Metric | Difference (ms) | SE |
|---|---|---|
| TTFT p50 | +38.09 | 0.16 |
| TTFT p99 | +101.16 | 6.08 |
| ITL p50 | +0.083 | 0.007 |
| ITL p99 | -3.96 | 1.17 |

Output tokens/s differ by -0.3 (6,234.6 against 6,234.9). This run does not log a standard error for throughput.

At this load, disaggregation shortened ITL p99 by about 4 ms and added about 38 ms to median TTFT and about 101 ms to p99 TTFT. Throughput matched, because both layouts completed the same offered load.

**Earlier default runs on `databricks_ai_v6`.** Runs 2 and 3 used the same parameters as run 4. Run 2 had a FlashInfer header workaround, whose logs showed no JIT compilation, and run 3 removed it. Run 4 changed only comments. Results per earlier run (SE):

| Metric | Run 2 colocated | Run 2 disaggregated | Run 3 colocated | Run 3 disaggregated | Run 4 colocated | Run 4 disaggregated |
|---|---|---|---|---|---|---|
| TTFT p50 (ms) | 25.88 (0.08) | 62.49 (0.12) | 26.35 (0.09) | 64.12 (0.13) | 25.90 (0.09) | 62.55 (0.12) |
| TTFT p99 (ms) | 122.95 (0.33) | 222.56 (6.05) | 122.92 (0.36) | 221.52 (6.19) | 122.80 (0.29) | 220.66 (6.93) |
| ITL p50 (ms) | 7.06 (0.005) | 7.11 (0.005) | 7.31 (0.005) | 7.40 (0.005) | 7.06 (0.005) | 7.13 (0.005) |
| ITL p99 (ms) | 12.93 (0.34) | 10.46 (0.03) | 14.43 (0.91) | 10.64 (0.02) | 13.41 (0.37) | 10.26 (0.02) |
| Output tokens/s | 6,237.6 | 6,237.2 | 6,234.8 | 6,233.8 | 6,237.6 | 6,237.1 |
| ITL p99 difference (disagg minus colocated, ms) | -2.5 | | -3.8 | | -3.1 | |

TTFT, ITL p50, and throughput were similar across the three runs. Colocated ITL p99 was 12.9, 14.4 and 13.4 ms, and colocated long-prompt ITL p99 was 17.0, 24.3 and 19.0 ms. The colocated tail varies by more than its bootstrap SE from run to run, so treat the ITL p99 difference as approximate. Disaggregated ITL p99 was stable at 10.3 to 10.6 ms.

**Transport check (earlier `databricks_ai_v6` smoke run, not the standard v6 run).** The smoke run set `UCX_PROTO_INFO=y`. Its decode log shows the KV reads as `remote memory read by ucp_get into cuda/GPU0 from cuda/dev[8]`, with a `cuda_ipc/cuda` zero-copy protocol. The `tcp/eth0` entries in the same log are for small control messages. The standard v6 run did not enable `UCX_PROTO_INFO`, so its transport was not logged separately.

Earlier container times: 12 minutes 7 seconds (run 2), 11 minutes 43 seconds (run 3), and 11 minutes 48 seconds (run 4). The smoke run took 5 minutes 22 seconds, with 40-second arrivals at 8 requests per second.

**Shutdown errors.** The standard v6 decode log includes `AsyncLLM output_handler failed` and `EngineDeadError` after the harness sends SIGTERM during teardown. Teardown starts only after the last request of a layout finishes and before its summary is written. The run reports SUCCESS, and every measured request completed without error.

**FlashInfer.** The FlashInfer sampler is off; a standard v6 instance log reports FlashAttention version 3. The workload does not add FlashInfer header workarounds.

## Notes on the design

- **Why disaggregate.** A 4,096-token prefill can delay decode steps for other requests on the same GPU. Separating the phases might shrink the ITL tail, at the cost of a KV transfer on the TTFT path. This recipe measures that trade-off at one load.
- **Load in this recipe was not saturated.** Both layouts kept up with 24 requests per second, so both layouts completed the same offered load, and throughput matched. The run does not test saturation, so it cannot show a throughput difference. Conditions where disaggregation may help, none tested here: near saturation, where long prefills interfere with decode on colocated GPUs; higher request rates; and inter-token latency service-level objectives.
- **Split.** The 3 prefill and 5 decode split is an estimate (prefill at about one third of GPU time for this mix), not a measurement. Only this split was run. This recipe does not recommend any other split.
- **Proxy.** The router in `disagg_router.py` sends each request to a prefill instance with `max_tokens=1` and `kv_transfer_params.do_remote_decode`. It then sends the request to a decode instance with the `kv_transfer_params` returned by prefill. It follows vLLM's NIXL integration toy proxy at v0.34.0. The colocated layout uses the same router with round-robin dispatch. A missing `kv_transfer_params` fails the request, so decode cannot silently recompute the prompt. The workload uses `kv_load_failure_policy: fail` for the same reason.
- **Failures stop the run.** Any failed request, including one in the warm-up window, raises an error. Failed requests are never silently dropped from the metrics.
- **Load.** Check `achieved_rate_rps` against `offered_rate_rps` in each `summary.json`. If the rate is below 95% of the offered rate, that layout did not keep up, and the latencies are not comparable. Lower `arrival_rate` and rerun.

## Environment

- Standard AIR environment version `"6"`, with `vllm==0.34.0` and its required `torch==2.14.0` declared as dependencies.
- `nixl==1.5.2` is the version vLLM 0.34 pins for `NixlConnector`. The package installs the CUDA 12 and CUDA 13 backends. Each bundles UCX.
- `nvidia-ml-py==13.615.71` provides the NVML bindings used by vLLM.
- `VLLM_USE_FLASHINFER_SAMPLER=0` keeps the FlashInfer sampler off. Its JIT compile needs CUDA headers outside the default include path.
