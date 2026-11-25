# Long-context SFT with Ulysses sequence parallelism on 8xH100

This recipe splits one long sequence across 8 H100 GPUs with DeepSpeed Ulysses sequence parallelism (SP=8), then fully fine-tunes `Qwen/Qwen3-8B` on `zai-org/LongAlign-10k` at 16K to 128K tokens. It measures tokens per second and peak memory per GPU for SP=8 and for the SP=1 baseline, which runs one sequence per GPU with data parallelism.

The model is trained with DeepSpeed ZeRO-3 in both cases, so the sequence split is the only difference. The recipe measures throughput and memory. It makes no claim about model quality.

## Before you run

- No secrets are needed. The model and dataset are ungated.
- The outputs go to `parameters.output_root`, which is a shared volume (`/Volumes/main/default/air_examples/long-context-sft-ulysses-sequence-parallel-8xh100`). Replace it with another writable volume if you cannot use this one.
- Check the license of `zai-org/LongAlign-10k` before you redistribute anything derived from it. The dataset card does not state one.

## Run

```bash
databricks air run -f workload.yaml --watch
```

The default sweep runs six points, each in its own process group:

| Point | Sequence lengths (tokens) |
|---|---|
| SP=8 (all 8 GPUs on one sequence) | 16384, 65536, 131072 |
| SP=1 (one sequence per GPU) | 16384, 32768, 65536 |

Each point runs one untimed warm-up step and then 3 timed optimizer steps. A point that runs out of memory is recorded as `oom`, and the sweep continues. The sweep exits non-zero if any SP=8 point is not ok.

The run took about 7 minutes in the container, including the model and dataset download.

## How it works

- `train_sp.py` runs one point. It builds sequences of up to 131,072 tokens by concatenating chat-formatted conversations, because most LongAlign conversations are 8K to 64K tokens. Attention is not masked across these boundaries, and the loss covers assistant tokens only.
- Qwen3-8B is trained to 32,768 tokens. Every point uses the YaRN RoPE scaling that the model card documents for 131,072 tokens (`rope_type: yarn`, `factor: 4.0`, `original_max_position_embeddings: 32768`). The scaling is static, so the 16K points also run with it. Both SP settings use the same setting, so the comparison stays like for like.
- With SP=8, `accelerate` (`ParallelismConfig(sp_backend="deepspeed", sp_size=8)`) splits each sequence along the sequence dimension and exchanges heads with all-to-all. Labels are pre-shifted in the loader, and the per-shard losses are combined as in DeepSpeed's ALST tutorial.
- `sweep.py` runs each point as a separate `accelerate launch`. It logs the results to the AIR MLflow run from one process, so no extra runs are created.

## Results

Metrics are logged to the AIR run, keyed by point, for example `sp8_seq131072_tokens_per_s`. Each point also has `_peak_allocated_gb_per_gpu`, `_sec_per_iter`, `_final_loss`, and `_ok` (1 for success, 0 otherwise). The table is written to `<output_root>/<MLFLOW_RUN_ID>/sweep_table.csv` and `sweep_results.json`, with one log per point beside them.

Measured on one `GPU_8xH100` run with `databricks_ai_v6`, at the default parameters. Each figure is the mean over 3 timed steps. The run did not record per-step times, so no standard error is reported. Peak memory is the peak allocated memory per GPU.

| Point | Status | Tokens/s (global) | Seconds per step | Peak GB per GPU |
|---|---|---|---|---|
| SP=8, 16384 | ok | 11,044 | 1.48 | 25.6 |
| SP=8, 65536 | ok | 17,559 | 3.73 | 39.2 |
| SP=8, 131072 | ok | 11,363 | 11.53 | 60.2 |
| SP=1, 16384 | ok | 33,977 | 3.86 | 55.6 |
| SP=1, 32768 | oom | n/a | n/a | n/a |
| SP=1, 65536 | oom | n/a | n/a | n/a |

The SP=1 points at 32K and 64K ran out of memory. Their CUDA errors were allocating 18.55 GiB and 37.09 GiB, which are the fp32 logits for those lengths. The sweep recorded both as `oom` and did not fail the run. At 16K, SP=1 gives more tokens per second than SP=8 on this hardware. SP=8 is the only layout in this sweep that runs 128K.
