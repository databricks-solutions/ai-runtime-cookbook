# Multi-turn tool-use reinforcement learning with TRL's AsyncGRPOTrainer

This recipe trains `Qwen/Qwen3-1.7B` with LoRA to answer SQL questions by calling tools over multiple turns. Each rollout gets a fresh in-memory SQLite database. The model explores it with a read-only `run_query` tool, then submits an answer with `submit_answer`. The reward is 1 when the submitted query returns the same rows as the gold query. It uses [TRL](https://github.com/huggingface/trl) 1.18.0's `AsyncGRPOTrainer`, which decouples generation from training: a `vllm serve` process on GPU 0 generates rollouts while the trainer on GPU 1 trains. `AsyncGRPOTrainer` has no colocate mode, so the server and the trainer need different GPUs, and no single-GPU shape can hold both. The recipe therefore runs on `GPU_8xH100` and uses two of the eight GPUs. It runs 80 steps of GRPO (Group Relative Policy Optimization) and takes about 47 minutes in the container.

The recipe measures task success on 300 held-out test questions before and after training, through the same tool loop. It logs reward curves and before and after success to MLflow.

## Before you run

- Data: [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0, public, ungated). The recipe trains on rows from the train split and evaluates on rows from the test split.
- Model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0, public, ungated). Its chat template supports tool calls.
- The recipe writes to the shared volume `/Volumes/main/default/air_examples/trl-async-grpo-tool-use`. Replace it with another writable volume in `workload.yaml` if you cannot use this one.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run writes `metrics.json` and the LoRA adapter (`adapter/`) to `<output_root>/<MLflow run ID>/`. A validation run on an 8×H100 node produced these results; training took about 41 of the 47 container minutes:

| Metric | Base model | After 80 GRPO steps |
|---|---|---|
| Held-out test success (300 questions) | 30.3% ± 2.6 | 69.0% ± 2.7 |
| Submitted an answer | 48.0% ± 2.9 | 97.7% ± 0.9 |
| Tool calls per question | 1.65 ± 0.04 | 2.23 ± 0.06 |
| Tool calls that returned an error | 7.7% | 3.7% |

Each success rate is a proportion over 300 questions, so its standard error is about 2.7 points, and the standard error of the difference is about 3.8 points treating the two evaluations as independent. The 38.7-point gain is about 10 times that, so the improvement is clear. The test split is only used for this report; no checkpoint is selected on it.

Training reward rose from about 0.56 (mean of steps 1 to 10) to about 0.72 (steps 71 to 80). Most groups stopped carrying a learning signal along the way: the fraction of groups where all 8 completions got the same reward rose from about 0.78 to about 0.90, so later steps learn from fewer groups. Entropy fell from about 0.04 to about 0.03. The logs show generation was the bottleneck: each step waited 15 to 45 seconds for rollouts, and the forward and backward pass took about 2 seconds.

The run logs these metrics to the AIR MLflow run:

- `eval/success`, `eval/submit_rate`, `eval/tool_calls_per_episode` and `eval/tool_error_rate` at step 0 (the base model) and at the final step (the trained adapter), each with an `_se` companion holding its standard error.
- Reward and generation metrics for every step, including `reward`, `frac_zero_std_groups`, `perf/rollout_wait_s`, `rollout/backpressure_s` and `rollout/queue_size`. Read these with the guide in TRL's [async GRPO documentation](https://huggingface.co/docs/trl/async_grpo_trainer). They show whether the run is generation-bound or trainer-bound.
