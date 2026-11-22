# Multi-turn tool-use reinforcement learning with TRL's AsyncGRPOTrainer

This recipe trains `Qwen/Qwen3-1.7B` with LoRA to answer SQL questions by calling tools over several turns. It runs 80 steps of GRPO (Group Relative Policy Optimization) with [TRL](https://github.com/huggingface/trl) 1.18.0's `AsyncGRPOTrainer` on `GPU_8xH100` and takes about 50 minutes in the container.

Each rollout gets a fresh in-memory SQLite database. The model explores it with a read-only `run_query` tool, then submits an answer with `submit_answer`. The first answer is final. The reward is 1 when the submitted query returns the same rows as the gold query.

`AsyncGRPOTrainer` separates generation from training. A `vllm serve` process on GPU 0 generates rollouts while the trainer on GPU 1 trains. The trainer has no colocate mode and can't share a GPU with the server. The recipe uses two of the node's eight GPUs.

The recipe trains on 2,000 questions from the train split. It measures task success on 300 held-out questions from the test split, before and after training, with the same tools and reward. Both sets skip questions with any of these:

- A database script longer than 1,500 characters. This keeps prompts within the server's 8,192-token window.
- A gold query that uses `LIMIT`. Without a full `ORDER BY`, it has no single correct answer.
- A gold result that is empty or has more than 200 rows.
- A database or gold query that fails to run.

The recipe logs reward curves and before and after success to MLflow.

## Before you run

- Data: [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql) (Apache 2.0, public, ungated).
- Model: [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) (Apache 2.0, public, ungated). Its chat template supports tool calls.
- The recipe writes to the shared volume `/Volumes/main/default/air_examples/trl-async-grpo-tool-use`. Replace it with another writable volume in `workload.yaml` if you cannot use this one.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run writes `metrics.json` and the LoRA adapter (`adapter/`) to `<output_root>/<MLflow run ID>/`. A validation run on an 8×H100 node produced these results. It used model revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` and dataset revision `740ab236e64503fba51be1101df7a1be83bf455d`.

| Metric | Base model | After 80 GRPO steps |
|---|---|---|
| Held-out test success (300 questions) | 30.7% ± 2.7 | 67.0% ± 2.7 |
| Submitted an answer | 48.3% ± 2.9 | 95.0% ± 1.3 |
| Tool calls per question | 1.66 ± 0.04 | 2.14 ± 0.04 |
| Tool calls that returned an error | 6.2% | 4.8% |

Each success rate is a proportion over 300 questions. Its standard error is about 2.7 points. Treating the two evaluations as independent, the standard error of the difference is about 3.8 points. The 36.3-point gain is about 10 times that. The test split is used only for this report. No checkpoint is selected on it.

Training reward rose from 0.58 (mean of steps 1 to 10) to 0.74 (steps 71 to 80). The fraction of groups where all 8 completions got the same reward rose from 0.81 to 0.89. Those groups carry no learning signal. Later steps learn from fewer groups. Entropy fell from 0.037 to 0.028. Generation was the bottleneck. Each step waited about 30 seconds on average for rollouts. The forward and backward pass took about 2 seconds.

The run logs these metrics to the AIR MLflow run:

- `eval/success`, `eval/submit_rate`, `eval/tool_calls_per_episode` and `eval/tool_error_rate` at step 0 (the base model) and at the final step (the trained adapter), each with an `_se` companion holding its standard error.
- Reward and generation metrics for every step, including `reward`, `frac_zero_std_groups`, `perf/rollout_wait_s`, `rollout/backpressure_s` and `rollout/queue_size`. Read these with the guide in TRL's [async GRPO documentation](https://huggingface.co/docs/trl/async_grpo_trainer). They show whether the run is generation-bound or trainer-bound.
