# GRPO on GSM8K math with Hugging Face TRL

This recipe trains `Qwen/Qwen2.5-0.5B-Instruct` with GRPO (Group Relative Policy Optimization) on GSM8K math questions on one A10, using [TRL](https://github.com/huggingface/trl) and a LoRA adapter. For each prompt, GRPO samples 8 answers and rewards each one: 1 for a correct final answer, plus 0.2 for ending with the requested `#### <number>` format. It then pushes the model toward the answers that scored above their group's average.

The run trains for 100 steps and measures greedy accuracy on all 1,319 GSM8K test questions before and after training. It takes about 36 minutes in the container, plus provisioning and queueing.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints accuracy before and after training and the mean reward per logging interval. It writes `metrics.json` and the LoRA adapter (`grpo_adapter/`) to `<output_root>/<MLflow run ID>/`. It also logs the metrics to the MLflow run and uploads `metrics.json`.

A validation run on an A10 produced these results:

| Metric | Value |
|---|---|
| GSM8K accuracy, base model | 36.3% |
| GSM8K accuracy, after GRPO | 39.3% |
| Mean reward, first → last logging interval | 0.31 → 0.60 |
| Training time (100 steps) | 24 minutes |

With 1,319 questions, each accuracy has a standard error of about 1.3 points. The test set is only used for this report. No checkpoint is selected on it.

To try other GRPO variants, change `loss_type` to `grpo`, `dr_grpo` or `bnpo`. For a larger model, switch `accelerator_type` to `GPU_1xH100`. Neither change has been validated.
