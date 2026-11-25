# GRPO on GSM8K math with Hugging Face TRL

This recipe trains `Qwen/Qwen2.5-0.5B-Instruct` with GRPO (Group Relative Policy Optimization) on GSM8K math questions on one A10, using [TRL](https://github.com/huggingface/trl) and a LoRA adapter. For each prompt, GRPO samples 8 answers and rewards each one: 1 for a correct final answer, plus 0.2 if it contains the `####` marker the prompt asks for. It then pushes the model toward the answers that scored above their group's average.

The run trains for 100 steps and measures greedy accuracy on all 1,319 GSM8K test questions before and after training. It takes about 37 minutes in the container, plus provisioning and queueing.

## Before you run

The default `output_root` is the shared Unity Catalog volume `/Volumes/main/default/air_examples/trl-grpo-qwen2.5-0.5b-gsm8k`. The volume must already exist and be writable. To use another volume, run:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints accuracy before and after training and the mean reward per logging interval. It writes `metrics.json` and the LoRA adapter (`grpo_adapter/`) to `<output_root>/<MLflow run ID>/`. It also logs the training loss and reward curves and the final metrics to the MLflow run, and uploads `metrics.json`.

A validation run on an A10 produced these results:

| Metric | Value |
|---|---|
| GSM8K accuracy, base model | 36.4% |
| GSM8K accuracy, after GRPO | 42.3% |
| Mean reward, first → last logging interval | 0.27 → 0.61 |
| Training time (100 steps) | 25 minutes |

With 1,319 questions, each accuracy has a standard error of about 1.3 points, and their difference about 1.9 points if the two evaluations are treated as independent. The 5.9-point gain is about 3 times that. The rising reward shows GRPO is learning. The test set is only used for this report. No checkpoint is selected on it.
