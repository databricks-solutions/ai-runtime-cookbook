# Knowledge distillation: vLLM teacher, TRL student

This recipe distills math reasoning from `Qwen/Qwen2.5-7B-Instruct` (the teacher) into `Qwen/Qwen2.5-0.5B-Instruct` (the student) on one H100.

The teacher samples up to 4 solutions for each of 4,000 GSM8K training questions using vLLM batch inference. The recipe keeps the first solution with a correct final answer. The student then runs 2 epochs of supervised fine-tuning on those solutions with TRL, with loss on the solution tokens only. The recipe measures the student's greedy accuracy on all 1,319 GSM8K test questions before and after. The run takes about 10 minutes in the container, plus provisioning and queueing.

## Before you run

The run writes `summary.json` to `output_root` in `workload.yaml`. That Unity Catalog volume must already exist and be writable. To write elsewhere, pass another existing writable volume directory:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints each stage's results. It writes `summary.json` to `<output_root>/<MLflow run ID>/` and logs the metrics to MLflow.

A validation run on an H100 produced these results:

| Stage | Result |
|---|---|
| Teacher batch generation | 3,882 of 4,000 questions solved correctly, in 176 s |
| Student fine-tuning | 3,882 examples × 2 epochs, in 105 s |
| Student GSM8K accuracy (1,319 questions) | 47.1% → 51.2% |

With 1,319 questions, each accuracy has a standard error of about 1.4 points, and their difference about 1.9 points if the two evaluations are treated as independent. The 4.1-point gain is about twice that. The test set is only used for this report. No checkpoint is selected on it.

Notes:

- `train_student.py` disables PyTorch's cuDNN attention backend. An earlier validation run with it enabled hit NaN gradients partway through the first epoch and produced a broken student. The training also stops with an error if gradients become non-finite.
- Each stage runs in its own process. This fully releases GPU memory between vLLM and training.
