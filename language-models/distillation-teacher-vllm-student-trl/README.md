# Knowledge distillation: vLLM teacher, TRL student

This recipe distills math reasoning from `Qwen/Qwen2.5-7B-Instruct` (the teacher) into `Qwen/Qwen2.5-0.5B-Instruct` (the student) on one H100.

The teacher samples up to 4 solutions for each of 4,000 GSM8K training questions using vLLM batch inference. The recipe keeps the first solution with a correct final answer. The student then runs 2 epochs of supervised fine-tuning on those solutions with TRL, with loss on the solution tokens only. The recipe measures the student's GSM8K test accuracy (500 questions, greedy) before and after. The whole run takes about 10 minutes.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints each stage's results. It writes `summary.json` to `<output_root>/<MLflow run ID>/` and logs the metrics to MLflow.

A validation run on an H100 produced these results:

| Stage | Result |
|---|---|
| Teacher batch generation | 3,886 of 4,000 questions solved correctly, in 181 s |
| Student fine-tuning | 3,886 examples × 2 epochs, in 105 s |
| Student GSM8K accuracy | 49.4% → 53.6% |

With 500 questions, the standard error is about 2.2 points. Raise `num_eval_questions` (up to 1,319) for a tighter estimate.

Notes:

- `train_student.py` disables PyTorch's cuDNN attention backend. An earlier validation run with it enabled hit NaN gradients partway through the first epoch and produced a broken student. The training also stops with an error if gradients become non-finite.
- Each stage runs in its own process. This fully releases GPU memory between vLLM and training.
