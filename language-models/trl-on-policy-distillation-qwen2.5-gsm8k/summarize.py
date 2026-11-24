"""Collect every stage's results into summary.json and log the final metrics to the AIR MLflow run."""

import json
import math
import os
import statistics
from pathlib import Path

import mlflow

from common import NUM_EVAL_QUESTIONS, P, WORK

RUN_ID = os.environ["MLFLOW_RUN_ID"]
TAGS = ["student_before", "teacher", "opd", "offline", "control_opd", "control_offline"]

work = Path(WORK)
metrics = {}
for name in ["opd_stats", "offline_data_stats", "offline_train_stats"]:
    metrics.update(json.loads((work / f"{name}.json").read_text()))

evals = {tag: json.loads((work / f"eval_{tag}.json").read_text()) for tag in TAGS}
for tag, r in evals.items():
    if r["n"] != NUM_EVAL_QUESTIONS:
        raise RuntimeError(f"{tag} evaluated {r['n']} questions, expected {NUM_EVAL_QUESTIONS}")
    metrics[f"gsm8k_acc_{tag}"] = r["accuracy"]
    metrics[f"gsm8k_se_{tag}"] = r["standard_error"]
    metrics[f"gsm8k_share_answer_marker_{tag}"] = r["share_with_answer_marker"]
    metrics[f"gsm8k_share_finish_length_{tag}"] = r["share_finish_length"]
    metrics[f"gsm8k_mean_completion_tokens_{tag}"] = r["mean_completion_tokens"]


def paired_difference(a, b):
    """Mean and standard error of the per-question difference a - b. The questions are the same, so this is paired."""
    d = [x - y for x, y in zip(evals[a]["correct"], evals[b]["correct"])]
    return statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d))


for a, b in [
    ("opd", "student_before"),
    ("offline", "student_before"),
    ("opd", "offline"),
    ("control_opd", "student_before"),
    ("control_offline", "student_before"),
]:
    diff, se = paired_difference(a, b)
    metrics[f"gsm8k_diff_{a}_minus_{b}"] = diff
    metrics[f"gsm8k_diff_se_{a}_minus_{b}"] = se

summary = {
    "parameters": {k: P[k] for k in sorted(P)},
    "metrics": metrics,
    "last_tokens": {tag: evals[tag]["last_tokens_first_completion"] for tag in TAGS},
}
print(json.dumps(summary, indent=2), flush=True)

out = Path(P["output_root"]) / RUN_ID
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
# The HF trainer ended the AIR run after training, so reopen it by ID for the final metrics.
with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(str(out / "summary.json"))
