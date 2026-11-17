"""Collect each stage's results into summary.json and log them to the MLflow run."""

import json
import os
from pathlib import Path

import mlflow

from common import P, WORK

work = Path(WORK)
summary = {}
for name in ["teacher_stats", "student_stats"]:
    summary.update(json.loads((work / f"{name}.json").read_text()))
for tag in ["student_before", "student_after"]:
    summary[f"gsm8k_acc_{tag}"] = json.loads((work / f"eval_{tag}.json").read_text())["gsm8k_accuracy"]
print(json.dumps(summary, indent=2))

out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metrics(summary)
    mlflow.log_artifact(str(out / "summary.json"))
