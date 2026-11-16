#!/bin/bash
# Distillation pipeline on one GPU: each stage runs in its own process so GPU memory is freed.
set -eo pipefail
cd "$CODE_SOURCE_PATH"
STUDENT=$(python -c "from common import P; print(P['student_model'])")

echo "=== 1/4 Baseline student accuracy"
python generate.py eval "$STUDENT" student_before
echo "=== 2/4 Teacher generates solutions (batch inference with vLLM)"
python generate.py teacher
echo "=== 3/4 Student fine-tunes on the teacher's correct solutions"
python train_student.py
echo "=== 4/4 Distilled student accuracy"
python generate.py eval /tmp/distill/student student_after

python - <<'PY'
import json, os
from pathlib import Path
w = Path("/tmp/distill")
summary = {}
for name in ["teacher_stats", "student_stats"]:
    summary.update(json.loads((w / f"{name}.json").read_text()))
summary["gsm8k_acc_student_before"] = json.loads((w / "eval_student_before.json").read_text())["gsm8k_accuracy"]
summary["gsm8k_acc_student_after"] = json.loads((w / "eval_student_after.json").read_text())["gsm8k_accuracy"]
print(json.dumps(summary, indent=2))
from common import P
out = Path(P["output_root"]) / os.environ.get("MLFLOW_RUN_ID", "local")
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=2))
try:
    import mlflow
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics(summary)
except Exception as e:
    print(f"MLflow logging skipped: {e}")
PY
