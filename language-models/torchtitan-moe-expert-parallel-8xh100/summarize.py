"""Pull loss, tokens/s and memory from torchtitan's step log lines, save them, and upload the log."""

import json
import os
import re
import sys
from pathlib import Path

lines = [l for l in Path(sys.argv[1]).read_text().splitlines() if re.search(r"step:\s*\d+", l)]
rows = []
for line in lines:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
    clean = clean[clean.index("step:"):]  # skip the timestamp prefix
    row = {k: float(v.replace(",", "")) for k, v in re.findall(r"(\w+):\s*([\d.,]+)", clean)}
    if "step" in row:
        rows.append(row)
for r in rows[:2] + rows[-3:]:
    print(r)
if not rows:
    sys.exit("no step metrics found in the training log")
last = rows[-1]
summary = {"steps": last.get("step"), "final_loss": last.get("loss"),
           "tokens_per_second_per_gpu": last.get("tps"), "tflops_per_gpu": last.get("tflops"),
           "mfu_percent": last.get("mfu")}
print(json.dumps(summary, indent=2))
out = Path(os.environ["OUTPUT_ROOT"]) / os.environ.get("MLFLOW_RUN_ID", "local")
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps({"summary": summary, "steps": rows}, indent=2))
try:
    import mlflow

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics({k: v for k, v in summary.items() if v is not None})
        mlflow.log_artifact(sys.argv[1], artifact_path="logs")
        mlflow.log_artifact(str(out / "summary.json"))
except Exception as e:  # MLflow logging is optional for the recipe.
    print(f"MLflow logging skipped: {e}")
