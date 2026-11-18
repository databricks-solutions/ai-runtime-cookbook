"""Report the training-loss trend and copy the final policy checkpoint to a volume."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import mlflow
import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

log, run_dir, log_freq = Path(sys.argv[1]).read_text(), Path(sys.argv[2]), int(sys.argv[3])
# LeRobot logs every log_freq steps and abbreviates the step count (step:1K), so number the
# points from the logging interval instead of parsing the abbreviated value.
losses = [float(l) for l in re.findall(r"step:\S+ .*?loss:([\d.]+)", log)]
points = [((i + 1) * log_freq, loss) for i, loss in enumerate(losses)]
if not points:
    sys.exit("no loss lines found in the training log")
first, last = points[0][1], points[-1][1]
summary = {"logged_points": len(points), "loss_first": first, "loss_last": last,
           "loss_reduction": 1 - last / first}
print("loss by step:", points)
print(json.dumps(summary, indent=2))

out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps({"summary": summary, "loss": points}, indent=2))
model_dirs = sorted(run_dir.glob("checkpoints/*/pretrained_model"))
if model_dirs:
    shutil.copytree(model_dirs[-1], out / "pretrained_model", dirs_exist_ok=True)
    print(f"Copied {model_dirs[-1]} to {out / 'pretrained_model'}")
with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
    mlflow.log_metrics(summary)
