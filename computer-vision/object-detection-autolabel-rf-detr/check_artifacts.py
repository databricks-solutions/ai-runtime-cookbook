"""Load the saved RF-DETR checkpoint in a fresh process and predict on one test image.

workload.yaml runs this after train.py, so it shares no state with training. The image comes from the test split
that train.py staged in scratch space. The [artifact-load] line shows that the checkpoint loads and predicts.
"""

import json
import os
from pathlib import Path

import yaml
from PIL import Image
from rfdetr import RFDETR

with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
    p = yaml.safe_load(handle)

run_id = os.environ["MLFLOW_RUN_ID"]
output_dir = Path(p["output_root"]) / run_id
model = RFDETR.from_checkpoint(output_dir / "rfdetr_best.pth")

test_dir = Path(p["data_root"]) / run_id / "hf_dataset" / "test"
first_row = json.loads((test_dir / "metadata.jsonl").read_text().splitlines()[0])
image = Image.open(test_dir / first_row["file_name"]).convert("RGB")
detections = model.predict(image, threshold=0.1)
print(f"[artifact-load] rfdetr detections={len(detections.xyxy)} on {first_row['file_name']}", flush=True)
