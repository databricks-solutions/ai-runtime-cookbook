"""Load the saved DINOv3 state dictionary in a fresh process and classify one test image."""

import json
import os
from pathlib import Path

import timm
import torch
import yaml
from torchvision.datasets import Flowers102

with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
    p = yaml.safe_load(handle)

output_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"]
metadata = json.loads((output_dir / "model_config.json").read_text())
transform = timm.data.create_transform(**metadata["data_config"], is_training=False)

model = timm.create_model(metadata["model_name"], pretrained=False, num_classes=102)
model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True))
model.eval()

# Test image 0 with the evaluation transform. Dataset labels are 0-based, so index i is label i + 1.
image, label = Flowers102(p["data_root"], split="test", download=False, transform=transform)[0]
with torch.inference_mode():
    logits = model(image.unsqueeze(0))
assert torch.isfinite(logits).all(), "The loaded model returned non-finite logits"
print(f"Predicted index {int(logits.argmax(dim=1))}, true index {label}")
