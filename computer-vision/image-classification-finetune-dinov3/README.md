# Image classification fine-tuning with DINOv3

Fine-tune [DINOv3 ViT-Small](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), a 21.6M-parameter pretrained vision transformer, on [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) for ten epochs on one A10 GPU. [DINOv3](https://arxiv.org/abs/2508.10104) is a recent state-of-the-art self-supervised vision method; this compact example adapts its distilled backbone and a new classification head with AdamW, separate backbone/head learning rates, warmup, cosine decay, and BF16 mixed precision.

The official splits contain 1,020 training, 1,020 validation, and 6,149 test images. Only training images receive random crops and flips; evaluation uses the pretrained model's normalization and deterministic resize/crop. The validation split selects the best checkpoint, and the test split is evaluated once after training. Ten epochs are a demonstration budget, not a claim of state-of-the-art Flowers-102 accuracy.

## Before you run

Review the weights' [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license) and the dataset's [usage information](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt). The public `timm` weights and dataset download automatically without a Hugging Face token. The recipe writes its outputs to `/Volumes/main/default/air_examples/image-classification-finetune-dinov3`, which must exist and be writable. To write to another volume, override `parameters.output_root`:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The AIR MLflow run contains per-epoch loss/accuracy, final test metrics, and the selected model packaged for inference. `<output_root>/<MLFLOW_RUN_ID>/` contains `best_model.pt` (a state dictionary), `model_config.json` (architecture, preprocessing, and label mapping), and `metrics.json`. Output indices 0–101 correspond to Oxford flower labels 1–102.

For inference, recreate the evaluation transform with `timm.data.create_transform(**metadata["data_config"], is_training=False)` from `model_config.json`. The MLflow model accepts batches of transformed float32 images in NCHW order and returns 102 logits; use `argmax` for the class or `softmax` for probabilities. To load the state dictionary directly, create the recorded `timm` model with `pretrained=False, num_classes=102`, load it with `torch.load(..., map_location="cpu", weights_only=True)`, and call `model.eval()`.

After training, `load_checkpoint.py` runs these two steps in a new process on one test image. To run them yourself, use the same code:

```python
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
```
