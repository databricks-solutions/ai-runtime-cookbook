"""Load the fine-tuned Parakeet checkpoint in a fresh process and transcribe one test clip."""

import json
import os

import torch
import yaml
from nemo.collections.asr.models import ASRModel

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

checkpoint = f"{P['output_root']}/{os.environ['MLFLOW_RUN_ID']}/finetuned.nemo"
model = ASRModel.restore_from(checkpoint, map_location=torch.device("cuda")).eval()

# The first test utterance from the training run's scratch manifest, with its normalized reference.
with open("/tmp/ami_parakeet_data/test_manifest.json") as f:
    record = json.loads(f.readline())
with torch.inference_mode():
    hypothesis = model.transcribe([record["audio_filepath"]], batch_size=1, verbose=False)[0]
print("Reference:", record["text"])
print("Fine-tuned transcription:", hypothesis.text)
