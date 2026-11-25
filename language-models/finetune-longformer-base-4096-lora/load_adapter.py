#!/usr/bin/env python3
"""Load the saved LoRA adapter in a fresh process and run one prediction.

The workload runs this after train.py. The process starts with no training state, so a pass
means the saved adapter, classifier head and tokenizer load on their own.
"""

import os

# Same cache settings as train.py; the default /root/.cache is read-only on AIR workers.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")

import torch
import yaml
from peft import PeftModel
from transformers import AutoTokenizer, LongformerForSequenceClassification


def main() -> None:
    with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
        p = yaml.safe_load(f)
    num_labels = int(p["num_labels"])
    adapter_path = f"{p['output_root']}/{os.environ['MLFLOW_RUN_ID']}"

    base = LongformerForSequenceClassification.from_pretrained(p["model_name"], num_labels=num_labels)
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model.eval()

    review = "A slow start, but the final act is excellent. Worth the time."
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert logits.shape == (1, num_labels) and torch.isfinite(logits).all(), logits
    print(f"Loaded {adapter_path}. Logits for one review: {logits.tolist()}", flush=True)


if __name__ == "__main__":
    main()
