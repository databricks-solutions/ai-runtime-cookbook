#!/usr/bin/env python3
"""Load the saved LoRA adapter in a fresh process and run one prediction.

The workload runs this after train.py. The process starts with no training state, so a pass
means the adapter copied to ADAPTER_OUTPUT_ROOT loads on the base model and predicts.
"""

import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    base_id = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    adapter_path = f"{os.environ['ADAPTER_OUTPUT_ROOT']}/{os.environ['MLFLOW_RUN_ID']}/final_adapter"
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model.eval()

    inputs = tokenizer("A man is sitting on a roof. He", return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    assert torch.isfinite(logits).all()
    next_id = int(logits[0, -1].argmax())
    print(f"Loaded {adapter_path}. Next token: {tokenizer.decode([next_id])!r}", flush=True)


if __name__ == "__main__":
    main()
