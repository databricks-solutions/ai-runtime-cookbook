#!/usr/bin/env python3
"""Load the saved LoRA adapter in a new process and extract one test receipt with it.

Run after train.py, so nothing from that process is in memory. The adapter names the base model
in adapter_config.json, so the base model is loaded first and the adapter is applied on top.
"""

import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

from train import PROMPT, as_rgb, load_processor


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
        p = yaml.safe_load(f)
    adapter_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"] / "lora_adapter"
    processor = load_processor(p["model_name"], int(p["min_pixels"]), int(p["max_pixels"]))
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        p["model_name"], dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    row = load_dataset(p["dataset_name"], split="test")[0]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = processor(text=[prompt], images=[as_rgb(row["image"])], return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=int(p["max_new_tokens"]), do_sample=False)
    answer = processor.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)[0]
    if not answer.strip():
        raise RuntimeError(f"adapter at {adapter_dir} produced an empty answer")
    print(f"loaded adapter from {adapter_dir}; answer: {answer}", flush=True)


if __name__ == "__main__":
    main()
