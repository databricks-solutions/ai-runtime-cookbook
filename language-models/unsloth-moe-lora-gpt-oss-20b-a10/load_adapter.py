#!/usr/bin/env python3
"""Load the saved LoRA adapter in a new process and generate one sample answer.

The workload runs this after train.py, so nothing from that process is in memory. Unsloth reads
the base model name from adapter_config.json, so the adapter directory is enough to load.
"""

# Unsloth patches transformers and PEFT on import, so import it first.
import unsloth  # noqa: F401  # isort: skip
from unsloth import FastLanguageModel  # isort: skip

import os
from pathlib import Path

import yaml


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
        p = yaml.safe_load(f)
    if not p["save_adapter"]:
        # Runs only when save_adapter is true; otherwise train.py wrote no adapter.
        print("save_adapter is false; no adapter to load", flush=True)
        return
    adapter_dir = Path(p["output_root"]) / os.environ["MLFLOW_RUN_ID"] / "lora_adapter"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=p["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    messages = [{"role": "user", "content": p["sample_prompt"]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        reasoning_effort="low",
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if not answer.strip():
        raise RuntimeError(f"adapter at {adapter_dir} produced an empty answer")
    print(f"loaded adapter from {adapter_dir}; answer: {answer[:1500]}", flush=True)


if __name__ == "__main__":
    main()
