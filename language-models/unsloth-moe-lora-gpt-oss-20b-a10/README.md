# Fine-tune gpt-oss-20b (mixture of experts) on one A10 with Unsloth

This recipe runs QLoRA fine-tuning of `unsloth/gpt-oss-20b` on a single 24 GB A10 using [Unsloth](https://github.com/unslothai/unsloth). gpt-oss-20b is a 21B-parameter mixture-of-experts (MoE) model. Unsloth loads it in 4-bit and trains LoRA adapters (0.44% of parameters) on the `HuggingFaceH4/Multilingual-Thinking` reasoning dataset for 30 steps, then generates a sample answer.

The run takes about 9 minutes in the container.

## Before you run

The default `output_root` is the shared Unity Catalog volume `/Volumes/main/default/air_examples/unsloth-moe-lora-gpt-oss-20b-a10`. The volume must already exist and be writable. To use another volume, run:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints the training loss, peak GPU memory, time per step and a sample answer. It writes `metrics.json` and `sample.txt` to `<output_root>/<MLflow run ID>/`. Set `save_adapter: true` to also save the LoRA adapter there. The validation run used `save_adapter: true`.

With `save_adapter: true`, the workload then runs `load_adapter.py` in a new process. It loads the saved adapter and generates once, so a pass means the adapter loads on its own. To load it yourself, use the same script:

```python
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
```

A validation run on an A10 produced these results:

| Metric | Value |
|---|---|
| Peak GPU memory after loading | 12.6 GB |
| Peak GPU memory during training | 15.8 GB of 23.7 GB |
| Time per step (batch 1 × 4 accumulation, 1,024 tokens) | 9.7 s |
| Training loss, first → last logged | 5.75 → 4.80 |
