"""Shared pieces for the NVFP4 accuracy-recovery stages: parameters, paths, GSM8K data, scoring."""

import os
import re
from pathlib import Path

import datasets
import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)
# AIR passes parameters back as strings in some cases (for example 1e-05), so cast explicitly.
for _key in ("max_length", "train_examples", "calib_samples", "eval_limit", "max_tokens", "epochs",
             "per_device_batch_size"):
    P[_key] = int(P[_key])
P["learning_rate"] = float(P["learning_rate"])

# Same system prompt and answer format as the batch-inference recipe, so the numbers are comparable.
SYSTEM = "Solve the math problem step by step. End with the final answer as: #### <number>"

# Intermediate quantized checkpoints stay in scratch space; they are rebuilt on every run.
SCRATCH = Path("/tmp/qad")


def paths():
    """Locations for one AIR attempt. Call this before a process removes MLFLOW_RUN_ID."""
    run_dir = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    return {
        "run": run_dir,
        "ptq_ckpt": SCRATCH / "ptq_ckpt",
        "ptq_export": run_dir / "ptq_nvfp4",
        "qad_ckpt": run_dir / "qad_ckpt",
        "qad_export": run_dir / "qad_nvfp4",
    }


def model_path(label):
    """Model directory that evaluate.py loads for each label."""
    if label == "bf16":
        return P["model_name"]
    return str(paths()[{"ptq": "ptq_export", "qad": "qad_export"}[label]])


def train_examples(tok, limit):
    """GSM8K train split only (never the test split) as chat-formatted, loss-on-answer examples.

    The prompt tokens get label -100, so the loss covers the assistant answer (including the
    empty think block Qwen3 emits when thinking is off) and the closing <|im_end|> token.
    """
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    if limit:
        ds = ds.select(range(limit))
    rows = []
    for ex in ds:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": ex["answer"]}],
                                       tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("chat template: the answer rendering does not extend the prompt")
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        if len(f_ids) > P["max_length"]:
            continue
        rows.append({"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]})
    print(f"[data] GSM8K train examples: {len(rows)} (dropped {len(ds) - len(rows)} over "
          f"{P['max_length']} tokens)", flush=True)
    return datasets.Dataset.from_list(rows)


def collate_fn(pad_id):
    import torch

    def collate(batch):
        width = max(len(x["input_ids"]) for x in batch)
        pad = [width - len(x["input_ids"]) for x in batch]
        return {
            "input_ids": torch.tensor([x["input_ids"] + [pad_id] * p for x, p in zip(batch, pad)]),
            "attention_mask": torch.tensor([[1] * len(x["input_ids"]) + [0] * p
                                            for x, p in zip(batch, pad)]),
            "labels": torch.tensor([x["labels"] + [-100] * p for x, p in zip(batch, pad)]),
        }
    return collate


def load_gsm8k_test(limit):
    ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
    return ds.select(range(limit)) if limit else ds


def gold_number(answer):
    return float(answer.split("####")[-1].strip().replace(",", ""))


def parse_number(text):
    """The number after the last '####', else the last number in the text."""
    found = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    if not found:
        found = re.findall(r"-?[\d,]*\.?\d+", text)
    if not found:
        return None
    return float(found[-1].replace(",", "").rstrip("."))
