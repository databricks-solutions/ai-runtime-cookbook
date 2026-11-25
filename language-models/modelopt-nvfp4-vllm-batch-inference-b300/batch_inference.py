"""Offline batch inference with vLLM; scores GSM8K accuracy and measures throughput.

Usage: python batch_inference.py bf16|quantized
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import nvidia
import yaml
from datasets import load_dataset
from vllm import LLM, SamplingParams

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

SYSTEM = "Solve the math problem step by step. End with the final answer as: #### <number>"


def expose_curand_headers():
    """Put curand.h on CPATH for vLLM's FlashInfer JIT.

    FlashInfer JIT-compiles some kernels at vLLM startup. In this environment curand.h ships only
    inside the nvidia-* pip wheels, so copy just those headers to one include directory.
    """
    include = Path("/tmp/curand-include")
    include.mkdir(parents=True, exist_ok=True)
    for header in Path(nvidia.__path__[0]).rglob("curand*.h"):
        shutil.copy(header, include)
    os.environ["CPATH"] = f"{include}:{os.environ.get('CPATH', '')}"


def number(text):
    """Final answer after '####', else the last number in the text."""
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    found = match.group(1) if match else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if found is None:
        return None
    found = found.replace(",", "").rstrip(".")
    return found[:-2] if found.endswith(".0") else found


def main(label):
    expose_curand_headers()
    # bf16 runs the original checkpoint; quantized runs the export from quantize.py.
    model_path = P["model_name"] if label == "bf16" else P["quantized_dir"]
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(P["num_prompts"]))
    gold = [number(a) for a in ds["answer"]]
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
                for q in ds["question"]]

    llm = LLM(model=model_path, max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=P["max_tokens"])
    t0 = time.time()
    outs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    elapsed = time.time() - t0

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    correct = sum(number(o.outputs[0].text) == g for o, g in zip(outs, gold))
    result = {
        "label": label,
        "model_path": model_path,
        "prompts": len(outs),
        "seconds": elapsed,
        "output_tokens_per_second": gen_tokens / elapsed,
        "gsm8k_accuracy": correct / len(outs),
    }
    print(json.dumps(result), flush=True)
    Path(f"/tmp/vllm_result_{label}.json").write_text(json.dumps(result, indent=2))
    out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"generations_{label}.jsonl", "w") as f:
        for q, o in zip(ds["question"], outs):
            f.write(json.dumps({"question": q, "answer": o.outputs[0].text}) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
