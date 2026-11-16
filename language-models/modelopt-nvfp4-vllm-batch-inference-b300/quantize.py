"""Post-training quantization (PTQ) to NVFP4 with NVIDIA Model Optimizer.

Calibrates activation scales on a few hundred text samples, then exports a Hugging Face
checkpoint that vLLM loads natively.
"""

import os
import time
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
import yaml
from datasets import load_dataset
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

CONFIGS = {"nvfp4": mtq.NVFP4_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG}


def main():
    out = Path(P["quantized_dir"])
    tok = AutoTokenizer.from_pretrained(P["model_name"])
    model = AutoModelForCausalLM.from_pretrained(P["model_name"], dtype=torch.bfloat16, device_map="cuda")

    calib = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train")
    calib = calib.shuffle(seed=0).select(range(P["calib_samples"]))["article"]

    @torch.no_grad()
    def forward_loop(m):
        for text in calib:
            ids = tok(text, return_tensors="pt", truncation=True, max_length=P["calib_seq_len"])
            m(ids["input_ids"].cuda())

    t0 = time.time()
    model = mtq.quantize(model, CONFIGS[P["quant_format"]], forward_loop)
    print(f"[quantize] {P['quant_format']} calibration done in {time.time() - t0:.0f}s")
    mtq.print_quant_summary(model)

    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=out)
    tok.save_pretrained(out)
    size_gb = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    print(f"[quantize] exported to {out} ({size_gb:.1f} GB of weights)")


if __name__ == "__main__":
    main()
