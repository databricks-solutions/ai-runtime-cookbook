"""Stage 2 (PTQ) and stage 4 (QAD): export a ModelOpt checkpoint to a vLLM-loadable NVFP4 model.

Usage: python export.py ptq|qad
"""

import sys

import modelopt.torch.opt as mto
import torch
from common import paths
from modelopt.torch.export import export_hf_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

mto.enable_huggingface_checkpointing()


def main(label):
    dirs = paths()
    src = dirs[f"{label}_ckpt"]
    dst = dirs[f"{label}_export"]
    # from_pretrained restores the quantizers saved with the checkpoint (ModelOpt HF checkpointing).
    model = AutoModelForCausalLM.from_pretrained(str(src), dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    dst.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        export_hf_checkpoint(model, export_dir=dst)
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(dst)
    size_gb = sum(f.stat().st_size for f in dst.glob("*.safetensors")) / 1e9
    print(f"[export] {label}: wrote {dst} ({size_gb:.2f} GB of safetensors)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
