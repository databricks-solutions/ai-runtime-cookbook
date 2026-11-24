"""Stage 1: post-training quantization (PTQ) of the bf16 model to NVFP4.

Max calibration on GSM8K train questions, saved as a ModelOpt checkpoint. Stage 3 (QAD) starts
from this checkpoint, and export.py turns it into a vLLM-loadable NVFP4 model.
"""

import time

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
from common import P, collate_fn, paths, train_examples
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

mto.enable_huggingface_checkpointing()


def main():
    dirs = paths()
    tok = AutoTokenizer.from_pretrained(P["model_name"])
    model = AutoModelForCausalLM.from_pretrained(P["model_name"], dtype=torch.bfloat16,
                                                 device_map="cuda")

    # Calibration uses the GSM8K train split only, so the test split is never seen.
    calib = train_examples(tok, limit=0).select(range(P["calib_samples"]))
    loader = DataLoader(calib, batch_size=8, collate_fn=collate_fn(tok.pad_token_id or tok.eos_token_id))

    @torch.no_grad()
    def forward_loop(m):
        for batch in loader:
            m(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda())

    t0 = time.time()
    # NVFP4 weights and activations, with the KV cache left in bf16. The batch-inference recipe
    # validated this config through vLLM on B300, so this recipe uses it too.
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop)
    print(f"[ptq] calibrated {len(calib)} GSM8K train examples in {time.time() - t0:.0f}s", flush=True)
    mtq.print_quant_summary(model)

    dirs["ptq_ckpt"].mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dirs["ptq_ckpt"])
    tok.save_pretrained(dirs["ptq_ckpt"])
    print(f"[ptq] saved ModelOpt checkpoint to {dirs['ptq_ckpt']}", flush=True)


if __name__ == "__main__":
    main()
