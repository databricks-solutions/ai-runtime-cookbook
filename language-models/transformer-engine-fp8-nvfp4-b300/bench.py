"""Train the same Llama-style model in bf16, FP8 and NVFP4 with Transformer Engine.

`python bench.py prepare` tokenizes WikiText-103 once. `python bench.py train <precision>`
trains on one GPU and writes tokens/s and loss. run_all.py starts one precision per GPU,
so the comparison runs in parallel on a single node.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
import yaml
from transformer_engine.common import recipe
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

WORK = Path("/tmp/te-bench")
TOKENS = WORK / "tokens.npy"


def quantization_recipe(name):
    if name == "bf16":
        return None
    if name == "fp8_current":
        return recipe.Float8CurrentScaling()
    if name == "mxfp8":
        return recipe.MXFP8BlockScaling()
    if name == "nvfp4":
        return recipe.NVFP4BlockScaling()
    raise ValueError(f"unknown precision {name}")


class Llama(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        h, heads = P["hidden_size"], P["num_heads"]
        self.embed = torch.nn.Embedding(vocab, h)
        self.rope = RotaryPositionEmbedding(h // heads)
        self.layers = torch.nn.ModuleList(
            te.TransformerLayer(
                hidden_size=h,
                ffn_hidden_size=P["ffn_hidden_size"],
                num_attention_heads=heads,
                num_gqa_groups=P["num_kv_heads"],
                normalization="RMSNorm",
                activation="swiglu",
                bias=False,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                self_attn_mask_type="causal",
                fuse_qkv_params=True,
                layer_number=i + 1,
                attn_input_format="bshd",
            )
            for i in range(P["num_layers"])
        )
        self.norm = te.RMSNorm(h)
        # Keep the LM head in bf16; low-precision output projections hurt the loss most.
        self.head = torch.nn.Linear(h, vocab, bias=False)

    def forward(self, ids, qrecipe):
        x = self.embed(ids)
        freqs = self.rope(ids.shape[1])
        with te.autocast(enabled=qrecipe is not None, recipe=qrecipe):
            for layer in self.layers:
                x = layer(x, rotary_pos_emb=freqs)
        return self.head(self.norm(x))


def prepare():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    WORK.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(P["tokenizer"])
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    need = (P["warmup_steps"] + P["steps"]) * P["micro_batch_size"] * (P["seq_len"] + 1)
    ids, i = [], 0
    while len(ids) < need:
        batch = [t for t in ds[i : i + 5000]["text"] if t.strip()]
        for enc in tok(batch)["input_ids"]:
            ids.extend(enc + [tok.eos_token_id])
        i += 5000
    np.save(TOKENS, np.array(ids[:need], dtype=np.int32))
    print(f"[prepare] {need:,} tokens -> {TOKENS}")


def train(precision):
    torch.manual_seed(0)
    data = torch.from_numpy(np.load(TOKENS)).long()
    vocab = P["vocab_size"]  # GPT-2 vocab padded to a multiple of 64 for efficient GEMMs
    b, s = P["micro_batch_size"], P["seq_len"]
    batches = data.view(-1, b, s + 1)

    model = Llama(vocab).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(P["learning_rate"]), betas=(0.9, 0.95),
                            weight_decay=0.1, fused=True)
    total = P["warmup_steps"] + P["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / total))
    )
    qrecipe = quantization_recipe(precision)

    losses, t_start = [], None
    for step in range(total):
        if step == P["warmup_steps"]:
            torch.cuda.synchronize()
            t_start = time.time()
        batch = batches[step].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch[:, :-1], qrecipe)
        loss = F.cross_entropy(logits.float().view(-1, vocab), batch[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.detach())
        if step % 25 == 0:
            print(f"[{precision}] step {step} loss {loss.item():.4f}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    # Converting per step would sync with the GPU inside the timed loop, so convert once here.
    losses = torch.stack(losses).tolist()

    tokens = P["steps"] * b * s
    result = {
        "precision": precision,
        "gpu": torch.cuda.get_device_name(0),
        "params_billion": n_params / 1e9,
        "tokens_per_second": tokens / elapsed,
        # 6 * params * tokens is the standard dense-transformer training FLOP estimate.
        "model_tflops_per_second": 6 * n_params * tokens / elapsed / 1e12,
        "seconds_per_step": elapsed / P["steps"],
        "loss_first": losses[0],
        "loss_last_20_mean": float(np.mean(losses[-20:])),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    print(json.dumps(result), flush=True)
    (WORK / f"result_{precision}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    else:
        train(sys.argv[2])
