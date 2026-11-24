"""Train an EAGLE-3 draft head for Qwen3-8B and export it for SGLang.

The frozen target runs in this process, so no separate inference server is needed. The draft
learns to match the target's next-token distribution (ModelOpt's EAGLE distillation loss) on
UltraChat conversations. The exported checkpoint uses SGLang's LlamaForCausalLMEagle3 layout.
"""

import json
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM

import modelopt.torch.speculative as mtsp
from modelopt.torch.export import export_speculative_decoding

from common import encode_example, load_split, tokenizer

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_ID = os.environ["MLFLOW_RUN_ID"]
OUT = Path(P["output_root"]) / RUN_ID
DRAFT_DIR = OUT / "draft"
UNTRAINED_DIR = OUT / "draft_untrained"
LR = float(P["learning_rate"])
EPOCHS = int(P["epochs"])
MICRO_BS = int(P["micro_batch_size"])
GRAD_ACCUM = int(P["grad_accum_steps"])
MAX_LEN = int(P["max_length"])
SEED = int(P["seed"])

def target_rope(cfg):
    """RoPE base and scaling of the target. Transformers 5 keeps them in rope_parameters."""
    params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    theta = params.get("rope_theta", getattr(cfg, "rope_theta", None))
    assert theta is not None, "target config has no rope_theta"
    assert params.get("rope_type", params.get("type", "default")) == "default", params
    return float(theta)


def eagle3_config(target_cfg):
    """ModelOpt's EAGLE-3 default trains the draft with rope_theta 10000 and exports a YaRN
    rope_scaling (factor 32) that the trained draft never saw. Qwen3-8B uses rope_theta
    1000000 with no scaling, so the draft is trained and exported with the target's RoPE."""
    theta = target_rope(target_cfg)
    cfg = deepcopy(mtsp.config.EAGLE3_DEFAULT_CFG)
    arch = cfg["config"]["eagle_architecture_config"]
    arch["rope_theta"] = theta
    arch["rope_scaling"] = {"rope_type": "default", "rope_theta": theta}
    cfg["config"]["eagle_export_rope_scaling"] = {}
    return cfg


# ModelOpt 0.48's EAGLE training forward caches attention state from the first call, so any
# later call with a different sequence length or batch size fails. Every batch is therefore
# padded to one fixed shape. Padding sits at the end, so causal attention is unaffected, and
# padded positions carry zero loss weight.
PAD_ID = 0


def collate(rows, tok_pad=PAD_ID):
    ids = torch.full((MICRO_BS, MAX_LEN), tok_pad, dtype=torch.long)
    am = torch.zeros((MICRO_BS, MAX_LEN), dtype=torch.long)
    lm = torch.zeros((MICRO_BS, MAX_LEN), dtype=torch.float32)
    for r, (row_ids, row_mask) in enumerate(rows):
        n = len(row_ids)
        ids[r, :n] = torch.tensor(row_ids)
        am[r, :n] = 1
        lm[r, :n] = torch.tensor(row_mask, dtype=torch.float32)
    return ids.cuda(), am.cuda(), lm.cuda()


def batch_loss(model, rows):
    ids, am, lm = collate(rows)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=am, labels=ids, loss_mask=lm)
    return out.loss


@torch.no_grad()
def mean_loss(model, examples):
    """Mean of per-batch losses over full batches, with a standard error across batches."""
    model.eval()
    losses = []
    for start in range(0, len(examples), MICRO_BS):
        loss = batch_loss(model, examples[start:start + MICRO_BS]).item()
        assert math.isfinite(loss), "non-finite held-out loss"
        losses.append(loss)
    model.train()
    mean = sum(losses) / len(losses)
    var = sum((x - mean) ** 2 for x in losses) / (len(losses) - 1)
    return mean, (var / len(losses)) ** 0.5


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = tokenizer(P["model_name"])

    train_pairs = load_split("train_sft", int(P["num_train_prompts"]), SEED)
    train = [e for e in (encode_example(tok, u, a, MAX_LEN) for u, a in train_pairs) if e]
    eval_pairs = load_split("test_sft", int(P["num_eval_prompts"]), SEED)
    heldout = [e for e in (encode_example(tok, u, a, MAX_LEN) for u, a in eval_pairs[:100]) if e]
    heldout = heldout[: len(heldout) // MICRO_BS * MICRO_BS]
    supervised = sum(sum(m) for _, m in train)
    train = train[: len(train) // MICRO_BS * MICRO_BS]
    print(f"train examples: {len(train)} of {len(train_pairs)} kept; "
          f"supervised tokens: {supervised}; held-out examples: {len(heldout)}", flush=True)
    assert len(train) >= MICRO_BS * GRAD_ACCUM, "too few training examples"

    model = AutoModelForCausalLM.from_pretrained(P["model_name"], dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    mtsp.convert(model, eagle3_config(model.config))
    # Target stays bf16 and frozen; the draft keeps fp32 master weights for AdamW.
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable draft parameters: {sum(p.numel() for p in params):,}", flush=True)
    model.train()

    base_loss, base_se = mean_loss(model, heldout)
    print(f"held-out loss before training: {base_loss:.4f} +- {base_se:.4f}", flush=True)
    # The untrained head is exported as a control row for the benchmark.
    export_speculative_decoding(model, dtype=torch.bfloat16, export_dir=UNTRAINED_DIR)
    model.train()

    total_steps = EPOCHS * (len(train) // MICRO_BS // GRAD_ACCUM)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    warmup = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)))

    mlflow.start_run(run_id=RUN_ID)
    mlflow.log_metrics({"heldout_loss_before": base_loss, "heldout_loss_before_se": base_se}, step=0)
    step, window, t0 = 0, [], time.time()
    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(train)
        for i in range(0, len(train), MICRO_BS):
            loss = batch_loss(model, train[i:i + MICRO_BS])
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite training loss at example {i} of epoch {epoch}")
            (loss / GRAD_ACCUM).backward()
            window.append(loss.item())
            if (i // MICRO_BS + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 25 == 0 or step == total_steps:
                    avg = sum(window) / len(window)
                    print(f"epoch {epoch} step {step}/{total_steps} loss {avg:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} elapsed {time.time() - t0:.0f}s", flush=True)
                    mlflow.log_metric("train_loss", avg, step=step)
                    window = []

    after_loss, after_se = mean_loss(model, heldout)
    print(f"held-out loss after training: {after_loss:.4f} +- {after_se:.4f}", flush=True)
    mlflow.log_metrics({"heldout_loss_after": after_loss, "heldout_loss_after_se": after_se,
                        "train_steps": step, "train_examples": len(train)}, step=step)

    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    export_speculative_decoding(model, dtype=torch.bfloat16, export_dir=DRAFT_DIR)
    cfg = json.loads((DRAFT_DIR / "config.json").read_text())
    print("exported architectures:", cfg["architectures"], flush=True)
    assert cfg["architectures"] == ["LlamaForCausalLMEagle3"], cfg["architectures"]
    mlflow.end_run()


if __name__ == "__main__":
    main()
