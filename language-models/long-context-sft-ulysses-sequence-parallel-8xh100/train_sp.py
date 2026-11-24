#!/usr/bin/env python3
"""One point of the sequence-parallel sweep: full SFT of Qwen3-8B on long chat data.

Launched by sweep.py with `accelerate launch` on 8 GPUs. With --sp-size 8 the 8 GPUs share
one sequence (DeepSpeed Ulysses splits it along the sequence dimension). With --sp-size 1
each GPU trains on its own sequence (data parallel). Both use DeepSpeed ZeRO-3 for model
states, so the sequence split is the only difference.

Rank 0 writes one JSON result to --result-json. A CUDA out-of-memory error exits non-zero,
and sweep.py records that point as "oom".
"""

import argparse
import json
import time

import torch
import torch.distributed as dist
import torch.distributed.nn  # differentiable all_gather used by sp_loss
from accelerate import Accelerator
from accelerate.utils import ParallelismConfig
from accelerate.utils.dataclasses import DeepSpeedSequenceParallelConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100
# Ulysses uses this kernel for the attention step, and it needs no padding mask,
# so every sequence stays a single unpadded row.
ATTN_IMPLEMENTATION = "flash_attention_2"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--sp-size", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--warmup-steps", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, required=True)
    ap.add_argument("--result-json", required=True)
    return ap.parse_args()


def build_chunks(tokenizer, data_path, seq_len, n_chunks):
    """Pack chat-formatted LongAlign conversations into n_chunks sequences of seq_len tokens.

    Most conversations are 8K-64K tokens, so a 128K sequence is built by concatenating
    several of them. Attention is not masked across these boundaries. The loss covers
    assistant tokens only, so a window with no assistant tokens is skipped: its mean loss
    would be NaN. The data is for a throughput and memory sweep, not quality.
    """
    chunks = []
    ids_buf, labels_buf = [], []
    with open(data_path) as f:
        for line in f:
            if len(chunks) == n_chunks:
                break
            messages = json.loads(line)["messages"]
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            # Masking by prompt length is only correct if the prompt is a token prefix of the
            # full conversation. Fail loudly rather than train on the wrong tokens.
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("chat template: generation prompt is not a prefix of the full conversation")
            ids_buf += full_ids
            labels_buf += [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
            while len(ids_buf) >= seq_len and len(chunks) < n_chunks:
                window_labels = labels_buf[:seq_len]
                if all(x == IGNORE_INDEX for x in window_labels):
                    ids_buf = ids_buf[seq_len:]
                    labels_buf = labels_buf[seq_len:]
                    continue
                ids = torch.tensor(ids_buf[:seq_len], dtype=torch.long)
                chunks.append(
                    {
                        "input_ids": ids,
                        "labels": torch.tensor(window_labels, dtype=torch.long),
                        "position_ids": torch.arange(seq_len, dtype=torch.long),
                    }
                )
                ids_buf = ids_buf[seq_len:]
                labels_buf = labels_buf[seq_len:]
    if len(chunks) < n_chunks:
        raise ValueError(f"dataset too small: built {len(chunks)} of {n_chunks} sequences of {seq_len} tokens")
    return chunks


def sp_loss(outputs, shift_labels, sp_group, sp_world_size):
    """Token-weighted mean loss over the SP group, as in DeepSpeed's ALST tutorial.

    A shard with no assistant tokens gives NaN from HF's mean loss, and NaN times a zero
    weight is still NaN. Such a shard contributes a finite zero term instead.
    """
    good = (shift_labels != IGNORE_INDEX).view(-1).sum()
    if good.item() > 0:
        local_loss = outputs.loss.float()
    else:
        local_loss = outputs.logits.sum(dtype=torch.float32) * 0.0
    losses = torch.distributed.nn.functional.all_gather(local_loss, group=sp_group)
    goods = torch.distributed.nn.functional.all_gather(good, group=sp_group)
    total = sum(losses[r] * goods[r] for r in range(sp_world_size))
    total_good = sum(goods)
    return total / total_good.clamp(min=1)


def main():
    args = parse_args()
    accelerator = Accelerator(
        parallelism_config=(
            ParallelismConfig(
                sp_backend="deepspeed",
                sp_size=args.sp_size,
                sp_handler=DeepSpeedSequenceParallelConfig(
                    sp_seq_length_is_variable=True,
                    sp_attn_implementation=ATTN_IMPLEMENTATION,
                ),
            )
            if args.sp_size > 1
            else None
        ),
    )
    world = accelerator.num_processes
    if world % args.sp_size != 0 or args.seq_len % args.sp_size != 0:
        raise ValueError(f"world {world} and seq_len {args.seq_len} must both be divisible by sp_size {args.sp_size}")
    dp_size = world // args.sp_size
    total_iters = args.warmup_steps + args.steps
    # The Ulysses loader shares each sample across the SP group, so every rank needs at least one
    # sample, and every data-parallel slot needs one per iteration.
    n_chunks = max(world, dp_size * total_iters)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    chunks = build_chunks(tokenizer, args.data_path, args.seq_len, n_chunks)

    # Qwen3 is trained to 32K tokens. Its model card documents YaRN (factor 4) for 131K, so the
    # RoPE scaling is set here. It is static, so it applies to every point, including the 16K ones.
    config = AutoConfig.from_pretrained(args.model_name)
    config.rope_parameters = {
        **config.rope_parameters,
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, config=config, dtype=torch.bfloat16, attn_implementation=ATTN_IMPLEMENTATION
    )
    model.config.use_cache = False
    # Gradient checkpointing only takes effect in training mode, and from_pretrained returns eval mode.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataloader = torch.utils.data.DataLoader(
        chunks, batch_size=1, shuffle=False, collate_fn=lambda items: {k: torch.stack([x[k] for x in items]) for k in items[0]}
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # With SP, prepare() wraps the loader in DeepSpeed's Ulysses adapter. Each batch is then
    # sharded along the sequence across the SP group.
    sp_group = dataloader.sp_group if args.sp_size > 1 else None
    sp_world_size = dataloader.sp_world_size if args.sp_size > 1 else 1

    is_main = accelerator.is_main_process
    step_times, step_losses = [], []
    loader_iter = iter(dataloader)
    for step in range(total_iters):
        # The Ulysses loader yields its shards on CPU, so move them to the GPU first.
        batch = {k: v.to(accelerator.device) for k, v in next(loader_iter).items()}
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if args.sp_size > 1:
            # transformers computes a loss only when `labels` is set, and then uses `shift_labels`
            # (already shifted by the loader) instead of shifting again.
            outputs = model(**batch, labels=batch["shift_labels"])
            loss = sp_loss(outputs, batch["shift_labels"], sp_group, sp_world_size)
        else:
            outputs = model(**batch)
            loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss {loss.item()} at step {step}")
        accelerator.backward(loss)
        optimizer.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= args.warmup_steps:
            step_times.append(dt)
            step_losses.append(loss.item())
        if is_main:
            print(f"step {step} loss {loss.item():.4f} sec {dt:.2f} (warmup={step < args.warmup_steps})", flush=True)

    peak_alloc = torch.tensor([torch.cuda.max_memory_allocated() / 2**30], device=accelerator.device)
    peak_resv = torch.tensor([torch.cuda.max_memory_reserved() / 2**30], device=accelerator.device)
    dist.all_reduce(peak_alloc, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_resv, op=dist.ReduceOp.MAX)

    if is_main:
        mean_sec = sum(step_times) / len(step_times)
        tokens_per_iter = args.seq_len * dp_size
        result = {
            "status": "ok",
            "model_name": args.model_name,
            "sp_size": args.sp_size,
            "dp_size": dp_size,
            "world_size": world,
            "seq_len": args.seq_len,
            "timed_steps": len(step_times),
            "tokens_per_iter": tokens_per_iter,
            "sec_per_iter": mean_sec,
            "tokens_per_s": tokens_per_iter / mean_sec,
            "peak_allocated_gb_per_gpu": peak_alloc.item(),
            "peak_reserved_gb_per_gpu": peak_resv.item(),
            "rank0_losses": step_losses,
        }
        with open(args.result_json, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
