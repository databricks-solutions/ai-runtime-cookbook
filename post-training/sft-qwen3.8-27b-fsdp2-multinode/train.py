#!/usr/bin/env python3
"""Qwen3.8-27B SFT fine-tuning example -- self-contained, plain torch (no training framework).

A torch + HF-transformers + FSDP2 training loop, one process per GPU via torchrun. The phases:

  - build:  load the pretrained Qwen3.8-27B checkpoint on every rank, then FSDP2-shard it
  - data:   stream UltraChat 200k (SFT), chat-template + tokenize with a response-only loss
            mask (loss on the assistant turns only), and pack into fixed seq-len blocks
  - train:  one full pass over the dataset

Packing is block-diagonal: examples are concatenated into SEQ_LEN blocks, but no token
attends across an example boundary. _pack_block derives the per-block signals (position_ids,
cu_seqlens, seq_idx) that keep attention, the gated-delta recurrent state, and the loss from
crossing boundaries -- which needs a flash-attention varlen backend (see workload.yaml).

See workload.yaml for the matching environment.
"""

import logging
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoModelForImageTextToText, AutoTokenizer

from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

# Use UTC instead of local time
logging.Formatter.converter = time.gmtime

logging.basicConfig(
    level=logging.INFO,
    format=f"[RANK: {LOCAL_RANK}] %(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)


MODEL = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
# SFT dataset streamed from the HF hub (UltraChat 200k, MIT-licensed multi-turn chat).
# Revision pinned so runs stay comparable as the repo moves.
DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"
SPLIT = "train_sft"
SEQ_LEN = 8192  # tokens per packed block


class PackedTokenDataset(IterableDataset):
    """Streams this rank's shard of the dataset and packs it into [1, SEQ_LEN] blocks.

    Each rank streams a disjoint shard (split_dataset_by_node), SFT-tokenizes examples
    (chat template + response-only loss mask), and jams them end-to-end into a buffer;
    whenever the buffer overfills SEQ_LEN it chops off one block and _pack_block builds
    the block-diagonal signals + masked labels. Nothing is materialized up front.

    Each item is (block, ran_out): ran_out is 0 for a real block, then 1 on a final all-masked
    block once the shard is exhausted.
    """

    def __init__(self, tokenizer, device: torch.device):
        self.tok = tokenizer
        self.device = device

    def _tokenize_sft(self, messages: list[dict]) -> tuple[list[int], list[bool]]:
        """Chat-template a (possibly multi-turn) conversation and return its token ids plus a
        per-token mask that is True only on the assistant turns' response tokens (SFT loss).

        Each assistant turn is located by prefix length: templating up to just before the turn
        (add_generation_prompt=True) vs through it brackets that turn's response tokens. This
        works with any chat template, without relying on {% generation %} markers.
        """
        full_ids = self.tok.apply_chat_template(messages, tokenize=True, return_dict=False)
        loss_mask = [False] * len(full_ids)
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue
            prefix = self.tok.apply_chat_template(
                messages[:i], add_generation_prompt=True, tokenize=True, return_dict=False
            )
            through = self.tok.apply_chat_template(messages[: i + 1], tokenize=True, return_dict=False)
            for j in range(len(prefix), min(len(through), len(full_ids))):
                loss_mask[j] = True  # this assistant turn's response tokens contribute to loss
        return full_ids, loss_mask

    def _pack_block(self, ids: list[int], interior_starts: list[int], loss_mask: list[bool]):
        """Turn one SEQ_LEN block of token ids + the offsets where interior examples begin +
        the per-token SFT loss mask into (input_ids, labels, position_ids, cu_seqlens, seq_idx, max_len).

        Block-diagonal, so nothing mixes across an example boundary:
          * position_ids -- reset to 0 per segment
          * cu_seqlens   -- segment boundaries, for the full-attention layers
          * seq_idx      -- per-token segment id, for the gated-delta layers
          * labels       -- -100 on prompt tokens (SFT loss is response-only) and on each
                            segment's first token (so the causal-shifted loss can't cross a boundary)
        """
        bounds = [0, *sorted({s for s in interior_starts if 0 < s < SEQ_LEN}), SEQ_LEN]
        ids_t = torch.tensor(ids, dtype=torch.long)
        position_ids = torch.empty(SEQ_LEN, dtype=torch.long)
        seq_idx = torch.empty(SEQ_LEN, dtype=torch.int32)
        labels = ids_t.clone()
        labels[~torch.tensor(loss_mask, dtype=torch.bool)] = -100  # SFT: mask prompt tokens
        for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            position_ids[a:b] = torch.arange(b - a)
            seq_idx[a:b] = i
            if a > 0:
                labels[a] = -100  # causal-shifted target would come from the previous segment
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32)
        max_len = max(b - a for a, b in zip(bounds[:-1], bounds[1:]))
        return (
            ids_t.unsqueeze(0).to(self.device),
            labels.unsqueeze(0).to(self.device),
            position_ids.unsqueeze(0).to(self.device),
            cu_seqlens.to(self.device),
            seq_idx.unsqueeze(0).to(self.device),
            max_len,
        )

    def _shard_examples(self):
        # Shard across every (rank, DataLoader-worker) stream, not just per rank, so
        # num_workers>1 gives each worker a disjoint slice instead of replaying the whole
        # rank shard. With the default num_workers=1 this reduces to per-rank sharding.
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_rank = RANK * num_workers + worker_id
        shard_world = WORLD_SIZE * num_workers
        ds = load_dataset(DATASET, split=SPLIT, streaming=True, revision=DATASET_REVISION)
        ds = split_dataset_by_node(ds, rank=shard_rank, world_size=shard_world)
        yield from ds

    def __iter__(self):
        buf: list[int] = []  # token ids, examples jammed end-to-end
        loss: list[bool] = []  # parallel SFT mask: True where the token contributes to loss
        starts: list[int] = []  # offsets in buf where each example begins
        for ex in self._shard_examples():
            starts.append(len(buf))
            ids, ex_loss = self._tokenize_sft(ex["messages"])
            buf.extend(ids)
            loss.extend(ex_loss)
            while len(buf) >= SEQ_LEN:
                yield self._pack_block(buf[:SEQ_LEN], starts, loss[:SEQ_LEN]), torch.tensor(0.0)
                buf = buf[SEQ_LEN:]
                loss = loss[SEQ_LEN:]
                # keep only example starts landing in the carried-over remainder
                starts = [s - SEQ_LEN for s in starts if s - SEQ_LEN > 0]
        yield self._pack_block([0] * SEQ_LEN, [], [False] * SEQ_LEN), torch.tensor(1.0)


def build_sharded_model() -> torch.nn.Module:
    """Load the pretrained Qwen3.8-27B on every rank, then FSDP2-shard each decoder layer + root.

    Every rank loads the full checkpoint, fully_shard reshards it in place, and .cuda() moves
    each rank's shard to GPU. Simple, at the cost of every rank holding the whole model in host
    RAM transiently -- fine at 27B. (For a much larger model, load on rank 0 and broadcast each
    rank its shard instead.)
    """
    # Explicit flash_attention_3: transformers otherwise defaults to sdpa, which is ~15% slower
    # per step here.
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, revision=MODEL_REVISION,
        attn_implementation="flash_attention_3", dtype=torch.bfloat16,
    )
    # The KV cache only helps generation; in training it wastes memory and is incompatible
    # with gradient checkpointing. (text_config is this VLM-style model's language tower.)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    layers = model.model.language_model.layers
    for layer in layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    model.cuda()

    if RANK == 0:
        log.info(f"model built + weights loaded: {model.config.model_type}, {len(layers)} layers, sharded over {WORLD_SIZE}")
    return model


def run_training_loop(model: torch.nn.Module, data) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    model.train()

    loss = None
    loop_start = time.perf_counter()
    for step, (batch, ran_out) in enumerate(data):
        exhausted = bool(ran_out)  # this rank's own flag; its final all-masked block has a NaN loss
        # Ranks get different numbers of batches, so they must stop on the same step: a rank that
        # stopped early would hang the others in the next collective.
        ran_out = ran_out.cuda(non_blocking=True)
        dist.all_reduce(ran_out, op=dist.ReduceOp.MAX)

        ids, labels, position_ids, cu_seqlens, seq_idx, max_len = batch
        # The loader builds batches on CPU (a worker can't return CUDA tensors); move to GPU here.
        ids = ids.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        position_ids = position_ids.cuda(non_blocking=True)
        cu_seqlens = cu_seqlens.cuda(non_blocking=True)
        seq_idx = seq_idx.cuda(non_blocking=True)
        out = model(
            input_ids=ids,
            labels=labels,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_len,
            max_length_k=max_len,
            seq_idx=seq_idx,
        )
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if not exhausted:
            loss = out.loss
        if (step + 1) % 100 == 0 and RANK == 0:
            log.info(f"[train] step {step + 1}, loss={loss.item():.3f}")

        # .item() is a sync point, so check it at the end rather than before the step.
        if ran_out.item():
            break
    torch.cuda.synchronize()  # let all queued GPU work finish before we stop the clock
    elapsed = time.perf_counter() - loop_start
    if RANK == 0:
        log.info(f"[train] {step + 1} steps in {elapsed:.2f}s ({elapsed / (step + 1):.2f}s/step), final loss={loss.item():.3f}")


def main() -> None:
    torch.cuda.set_device(LOCAL_RANK)
    torch.manual_seed(0)
    dist.init_process_group("nccl")
    try:
        model = build_sharded_model()
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
        if RANK == 0:
            log.info(f"streaming dataset={DATASET}, seq_len={SEQ_LEN}")

        data = DataLoader(
            PackedTokenDataset(tokenizer, torch.device("cpu")),
            batch_size=None, num_workers=1, prefetch_factor=4, pin_memory=True,
        )

        run_training_loop(model, data)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
