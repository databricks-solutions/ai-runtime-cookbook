"""Shared data preparation for the draft-training and benchmark steps.

Both steps use single-turn conversations from UltraChat, rendered with the target's chat
template with thinking disabled. Train and evaluation prompts come from different splits.
"""

import random

from datasets import load_dataset
from transformers import AutoTokenizer


def load_split(split, n, seed):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [(row["messages"][0]["content"], row["messages"][1]["content"])
            for row in ds if len(row["messages"]) >= 2 and row["messages"][0]["role"] == "user"
            and row["messages"][1]["role"] == "assistant"]


def render(tok, user, assistant=None):
    """Return (prefix_text, full_text). The prefix ends where the assistant reply starts."""
    prefix = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    if assistant is None:
        return prefix, None
    full = tok.apply_chat_template([{"role": "user", "content": user},
                                    {"role": "assistant", "content": assistant}],
                                   tokenize=False, enable_thinking=False)
    return prefix, full


def encode_example(tok, user, assistant, max_length):
    """Token ids plus a per-position loss mask for one conversation.

    EAGLE's distillation loss at position i uses the target's distribution for token i+1,
    so a position is supervised when the next token belongs to the assistant reply.
    Returns None when the chat template is not prefix-consistent or no reply tokens fit.
    """
    prefix, full = render(tok, user, assistant)
    if not full.startswith(prefix):
        return None
    prefix_ids = tok(prefix, add_special_tokens=False).input_ids
    full_ids = tok(full, add_special_tokens=False).input_ids
    if full_ids[:len(prefix_ids)] != prefix_ids:
        return None
    full_ids = full_ids[:max_length]
    n_prefix = len(prefix_ids)
    if len(full_ids) <= n_prefix + 1:
        return None
    loss_mask = [0] * len(full_ids)
    for i in range(n_prefix - 1, len(full_ids) - 1):
        loss_mask[i] = 1
    return full_ids, loss_mask


def tokenizer(name):
    return AutoTokenizer.from_pretrained(name)
