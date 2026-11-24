"""Shared settings, GSM8K prompts and answer scoring for every stage."""

import math
import os
import re

import yaml
from transformers import TrainerCallback

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

STUDENT_MODEL = P["student_model"]
TEACHER_MODEL = P["teacher_model"]
NUM_TRAIN_PROMPTS = int(P["num_train_prompts"])
BATCH_SIZE = int(P["batch_size"])
# One optimizer step per batch, one epoch: both arms take the same number of steps.
TRAIN_STEPS = NUM_TRAIN_PROMPTS // BATCH_SIZE
LEARNING_RATE = float(P["learning_rate"])
BETA = float(P["beta"])
MAX_COMPLETION_LENGTH = int(P["max_completion_length"])
OFFLINE_SAMPLES_PER_QUESTION = int(P["offline_samples_per_question"])
OFFLINE_TEMPERATURE = float(P["offline_temperature"])
NUM_EVAL_QUESTIONS = int(P["num_eval_questions"])
WORK = "/tmp/opd"

SYSTEM = "Solve the math problem step by step. End with the final answer as: #### <number>"


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def final_number(text):
    """The number after '####', else the last number in the text. None if there is none."""
    if text is None:
        return None
    match = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    found = match.group(1) if match else (re.findall(r"-?[\d,]*\.?\d+", text) or [None])[-1]
    if found is None:
        return None
    try:
        return float(found.replace(",", "").rstrip("."))
    except ValueError:
        return None


def is_correct(prediction_text, gold_answer_text):
    """Compare final answers as numbers, so '72' and '72.0' match."""
    pred, gold = final_number(prediction_text), final_number(gold_answer_text)
    return pred is not None and gold is not None and math.isclose(pred, gold, rel_tol=0.0, abs_tol=1e-6)


def gsm8k(split):
    from datasets import load_dataset

    return load_dataset("openai/gsm8k", "main", split=split)


def train_subset():
    """Random GSM8K train questions. The on-policy and offline arms use the same questions."""
    return gsm8k("train").shuffle(seed=0).select(range(NUM_TRAIN_PROMPTS))


class StopOnNonFiniteGradients(TrainerCallback):
    """Fail on non-finite gradients instead of saving a broken student.

    The default logging_nan_inf_filter hides NaN losses, so a run could otherwise exit 0 with NaN weights.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        grad_norm = (logs or {}).get("grad_norm")
        if grad_norm is not None and not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {state.global_step}: {logs}")
