"""Greedy GSM8K test accuracy with vLLM, one model per call.

  python evaluate.py student_before | teacher | opd | offline | control_opd | control_offline

The control tags are the untrained student, saved and reloaded through the same save call as each trained student.
They separate a save or reload effect from the training effect.

Writes per-question correctness and output-format statistics so the summary can compare models on the same questions.
"""

import json
import math
import sys
from pathlib import Path

from vllm import LLM, SamplingParams

from common import (
    MAX_COMPLETION_LENGTH,
    NUM_EVAL_QUESTIONS,
    STUDENT_MODEL,
    TEACHER_MODEL,
    WORK,
    gsm8k,
    is_correct,
    messages,
)

ANSWER_MARKER = "####"


def main(tag):
    model = {
        "student_before": STUDENT_MODEL,
        "teacher": TEACHER_MODEL,
        "opd": f"{WORK}/opd-student",
        "offline": f"{WORK}/offline-student",
        "control_opd": f"{WORK}/control-opd",
        "control_offline": f"{WORK}/control-offline",
    }[tag]
    ds = gsm8k("test").select(range(NUM_EVAL_QUESTIONS))
    llm = LLM(model=model, max_model_len=2048, gpu_memory_utilization=0.85)
    outs = llm.chat(
        [messages(q) for q in ds["question"]],
        SamplingParams(temperature=0.0, max_tokens=MAX_COMPLETION_LENGTH),
    )
    completions = [o.outputs[0] for o in outs]
    correct = [int(is_correct(c.text, a)) for c, a in zip(completions, ds["answer"])]
    n = len(correct)
    acc = sum(correct) / n
    tokenizer = llm.get_tokenizer()
    last_ids = completions[0].token_ids[-4:]
    result = {
        "tag": tag,
        "model": model,
        "n": n,
        "accuracy": acc,
        "standard_error": math.sqrt(acc * (1 - acc) / n),
        "correct": correct,
        "share_with_answer_marker": sum(ANSWER_MARKER in c.text for c in completions) / n,
        "share_finish_length": sum(c.finish_reason == "length" for c in completions) / n,
        "mean_completion_tokens": sum(len(c.token_ids) for c in completions) / n,
        "last_token_ids_first_completion": list(last_ids),
        "last_tokens_first_completion": tokenizer.decode(last_ids),
    }
    Path(WORK).mkdir(parents=True, exist_ok=True)
    Path(f"{WORK}/eval_{tag}.json").write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != "correct"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
