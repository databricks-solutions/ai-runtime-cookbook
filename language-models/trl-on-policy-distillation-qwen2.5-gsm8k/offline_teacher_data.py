"""Offline arm, step 1: the teacher writes fixed solutions to the same questions, before any training.

Keeps the first correct sample per question. The offline student trains on this fixed text, so it never sees its
own mistakes the way the on-policy student does.
"""

import json
import time
from pathlib import Path

from vllm import LLM, SamplingParams

from common import (
    MAX_COMPLETION_LENGTH,
    OFFLINE_SAMPLES_PER_QUESTION,
    OFFLINE_TEMPERATURE,
    TEACHER_MODEL,
    WORK,
    is_correct,
    messages,
    train_subset,
)


def main():
    ds = train_subset()
    llm = LLM(model=TEACHER_MODEL, max_model_len=2048, gpu_memory_utilization=0.85)
    params = SamplingParams(
        n=OFFLINE_SAMPLES_PER_QUESTION,
        temperature=OFFLINE_TEMPERATURE,
        top_p=0.95,
        max_tokens=MAX_COMPLETION_LENGTH,
        seed=0,
    )
    t0 = time.time()
    outs = llm.chat([messages(q) for q in ds["question"]], params)
    elapsed = time.time() - t0

    kept, total = [], 0
    for q, gold, out in zip(ds["question"], ds["answer"], outs):
        for candidate in out.outputs:
            total += 1
            if is_correct(candidate.text, gold):
                # Prompt-completion format: TRL trains on the completion tokens only.
                kept.append({"prompt": messages(q), "completion": [{"role": "assistant", "content": candidate.text}]})
                break
    Path(WORK).mkdir(parents=True, exist_ok=True)
    with open(f"{WORK}/offline_sft.jsonl", "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    stats = {
        "offline_teacher_seconds": elapsed,
        "offline_teacher_samples": total,
        "offline_questions_with_correct_solution": len(kept),
        "offline_questions": len(ds),
    }
    print(json.dumps(stats), flush=True)
    Path(f"{WORK}/offline_data_stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
