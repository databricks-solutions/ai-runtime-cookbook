"""vLLM batch generation for distillation.

  python generate.py teacher            -> sample teacher solutions for GSM8K train questions,
                                           keep the ones with a correct final answer
  python generate.py eval student_before|student_after -> greedy GSM8K test accuracy for the student
"""

import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams

from common import P, WORK, final_number, messages


def teacher():
    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    ds = ds.select(range(P["num_train_questions"]))
    gold = [final_number(a) for a in ds["answer"]]
    llm = LLM(model=P["teacher_model"], max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(n=P["samples_per_question"], temperature=0.7, top_p=0.95,
                            max_tokens=P["max_tokens"], seed=0)
    t0 = time.time()
    outs = llm.chat([messages(q) for q in ds["question"]], params)
    elapsed = time.time() - t0

    kept, total = [], 0
    for q, g, out in zip(ds["question"], gold, outs):
        for c in out.outputs:
            total += 1
            if final_number(c.text) == g:
                # Prompt-completion format: TRL trains on the completion tokens only.
                kept.append({"prompt": messages(q),
                             "completion": [{"role": "assistant", "content": c.text}]})
                break  # one correct solution per question keeps the dataset balanced
    Path(WORK).mkdir(parents=True, exist_ok=True)
    with open(f"{WORK}/teacher_sft.jsonl", "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    stats = {"teacher_seconds": elapsed, "teacher_completions": total,
             "questions_with_correct_solution": len(kept), "questions": len(ds)}
    print(json.dumps(stats), flush=True)
    Path(f"{WORK}/teacher_stats.json").write_text(json.dumps(stats))


def evaluate(tag):
    model = P["student_model"] if tag == "student_before" else f"{WORK}/student"
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(P["num_eval_questions"]))
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.85)
    outs = llm.chat([messages(q) for q in ds["question"]],
                    SamplingParams(temperature=0.0, max_tokens=P["max_tokens"]))
    acc = sum(final_number(o.outputs[0].text) == final_number(a)
              for o, a in zip(outs, ds["answer"])) / len(ds)
    print(json.dumps({"tag": tag, "gsm8k_accuracy": acc}), flush=True)
    Path(f"{WORK}/eval_{tag}.json").write_text(json.dumps({"gsm8k_accuracy": acc}))


if __name__ == "__main__":
    Path(WORK).mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "teacher":
        teacher()
    else:
        evaluate(sys.argv[2])
