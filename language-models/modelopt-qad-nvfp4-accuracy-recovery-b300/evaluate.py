"""Stage 5: greedy GSM8K test evaluation with vLLM. Usage: python evaluate.py bf16|ptq|qad

Writes one row per test question (correct, prediction, output length) to the run directory, so
summarize.py can compare the labels question by question.
"""

import json
import sys
import time

from common import P, load_gsm8k_test, gold_number, model_path, paths, parse_number, SYSTEM
from vllm import LLM, SamplingParams


def main(label):
    ds = load_gsm8k_test(P["eval_limit"])
    messages = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
                for q in ds["question"]]
    llm = LLM(model=model_path(label), max_model_len=4096, gpu_memory_utilization=0.85)
    params = SamplingParams(temperature=0.0, max_tokens=P["max_tokens"])
    t0 = time.time()
    outs = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False})
    seconds = time.time() - t0

    rows = []
    for i, (answer, out) in enumerate(zip(ds["answer"], outs)):
        gold = gold_number(answer)
        pred = parse_number(out.outputs[0].text)
        rows.append({"idx": i, "gold": gold, "pred": pred,
                     "correct": pred is not None and abs(pred - gold) < 1e-6,
                     "output_tokens": len(out.outputs[0].token_ids),
                     "text": out.outputs[0].text})
    accuracy = sum(r["correct"] for r in rows) / len(rows)
    print(f"[eval] {label}: {len(rows)} questions, accuracy {accuracy:.4f}, "
          f"{sum(r['output_tokens'] for r in rows) / seconds:,.0f} output tok/s "
          f"(generate call {seconds:.0f}s)", flush=True)

    run = paths()["run"]
    run.mkdir(parents=True, exist_ok=True)
    with open(run / f"eval_{label}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
