"""Stage 6: compare bf16, PTQ and QAD on the same GSM8K test questions and log to the AIR run.

Accuracy gets a binomial standard error. Differences between labels use the paired standard error
(the spread of per-question differences), because the three models answer the same questions.
"""

import json
import math
import os

import mlflow
from common import P, paths

LABELS = ["bf16", "ptq", "qad"]


def load(run, label):
    with open(run / f"eval_{label}.jsonl") as f:
        return [json.loads(line) for line in f]


def paired(a, b):
    """Mean of (b - a) per question with its paired standard error and discordant counts."""
    d = [int(y["correct"]) - int(x["correct"]) for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    return {"diff": mean, "se": math.sqrt(var / n), "a_only": sum(x == -1 for x in d),
            "b_only": sum(x == 1 for x in d)}


def main():
    run = paths()["run"]
    rows = {label: load(run, label) for label in LABELS}
    n = len(rows["bf16"])
    assert all(len(r) == n for r in rows.values()), "labels evaluated on different question sets"

    acc = {}
    for label in LABELS:
        p = sum(r["correct"] for r in rows[label]) / n
        acc[label] = {"accuracy": p, "se": math.sqrt(p * (1 - p) / n)}
    comparisons = {"ptq_minus_bf16": paired(rows["bf16"], rows["ptq"]),
                   "qad_minus_bf16": paired(rows["bf16"], rows["qad"]),
                   "qad_minus_ptq": paired(rows["ptq"], rows["qad"])}

    print(f"GSM8K test, {n} questions ({P['model_name']})")
    print(f"{'model':<8}{'accuracy':>10}{'SE':>8}")
    for label in LABELS:
        print(f"{label:<8}{acc[label]['accuracy']:>10.4f}{acc[label]['se']:>8.4f}")
    for name, c in comparisons.items():
        print(f"{name:<16} diff {c['diff']:+.4f}  paired SE {c['se']:.4f}  "
              f"(questions only the first model got right: {c['a_only']}, only the second: {c['b_only']})")

    results = {"n": n, "model": P["model_name"], "accuracy": acc, "comparisons": comparisons}
    (run / "results.json").write_text(json.dumps(results, indent=2))

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        for label in LABELS:
            mlflow.log_metrics({f"gsm8k_accuracy_{label}": acc[label]["accuracy"],
                                f"gsm8k_accuracy_se_{label}": acc[label]["se"]})
        for name, c in comparisons.items():
            mlflow.log_metrics({f"gsm8k_{name}": c["diff"], f"gsm8k_{name}_paired_se": c["se"]})


if __name__ == "__main__":
    main()
