"""Print a bf16-relative comparison table and save results to the volume and MLflow."""

import json
import os
from pathlib import Path

import mlflow
import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

results = [json.loads(p.read_text()) for p in sorted(Path("/tmp/te-bench").glob("result_*.json"))]
base = next((r for r in results if r["precision"] == "bf16"), None)
print(f"\n{'precision':<12}{'tokens/s':>12}{'speedup':>9}{'final loss':>12}{'mem GB':>8}")
for r in results:
    speedup = r["tokens_per_second"] / base["tokens_per_second"] if base else float("nan")
    print(f"{r['precision']:<12}{r['tokens_per_second']:>12,.0f}{speedup:>8.2f}x"
          f"{r['loss_last_20_mean']:>12.4f}{r['peak_mem_gb']:>8.1f}")

run_id = os.environ["MLFLOW_RUN_ID"]
out = Path(P["output_root"]) / run_id
out.mkdir(parents=True, exist_ok=True)
(out / "results.json").write_text(json.dumps(results, indent=2))
print(f"Wrote {out / 'results.json'}")
with mlflow.start_run(run_id=run_id):
    for r in results:
        mlflow.log_metrics({f"{r['precision']}_tokens_per_second": r["tokens_per_second"],
                            f"{r['precision']}_final_loss": r["loss_last_20_mean"]})
