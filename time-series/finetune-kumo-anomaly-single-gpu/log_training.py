"""Publish upstream training losses, checkpoints, logs and data provenance to MLflow."""

import json
import math
import os
import subprocess
from pathlib import Path


def main():
    work_dir = Path(os.environ["WORK_DIR"])
    history = json.loads((work_dir / "checkpoints/epoch_metrics.json").read_text())
    if not history or any(
        not math.isfinite(row[key])
        for row in history
        for key in ["train_loss", "val_loss"]
    ):
        raise ValueError(
            "Expected finite training and validation losses for each epoch."
        )
    parameters = {
        "epochs": int(os.environ.get("EPOCHS", "10")),
        "learning_rate": float(os.environ.get("LEARNING_RATE", "1e-5")),
        "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
        "seed": int(os.environ.get("TRAIN_SEED", "42")),
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(work_dir / "Kumo-TS"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "window_length": 100,
        "window_stride": 100,
    }
    manifest = json.loads((work_dir / "data/manifest.json").read_text())
    parameters.update(
        dataset=manifest["dataset"],
        train_windows=manifest["normal_train_windows"],
        normal_val_windows=manifest["normal_val_windows"],
    )
    if len(history) != parameters["epochs"] or [r["epoch"] for r in history] != list(
        range(1, parameters["epochs"] + 1)
    ):
        raise ValueError("Training history does not match the requested epoch count.")
    summary = {
        "parameters": parameters,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": min(r["val_loss"] for r in history),
        "note": "Upstream training and normal-validation losses; these do not measure anomaly-detection quality.",
    }
    (work_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if run_id := os.environ.get("MLFLOW_RUN_ID"):
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        for key, value in parameters.items():
            client.log_param(run_id, key, str(value))
        for row in history:
            for key in ["train_loss", "val_loss"]:
                client.log_metric(run_id, key, float(row[key]), step=int(row["epoch"]))
        client.log_metric(run_id, "best_val_loss", float(summary["best_val_loss"]))
        evaluation_path = work_dir / "evaluation.json"
        if evaluation_path.exists():
            evaluation = json.loads(evaluation_path.read_text())
            for model_name, metrics in evaluation["models"].items():
                client.log_metric(
                    run_id,
                    f"{model_name}_average_precision",
                    float(metrics["average_precision"]),
                )
                client.log_metric(run_id, f"{model_name}_f1", float(metrics["f1"]))
        for name in ["checkpoints", "data"]:
            client.log_artifacts(
                run_id, str(work_dir / name), artifact_path=f"kumo_finetune/{name}"
            )
        for name in [
            "training.log",
            "data_preparation.log",
            "training_summary.json",
            "evaluation.json",
            "evaluation.log",
        ]:
            path = work_dir / name
            if path.exists():
                client.log_artifact(run_id, str(path), artifact_path="kumo_finetune")
        print(
            f"Uploaded training artifacts to MLflow run {run_id} under kumo_finetune",
            flush=True,
        )


if __name__ == "__main__":
    main()
