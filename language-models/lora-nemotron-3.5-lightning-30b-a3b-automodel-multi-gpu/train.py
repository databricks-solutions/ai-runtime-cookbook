#!/usr/bin/env python3
"""LoRA fine-tuning of Nemotron 3.5 Lightning with NeMo AutoModel on one 8x H100 node.

Runs AutoModel with eight workers, uploads the final adapter to the AIR MLflow run, and copies it
to ADAPTER_OUTPUT_ROOT. The training config lives in nemotron_hellaswag.yaml.
"""

import os
import shutil
import subprocess
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def find_adapter_dir(checkpoint_dir: Path) -> Path:
    """Locate the adapter in AutoModel's completed final checkpoint."""
    latest_dir = checkpoint_dir / "LATEST"
    latest_file = checkpoint_dir / "LATEST.txt"
    if latest_dir.is_dir():
        return latest_dir / "model"
    if latest_file.is_file() and latest_file.stat().st_size > 0:
        latest = Path(latest_file.read_text().strip())
        if not latest.is_absolute():
            latest = checkpoint_dir / latest
        return latest / "model"
    raise RuntimeError(f"No completed LATEST checkpoint found in {checkpoint_dir}")


def main() -> None:
    run_id = os.environ["MLFLOW_RUN_ID"]
    checkpoint_dir = Path(os.environ["CHECKPOINT_ROOT"])

    # AutoModel's MLflow logger needs the AIR run's experiment name so it logs into the existing
    # run instead of starting a separate one.
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name

    subprocess.run(
        [
            "automodel", "nemotron_hellaswag.yaml", "--nproc-per-node=8",
            f"--mlflow.experiment_name={experiment_name}",
            f"--checkpoint.checkpoint_dir={checkpoint_dir}",
        ],
        check=True,
    )

    adapter_dir = find_adapter_dir(checkpoint_dir)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        path = adapter_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty adapter file: {path}")

    MlflowClient().log_artifacts(run_id, str(adapter_dir), artifact_path="final_adapter")
    print(f"Saved final_adapter to AIR MLflow run {run_id}", flush=True)

    # Also persist the adapter to a Unity Catalog volume, keyed by the MLflow run id so retries
    # never overwrite a prior attempt.
    volume_dir = Path(os.environ["ADAPTER_OUTPUT_ROOT"]) / run_id / "final_adapter"
    volume_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(adapter_dir, volume_dir, dirs_exist_ok=True)
    print(f"Copied final adapter to {volume_dir}", flush=True)


if __name__ == "__main__":
    main()
