"""Fine-tune Nemotron 3.5 Super VL with AutoModel on one 8x H100 node, then save the LoRA adapter."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow


def experiment_name():
    # AutoModel's MLflow logger needs the AIR run's experiment name so it logs into
    # the existing MLFLOW_RUN_ID instead of starting a separate run.
    run = mlflow.get_run(os.environ["MLFLOW_RUN_ID"])
    return mlflow.get_experiment(run.info.experiment_id).name


def find_adapter_dir(checkpoint_dir):
    # AutoModel writes LATEST as a symlink and writes LATEST.txt only where symlinks fail.
    if (checkpoint_dir / "LATEST").is_dir():
        return checkpoint_dir / "LATEST" / "model"
    sys.exit(f"No completed LATEST checkpoint found in {checkpoint_dir}")


def main():
    # Training config lives in nemotron_cord_v2.yaml; only inject the AIR experiment name
    # and the local-disk checkpoint path here.
    checkpoint_dir = Path(os.environ["CHECKPOINT_ROOT"])
    subprocess.run([
        "automodel", "nemotron_cord_v2.yaml", "--nproc-per-node=8",
        f"--mlflow.experiment_name={experiment_name()}",
        f"--checkpoint.checkpoint_dir={checkpoint_dir}",
    ], check=True)

    adapter_dir = find_adapter_dir(checkpoint_dir)
    for name in ["adapter_model.safetensors", "adapter_config.json"]:
        path = adapter_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            sys.exit(f"Missing or empty adapter file: {path}")

    # Upload the adapter to the AIR MLflow run.
    run_id = os.environ["MLFLOW_RUN_ID"]
    mlflow.MlflowClient().log_artifacts(run_id, str(adapter_dir), artifact_path="final_adapter")
    print(f"Saved final_adapter to AIR MLflow run {run_id}")

    # Also persist the adapter to a Unity Catalog volume (durable storage, keyed by
    # the MLflow run id so retries never overwrite a prior attempt).
    volume_dir = Path(os.environ["ADAPTER_OUTPUT_ROOT"]) / run_id / "final_adapter"
    shutil.copytree(adapter_dir, volume_dir, dirs_exist_ok=True)
    print(f"Copied final adapter to {volume_dir}")


if __name__ == "__main__":
    main()
