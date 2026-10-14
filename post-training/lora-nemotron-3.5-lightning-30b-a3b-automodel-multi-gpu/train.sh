#!/usr/bin/env bash

# Launch eight AutoModel workers on one H100 node.

set -euo pipefail

# Save checkpoints on the node's local disk.

checkpoint_dir="$CHECKPOINT_ROOT"

mlflow_experiment_name="$(python -c 'import os, mlflow; run = mlflow.get_run(os.environ["MLFLOW_RUN_ID"]); print(mlflow.get_experiment(run.info.experiment_id).name)')"

cd "$CODE_SOURCE_PATH"

automodel nemotron_hellaswag.yaml --nproc-per-node=8 \
  --mlflow.experiment_name="$mlflow_experiment_name" \
  --model.backend.experts="$EXPERTS_BACKEND" \
  --model.backend.dispatcher="$TOKEN_DISPATCHER" \
  --step_scheduler.max_steps="$MAX_STEPS" \
  --step_scheduler.num_epochs="$NUM_EPOCHS" \
  --step_scheduler.global_batch_size="${GLOBAL_BATCH_SIZE:-256}" \
  --step_scheduler.local_batch_size="${LOCAL_BATCH_SIZE:-32}" \
  --step_scheduler.ckpt_every_steps="$MAX_STEPS" \
  --step_scheduler.save_checkpoint_every_epoch=false \
  --step_scheduler.val_every_steps="$VALIDATE_EVERY_STEPS" \
  --checkpoint.checkpoint_dir="$checkpoint_dir" \
  --dist_env.timeout_minutes="${DIST_TIMEOUT_MINUTES:-10}" \
  --lr_scheduler.lr_warmup_steps="$WARMUP_STEPS"

# Locate the adapter from AutoModel's completed final checkpoint.

if [[ -d "$checkpoint_dir/LATEST" ]]; then
  adapter_dir="$checkpoint_dir/LATEST/model"
elif [[ -s "$checkpoint_dir/LATEST.txt" ]]; then
  latest="$(< "$checkpoint_dir/LATEST.txt")"
  if [[ "$latest" != /* ]]; then
    latest="$checkpoint_dir/$latest"
  fi
  adapter_dir="$latest/model"
else
  echo "No completed LATEST checkpoint found in $checkpoint_dir" >&2
  exit 1
fi

for file in adapter_model.safetensors adapter_config.json; do
  if [[ ! -s "$adapter_dir/$file" ]]; then
    echo "Missing or empty adapter file: $adapter_dir/$file" >&2
    exit 1
  fi
done

# Upload the adapter to the AIR MLflow run.

mlflow artifacts log-artifacts \
  --local-dir "$adapter_dir" \
  --run-id "$MLFLOW_RUN_ID" \
  --artifact-path final_adapter

echo "Saved final_adapter to AIR MLflow run $MLFLOW_RUN_ID"

# Also persist the adapter to a Unity Catalog volume (durable storage, keyed by
# the MLflow run id so retries never overwrite a prior attempt).

if [[ -z "${ADAPTER_OUTPUT_ROOT:-}" ]]; then
  echo "ADAPTER_OUTPUT_ROOT is not set; set it to a writable volume in workload.yaml" >&2
  exit 1
fi

volume_dir="$ADAPTER_OUTPUT_ROOT/$MLFLOW_RUN_ID/final_adapter"
mkdir -p "$volume_dir"
cp -R "$adapter_dir"/. "$volume_dir"/

echo "Copied final adapter to $volume_dir"
