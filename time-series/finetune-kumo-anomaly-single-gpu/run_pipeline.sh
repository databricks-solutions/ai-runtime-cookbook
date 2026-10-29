#!/usr/bin/env bash
# Settings come from workload.yaml; AIR sets CODE_SOURCE_PATH and MLFLOW_RUN_ID.
set -euo pipefail

# Check the durable output location before spending GPU time.
volume_dir="${OUTPUT_ROOT:-}/$MLFLOW_RUN_ID"
if [[ "${OUTPUT_ROOT:-}" != /Volumes/* ]] || ! mkdir -p "$volume_dir"; then
    echo "OUTPUT_ROOT must be a writable directory in an existing Unity Catalog volume: ${OUTPUT_ROOT:-<unset>}" >&2
    exit 1
fi

# Keep each run's data and checkpoints separate.
mkdir -p "$WORK_DIR"
if [[ -e "$WORK_DIR/data/manifest.json" || -e "$WORK_DIR/checkpoints" ]]; then
    echo "Use a fresh WORK_DIR for each training run." >&2
    exit 1
fi

# Check out the upstream example and its locked Python environment.
git clone https://github.com/NVIDIA/Kumo-TS.git "$WORK_DIR/Kumo-TS"
git -C "$WORK_DIR/Kumo-TS" checkout --detach af74798bbf7d020d3298f545fb9be383560a18ce
cd "$WORK_DIR/Kumo-TS/Kumo-Anomaly"

# Download and prepare normal training and validation windows from Kaggle.
uv run --frozen --no-dev python "$CODE_SOURCE_PATH/prepare_data.py" \
    --source "${DATA_DIR:-$WORK_DIR/source}" \
    --output "$WORK_DIR/data" \
    --train-rows "$TRAIN_ROWS" \
    --val-rows "$VAL_ROWS" \
    2>&1 | tee "$WORK_DIR/data_preparation.log"

# Pre-download the pinned pretrained weights and config from nvidia/Kumo-Anomaly.
# Pinning the Hugging Face revision keeps the fine-tuning inputs reproducible, so
# training uses the exact validated assets and not the repo's default branch.
WEIGHTS_REVISION=226c5003b0582adc1cac433123f47324bfae39f2
mkdir -p "$WORK_DIR/weights" "$WORK_DIR/checkpoints"
uv run --frozen --no-dev python - \
    "$WORK_DIR/weights/final_model.pth" \
    "$WORK_DIR/weights/curriculum_medium.yaml" \
    "$WEIGHTS_REVISION" <<'PY'
import sys

from sdk.inference_ad import download_model_weights

model_path, config_path, revision = sys.argv[1:4]
download_model_weights(model_path=model_path, config_path=config_path, revision=revision)
PY

# Fine-tune once against the pinned assets. --no-download makes training fail if a
# pinned file is missing, instead of silently fetching the HF default branch.
uv run --frozen --no-dev python examples/finetune_example.py \
    --csv "$WORK_DIR/data/train.csv" \
    --val-csv "$WORK_DIR/data/normal_val.csv" \
    --pretrained-model "$WORK_DIR/weights/final_model.pth" \
    --config "$WORK_DIR/weights/curriculum_medium.yaml" \
    --no-download \
    --epochs "$EPOCHS" \
    --lr "$LEARNING_RATE" \
    --seed "$TRAIN_SEED" \
    --batch-size "$BATCH_SIZE" \
    --num-gpus 1 \
    --window-length 100 --window-stride 100 --split 4 \
    --output-dir "$WORK_DIR/checkpoints" \
    2>&1 | tee "$WORK_DIR/training.log"

# Copy the checkpoints to the volume first, so an MLflow upload error can't lose them.
test -f "$WORK_DIR/checkpoints/final_finetuned_model.pth"
cp -R "$WORK_DIR/checkpoints" "$volume_dir/"

# Measure detection quality on the held-out test set (the last 20% of the series,
# which training never sees). Scoring it with the base and fine-tuned checkpoints
# shows the before/after effect of fine-tuning. This uses the upstream SDK and GPU,
# so it runs in the locked uv environment.
uv run --frozen --no-dev python "$CODE_SOURCE_PATH/evaluate_holdout.py" \
    --source "${DATA_DIR:-$WORK_DIR/source}" \
    --manifest "$WORK_DIR/data/manifest.json" \
    --finetuned-model "$WORK_DIR/checkpoints/best_finetuned_model.pth" \
    --finetuned-config "$WORK_DIR/checkpoints/finetune_config.yaml" \
    --base-model "$WORK_DIR/weights/final_model.pth" \
    --base-config "$WORK_DIR/weights/curriculum_medium.yaml" \
    --eval-rows "$EVAL_ROWS" \
    --output "$WORK_DIR/evaluation.json" \
    2>&1 | tee "$WORK_DIR/evaluation.log"

# Publish checkpoints, epoch losses, held-out metrics and data provenance to
# MLflow. This step only needs MLflow, which the AIR environment already provides.
python "$CODE_SOURCE_PATH/log_training.py"
cp "$WORK_DIR/training_summary.json" "$WORK_DIR/evaluation.json" "$volume_dir/"
echo "Copied checkpoints, training summary and evaluation to $volume_dir"
