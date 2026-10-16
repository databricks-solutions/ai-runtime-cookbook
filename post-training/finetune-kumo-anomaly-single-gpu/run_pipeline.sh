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

# Fine-tune once. The upstream script downloads the pretrained assets from
# nvidia/Kumo-Anomaly (validated with Hugging Face revision 226c500).
mkdir -p "$WORK_DIR/weights" "$WORK_DIR/checkpoints"
uv run --frozen --no-dev python examples/finetune_example.py \
    --csv "$WORK_DIR/data/train.csv" \
    --val-csv "$WORK_DIR/data/normal_val.csv" \
    --pretrained-model "$WORK_DIR/weights/final_model.pth" \
    --config "$WORK_DIR/weights/curriculum_medium.yaml" \
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

# Publish checkpoints, epoch losses and data provenance to MLflow. This step
# only needs MLflow, which the AIR environment already provides.
python "$CODE_SOURCE_PATH/log_training.py"
cp "$WORK_DIR/training_summary.json" "$volume_dir/"
echo "Copied checkpoints and training summary to $volume_dir"
