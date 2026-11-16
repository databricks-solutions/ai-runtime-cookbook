#!/bin/bash
# Fine-tune the SmolVLA vision-language-action policy on SO-100 robot demonstrations with LeRobot.
set -eo pipefail
cd "$CODE_SOURCE_PATH"
read_param() { python -c "import os, yaml; print(yaml.safe_load(open(os.environ['HYPERPARAMETERS_PATH']))['$1'])"; }
RUN_DIR=/tmp/lerobot/smolvla
STEPS=$(read_param steps)
LOG_FREQ=$(( STEPS / 40 > 0 ? STEPS / 40 : 1 ))

# PyAV's wheel bundles a RHEL build of OpenSSL 1.1.1k, used only by its SRT streaming library,
# that aborts with "FATAL FIPS SELFTEST FAILURE" on AI Runtime as soon as `av` is imported.
# Swap in the OpenSSL 1.1 libraries from the manylinux2014 OpenCV wheel, which load cleanly.
SITE=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for lib in ssl crypto; do
  src=$(ls "$SITE"/opencv_python_headless.libs/lib${lib}-*.so.1.1)
  cp "$src" "$(ls "$SITE"/av.libs/lib${lib}-*.so.1.1.1k)"
  cp "$src" "$SITE"/av.libs/  # the replacement libssl looks up libcrypto by its own name
done
python -c "import av; print('PyAV', av.__version__)"

# smolvla_base expects cameras named camera1..camera3; this dataset has top and wrist cameras.
lerobot-train \
  --policy.path="$(read_param policy_path)" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --dataset.repo_id="$(read_param dataset_repo_id)" \
  --dataset.video_backend=pyav \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size="$(read_param batch_size)" \
  --steps="$STEPS" \
  --save_freq="$STEPS" \
  --log_freq="$LOG_FREQ" \
  --num_workers=8 \
  --output_dir="$RUN_DIR" \
  --job_name=smolvla_so100 2>&1 | tee /tmp/train.log

python summarize.py /tmp/train.log "$RUN_DIR" "$LOG_FREQ"
