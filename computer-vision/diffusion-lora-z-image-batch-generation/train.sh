#!/bin/bash
# DreamBooth LoRA fine-tuning of Z-Image with the diffusers example script, then batch generation.
set -eo pipefail
cd "$CODE_SOURCE_PATH"
read_param() { python -c "import os, yaml; print(yaml.safe_load(open(os.environ['HYPERPARAMETERS_PATH']))['$1'])"; }
MODEL=$(read_param model_name)
INSTANCE_PROMPT=$(read_param instance_prompt)
DIFFUSERS_TAG=$(read_param diffusers_tag)

# The training script is maintained in the diffusers repository (pinned to the installed version).
curl -fsSL -o /tmp/train_dreambooth_lora_z_image.py \
  "https://raw.githubusercontent.com/huggingface/diffusers/${DIFFUSERS_TAG}/examples/dreambooth/train_dreambooth_lora_z_image.py"

# Five photos of one dog: the subject the LoRA learns to draw from the instance prompt.
# Copy only the images: the training script treats every entry in the directory as an image.
python -c "from huggingface_hub import snapshot_download; snapshot_download('diffusers/dog-example', repo_type='dataset', local_dir='/tmp/dog-download', allow_patterns=['*.jpeg', '*.jpg', '*.png'])"
mkdir -p /tmp/dog
find /tmp/dog-download -maxdepth 1 -type f \( -name '*.jpeg' -o -name '*.jpg' -o -name '*.png' \) -exec cp {} /tmp/dog/ \;
ls /tmp/dog

accelerate launch --num_processes 1 --mixed_precision bf16 /tmp/train_dreambooth_lora_z_image.py \
  --pretrained_model_name_or_path="$MODEL" \
  --instance_data_dir=/tmp/dog \
  --output_dir=/tmp/z-image-lora \
  --mixed_precision=bf16 \
  --gradient_checkpointing \
  --cache_latents \
  --instance_prompt="$INSTANCE_PROMPT" \
  --resolution="$(read_param resolution)" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --guidance_scale=5.0 \
  --rank="$(read_param lora_rank)" \
  --optimizer=adamW \
  --learning_rate="$(read_param learning_rate)" \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps="$(read_param max_train_steps)" \
  --report_to=tensorboard \
  --seed=0

python generate.py
