"""DreamBooth LoRA fine-tuning of Z-Image with the diffusers example training script.

Downloads the training script that matches the installed diffusers version, prepares the
training images, and runs the script with `accelerate launch` using the recipe parameters.
"""

import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import diffusers
import yaml
from huggingface_hub import snapshot_download

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

SCRIPT_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/v{diffusers.__version__}"
    "/examples/dreambooth/train_dreambooth_lora_z_image.py"
)
SCRIPT = Path("/tmp/train_dreambooth_lora_z_image.py")
IMAGES = Path("/tmp/dog")
LORA_DIR = "/tmp/z-image-lora"


def main():
    urllib.request.urlretrieve(SCRIPT_URL, SCRIPT)

    # Five photos of one dog: the subject the LoRA learns. The training script treats every
    # entry in the directory as an image, so copy only the image files out of the download.
    download = Path(snapshot_download("diffusers/dog-example", repo_type="dataset",
                                      local_dir="/tmp/dog-download",
                                      allow_patterns=["*.jpeg", "*.jpg", "*.png"]))
    IMAGES.mkdir(exist_ok=True)
    for image in download.iterdir():
        if image.suffix.lower() in {".jpeg", ".jpg", ".png"}:
            shutil.copy(image, IMAGES)
    print(f"Training on {len(list(IMAGES.iterdir()))} images of {P['subject']!r}")

    subprocess.run([
        "accelerate", "launch", "--num_processes", "1", "--mixed_precision", "bf16", str(SCRIPT),
        f"--pretrained_model_name_or_path={P['model_name']}",
        f"--instance_data_dir={IMAGES}",
        f"--output_dir={LORA_DIR}",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        f"--instance_prompt=a photo of {P['subject']}",
        f"--resolution={P['resolution']}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--guidance_scale=5.0",
        f"--rank={P['lora_rank']}",
        "--optimizer=adamW",
        f"--learning_rate={P['learning_rate']}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        f"--max_train_steps={P['max_train_steps']}",
        "--report_to=tensorboard",
        "--seed=0",
    ], check=True)


if __name__ == "__main__":
    main()
