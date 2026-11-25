"""Load the saved LoRA in a fresh process and generate one image with it."""

import os
from pathlib import Path

import torch
import yaml
from diffusers import ZImagePipeline

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

lora_path = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"] / "pytorch_lora_weights.safetensors"
pipe = ZImagePipeline.from_pretrained(P["model_name"], torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights(str(lora_path))

prompt = P["prompts"][0].format(subject=P["subject"])
image = pipe(
    prompt=prompt,
    height=P["resolution"],
    width=P["resolution"],
    num_inference_steps=P["num_inference_steps"],
    guidance_scale=P["generation_guidance_scale"],
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("/tmp/lora_check.png")
print(f"Generated one image with the LoRA from {lora_path}: {prompt!r}")
