"""Batch image generation with and without the trained LoRA, saved to a Unity Catalog volume."""

import json
import os
import time
from pathlib import Path

import mlflow
import torch
import yaml
from diffusers import ZImagePipeline

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)


def generate(pipe, prompts, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(prompts), P["batch_size"]):
        batch = prompts[i : i + P["batch_size"]]
        images = pipe(
            prompt=batch,
            height=P["resolution"],
            width=P["resolution"],
            num_inference_steps=P["num_inference_steps"],
            guidance_scale=P["generation_guidance_scale"],
            generator=[torch.Generator("cuda").manual_seed(i + j) for j in range(len(batch))],
        ).images
        for j, image in enumerate(images):
            image.save(out_dir / f"{i + j:03d}.png")
    return time.time() - t0


def main():
    prompts = [p.format(subject=P["subject"]) for p in P["prompts"]]
    out = Path(P["output_root"]) / os.environ["MLFLOW_RUN_ID"]
    pipe = ZImagePipeline.from_pretrained(P["model_name"], torch_dtype=torch.bfloat16).to("cuda")

    base_s = generate(pipe, prompts, out / "without_lora")
    pipe.load_lora_weights("/tmp/z-image-lora")
    lora_s = generate(pipe, prompts, out / "with_lora")

    stats = {"images_per_variant": len(prompts), "seconds_without_lora": base_s,
             "seconds_with_lora": lora_s, "images_per_second": len(prompts) / lora_s}
    (out / "prompts.json").write_text(json.dumps(prompts, indent=2))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    lora_file = next(Path("/tmp/z-image-lora").glob("*.safetensors"))
    (out / lora_file.name).write_bytes(lora_file.read_bytes())
    print(json.dumps(stats, indent=2))
    print(f"Wrote images and the LoRA weights to {out}")
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_metrics(stats)


if __name__ == "__main__":
    main()
