# DreamBooth LoRA fine-tuning of Z-Image, then batch image generation

This recipe teaches the `Tongyi-MAI/Z-Image` text-to-image model (6B parameters) a new subject with DreamBooth LoRA on one H100. It trains on the five photos of one dog in `diffusers/dog-example`, which are paired with the prompt "a photo of sks dog". Training runs 300 steps at 768×768 using the [diffusers example script](https://github.com/huggingface/diffusers/blob/v0.42.1/examples/dreambooth/train_dreambooth_lora_z_image.py).

The recipe then batch-generates images for 8 prompts twice: once with the LoRA and once without it. The whole run takes about 8 minutes.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run writes these files to `<output_root>/<MLflow run ID>/`:

- `with_lora/*.png` and `without_lora/*.png`: one image per prompt, using the same seeds.
- `pytorch_lora_weights.safetensors`: the trained LoRA. Load it with `pipe.load_lora_weights(...)`.
- `prompts.json` and `stats.json`.

In a validation run on an H100, training took 4 minutes 42 seconds (1.06 steps/s). Generation took about 43 seconds per 8 images (30 steps at 768×768). With the LoRA, "sks dog" prompts show the dog from the training photos. Without it, the base model draws a generic dog.

Edit `prompts` in `workload.yaml` to try other scenes. To train on your own subject, replace the dataset download in `train.sh` with a directory of 3–10 images, and change `instance_prompt`.
