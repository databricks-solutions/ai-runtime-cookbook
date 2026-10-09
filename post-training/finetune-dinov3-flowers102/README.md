# DINOv3 image classification on Flowers-102

Fine-tune [DINOv3 ViT-Small](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m), a 21.6M-parameter pretrained vision transformer, on [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) for ten epochs on one A10 GPU. [DINOv3](https://arxiv.org/abs/2508.10104) is a recent state-of-the-art self-supervised vision method; this compact example adapts its distilled backbone and a new classification head with AdamW, separate backbone/head learning rates, warmup, cosine decay, and BF16 mixed precision.

## Before you run

Review the weights' [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license) and the dataset's [usage information](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/README.txt). The public `timm` weights and dataset download automatically without a Hugging Face token. Set `parameters.output_root` to another existing writable Unity Catalog volume directory if needed.

The official splits contain 1,020 training, 1,020 validation, and 6,149 test images. Only training images receive random crops and flips; evaluation uses the pretrained model's normalization and deterministic resize/crop. The validation split selects the best checkpoint, and the test split is evaluated once after training. Ten epochs are a demonstration budget, not a claim of state-of-the-art Flowers-102 accuracy.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains per-epoch loss/accuracy, final test metrics, and the selected model packaged for inference. `<output_root>/<MLFLOW_RUN_ID>/` contains `best_model.pt` (a state dictionary), `model_config.json` (architecture, preprocessing, and label mapping), and `metrics.json`. Output indices 0–101 correspond to Oxford flower labels 1–102.

For inference, recreate the evaluation transform with `timm.data.create_transform(**metadata["data_config"], is_training=False)` from `model_config.json`. The MLflow model accepts batches of transformed float32 images in NCHW order and returns 102 logits; use `argmax` for the class or `softmax` for probabilities. To load the state dictionary directly, create the recorded `timm` model with `pretrained=False, num_classes=102`, load it with `torch.load(..., map_location="cpu", weights_only=True)`, and call `model.eval()`.
