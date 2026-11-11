# Batch image classification with annotated outputs

This recipe runs batch inference with ViT-Base ([google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)) on 3,669 Oxford-IIIT Pet photos using a single A10 GPU in Databricks AI Runtime. It predicts one of 1,000 ImageNet labels per photo, such as "Egyptian cat" or "English setter", and saves each labeled photo as a JPEG in a Unity Catalog volume.

One file per prediction, rather than one table, is the point of the recipe. Those writes are about 95% of the run. `write_workers` overlaps them in a thread pool so the GPU does not wait on object storage. Use this pattern when a batch job has to produce images for human review.

## Before you run

In `workload.yaml`, set `parameters.output_root` to an existing Unity Catalog volume path you can write to. Keep the shared default if you have write access. The model (Apache 2.0) and dataset (CC BY-SA 4.0) download automatically without a Hugging Face token.

## Run

From the recipe directory, run:

```bash
databricks air run -f workload.yaml --watch
```

For a quick test, set `max_images` to a small number. To compare parallel and serial writes, run once with the default `write_workers` of 32 and again with `write_workers` set to 1. Keep the other settings the same and compare `write_seconds`.

## Results

Each run saves its outputs under `<output_root>/<MLFLOW_RUN_ID>/`.

| Output | What it contains |
| --- | --- |
| `annotated/<index>.jpg` | One JPEG per input image with the predicted label and score, named by its position in the split. |
| `predictions.parquet` | Index, dataset image ID, predicted label, softmax score, and path to the annotated JPEG. |
| `metrics.json` | Image count, model load, data load, inference and write seconds, total seconds, and images per second. `total_seconds` is the sum of those four phases, so it and `images_per_second` include model and data loading. |
| `parameters.json` | The settings used for the run. |

The AIR MLflow run also records the metrics and settings, the predictions table, and up to eight sample JPEGs.
