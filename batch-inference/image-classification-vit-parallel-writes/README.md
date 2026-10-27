# Batch image classification with annotated image output

Classify the Oxford-IIIT Pet test split with [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) on one A10 GPU and write an annotated JPEG for every image. The job writes one image per prediction to a Unity Catalog volume, and `write_workers` overlaps those writes with a thread pool so the GPU is not held waiting on object storage.

Use this recipe when a batch job has to materialize derived images, such as review sets for human labeling or inspection overlays, rather than predictions alone.

## Before you run

Check `parameters.output_root` in [workload.yaml](workload.yaml). You need write access to the existing Unity Catalog volume; change the path if you cannot use the shared default.

The model (Apache 2.0) and dataset (CC BY-SA 4.0) download automatically without a Hugging Face token.

## Run

```bash
databricks air run -f workload.yaml
```

Set `write_workers: 1` to write serially instead. Use `max_images` for a quick test, `batch_size` to fit GPU memory, and `loader_workers` for the image decoding that feeds the GPU.

## Results

Each run writes its files to `<output_root>/<MLFLOW_RUN_ID>/`.

| File | What it contains |
| --- | --- |
| `annotated/<index>.jpg` | One annotated JPEG per image, named by its position in the split. |
| `predictions.parquet` | Index, dataset image ID, predicted label, softmax score, and annotated path. |
| `metrics.json` | Image count, inference and write seconds, total seconds, and images per second. |
| `parameters.json` | The settings used for this run. |

The AIR MLflow run records the same metrics and settings, along with the first eight annotated images.
