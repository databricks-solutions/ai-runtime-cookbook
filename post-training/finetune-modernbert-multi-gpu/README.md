# ModernBERT fine-tuning on AG News (multi-GPU DDP)

This recipe fine-tunes `answerdotai/ModernBERT-base` into a 4-way AG News topic classifier with PyTorch DDP, launched via `torchrun` across a `GPU_8xH100` node (the same command also scales to multiple nodes). Rank 0 owns the AIR MLflow run — it streams the loss/eval curves, logs the final metrics, and registers the model as a `text-classification` pipeline to Unity Catalog; the other ranks skip MLflow.

## Before you run

Set `parameters.uc_catalog` / `parameters.uc_schema` to a Unity Catalog location you can write to; the schema is auto-created when you have `CREATE` on the catalog. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.modernbert_agnews`). Both the model and dataset are public, so no Hugging Face token is needed.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

A new model version is registered in Unity Catalog, and rank 0's loss/eval curves plus final metrics are logged to the AIR MLflow run.
