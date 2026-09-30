# ModernBERT fine-tuning on AG News (single GPU)

This recipe fine-tunes `answerdotai/ModernBERT-base` into a 4-way AG News topic classifier on a single A10 GPU. The Hugging Face Trainer streams training loss and per-epoch eval metrics to the AIR MLflow run; the recipe then logs the final metrics and registers the model as a `text-classification` pipeline to Unity Catalog.

## Before you run

Set `parameters.uc_catalog` / `parameters.uc_schema` to a Unity Catalog location you can write to; the schema is auto-created when you have `CREATE` on the catalog. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.modernbert_agnews`). Both the model and dataset are public, so no Hugging Face token is needed.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

A new model version is registered in Unity Catalog, and the loss/eval curves plus final metrics are logged to the AIR MLflow run.
