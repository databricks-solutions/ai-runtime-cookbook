# XGBoost training on Forest CoverType (single GPU)

This recipe trains an XGBoost multi-class classifier on the public Forest CoverType dataset on a single A10 GPU (`tree_method="hist"`, `device="cuda"`). It streams the config, per-round train/test mlogloss curves, and evaluation metrics (accuracy, macro one-vs-rest AUC, log loss) to the AIR MLflow run, and registers the trained model to Unity Catalog.

## Before you run

Set `parameters.uc_catalog` / `parameters.uc_schema` to a Unity Catalog location you can write to; the schema is auto-created when you have `CREATE` on the catalog. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.xgboost_covertype`).

## Run

```bash
databricks air run -f workload.yaml
```

## Results

A new model version is registered in Unity Catalog, and the run's config, boosting curves, and evaluation metrics are logged to the AIR MLflow run.
