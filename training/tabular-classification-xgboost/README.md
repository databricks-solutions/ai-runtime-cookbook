# Tabular classification with XGBoost (single GPU)

This recipe trains an XGBoost multi-class classifier on the public Forest CoverType dataset on a single A10 GPU (`tree_method="hist"`, `device="cuda"`). It streams the config, per-round train/test mlogloss curves, and evaluation metrics (accuracy, macro one-vs-rest AUC, log loss) to the AIR MLflow run, and registers the trained model to Unity Catalog.

## Before you run

In `workload.yaml`, set `parameters.uc_catalog` and `parameters.uc_schema` to a catalog and schema in your workspace where you have `USE CATALOG`, `USE SCHEMA`, and `CREATE MODEL`. The default `main.default` usually needs to be changed. If the schema does not exist, the recipe creates it, which requires `CREATE SCHEMA` on the catalog. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.xgboost_covertype`).

## Run

```bash
databricks air run -f workload.yaml
```

## Results

A new model version is registered in Unity Catalog, and the run's config, boosting curves, and evaluation metrics are logged to the AIR MLflow run.
