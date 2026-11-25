# Tabular classification with XGBoost (single GPU)

This recipe trains an XGBoost multi-class classifier on the public Forest CoverType dataset on a single A10 GPU (`tree_method="hist"`, `device="cuda"`). It streams the config, per-round train/test mlogloss curves, and evaluation metrics (accuracy, macro one-vs-rest AUC, log loss) to the AIR MLflow run, and registers the trained model to Unity Catalog.

## Before you run

In `workload.yaml`, set `parameters.uc_catalog` and `parameters.uc_schema` to a catalog and schema in your workspace where you have `USE CATALOG`, `USE SCHEMA`, and `CREATE MODEL`. The default `main.default` usually needs to be changed. The recipe stops with an error before training if the schema is missing. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.xgboost_covertype`).

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

A new model version is registered in Unity Catalog, and the run's config, boosting curves, and evaluation metrics are logged to the AIR MLflow run.

The last step of the run, `load_check.py`, loads the newest registered version in a new Python process and scores one row. These calls are the same ones the snippet below uses. Replace `main.default.xgboost_covertype/<version>` with your `uc_catalog`, `uc_schema`, and `registered_model_name`, and with the version the run registered:

```python
import mlflow
import xgboost as xgb
from sklearn.datasets import fetch_covtype

mlflow.set_registry_uri("databricks-uc")
booster = mlflow.xgboost.load_model("models:/main.default.xgboost_covertype/<version>")
X, _ = fetch_covtype(return_X_y=True)
print(booster.predict(xgb.DMatrix(X[:1].astype("float32"))))
```
