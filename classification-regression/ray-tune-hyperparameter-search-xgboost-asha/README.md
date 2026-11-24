# XGBoost hyperparameter search on Forest CoverType (Ray Tune + ASHA)

This recipe uses Ray Tune and ASHA to search XGBoost hyperparameters on the public Forest CoverType dataset. The data is split train / validation / test. A Ray cluster of four A10 GPUs runs one trial per GPU; each trial reports its **validation** mlogloss after every boosting round so ASHA prunes trailing trials. The best config is retrained on train+validation and scored once on the **held-out test split** (never used for selection), then registered to Unity Catalog; the best config, held-out test metrics, and per-trial validation mlogloss are logged to the AIR MLflow run.

## Before you run

Set `uc_catalog`, `uc_schema`, and `registered_model_name` in the workload parameters to a writable Unity Catalog target (default `main.default.xgboost_covertype`). The workload identity needs `USE CATALOG` and `USE SCHEMA`. After tuning, the recipe reuses the schema or creates it if needed, which requires `CREATE SCHEMA` on the catalog.

MLflow attempts to create the registered model before adding a version. Check `CREATE MODEL` (or `CREATE FUNCTION`) on the schema and, for an existing model, ownership or `CREATE MODEL VERSION`. After the Ray cluster starts, the driver checks the target using read-only metadata before connecting to Ray, loading data, or starting trials. If Unity Catalog returns `CATALOG_DOES_NOT_EXIST` for the configured catalog, the driver stops. Set `uc_catalog` to an existing writable catalog. The driver also stops for a Delta Sharing, foreign, or system catalog, or when the caller has only browse access to the catalog or schema. When metadata cannot confirm a privilege, the driver prints what to check and continues. Unity Catalog evaluates authorization when MLflow writes.

Private regression tests exercise the driver and registration checks outside the uploaded recipe snapshot.

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

A new model version (the best config, retrained on train+validation) is registered in Unity Catalog, and the best config, held-out test metrics, and every trial's final validation mlogloss are logged to the AIR MLflow run.
