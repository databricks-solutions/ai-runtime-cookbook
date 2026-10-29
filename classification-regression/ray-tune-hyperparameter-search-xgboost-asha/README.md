# XGBoost hyperparameter search on Forest CoverType (Ray Tune + ASHA)

This recipe uses Ray Tune and ASHA to search XGBoost hyperparameters on the public Forest CoverType dataset. The data is split train / validation / test. A Ray cluster of four A10 GPUs runs one trial per GPU; each trial reports its **validation** mlogloss after every boosting round so ASHA prunes trailing trials. The best config is retrained on train+validation and scored once on the **held-out test split** (never used for selection), then registered to Unity Catalog; the best config, held-out test metrics, and per-trial validation mlogloss are logged to the AIR MLflow run.

## Before you run

Set `parameters.uc_catalog` / `parameters.uc_schema` to a Unity Catalog location you can write to; the schema is auto-created when you have `CREATE` on the catalog. Each run registers a new version of `<uc_catalog>.<uc_schema>.<registered_model_name>` (default `main.default.xgboost_covertype`).

## Run

```bash
databricks air run -f workload.yaml
```

## Results

A new model version (the best config, retrained on train+validation) is registered in Unity Catalog, and the best config, held-out test metrics, and every trial's final validation mlogloss are logged to the AIR MLflow run.
