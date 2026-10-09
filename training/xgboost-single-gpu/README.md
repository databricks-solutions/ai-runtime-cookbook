# XGBoost classification on Forest CoverType

Train a seven-class XGBoost classifier on the public [UCI Forest CoverType dataset](https://archive.ics.uci.edu/dataset/31/covertype) (581,012 rows, 54 numeric features) on one A10 GPU. GPU histogram training uses up to 500 boosting rounds, with early stopping on a stratified validation split and final evaluation on a separate test split (64%/16%/20%).

## Before you run

The dataset is downloaded automatically and is distributed under CC BY 4.0. To optionally register the model in Unity Catalog, set `parameters.registered_model_name` to a fully qualified name in an existing schema where you have model-creation permissions; each run creates a new model version. The default only logs the model to MLflow.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains training/validation log-loss curves, final test accuracy, macro F1, macro one-vs-rest AUC, log loss, and a packaged XGBoost model with an input signature. Its seven output columns are probabilities for UCI cover types 1–7; `class_labels.json` records the mapping. `max_samples` limits only training rows for a smaller demonstration.
