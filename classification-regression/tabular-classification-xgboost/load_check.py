"""Load the newest registered model version in a fresh process and score one row.

train.py registers the model before this script runs. The README snippet runs the same calls.
"""

import os

import mlflow
import xgboost as xgb
import yaml
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_covtype


def main() -> None:
    with open(os.environ["HYPERPARAMETERS_PATH"], encoding="utf-8") as handle:
        p = yaml.safe_load(handle)
    name = f"{p.get('uc_catalog', 'main')}.{p.get('uc_schema', 'default')}.{p.get('registered_model_name', 'xgboost_covertype')}"
    mlflow.set_registry_uri("databricks-uc")
    versions = MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name='{name}'")
    version = max(int(v.version) for v in versions)

    booster = mlflow.xgboost.load_model(f"models:/{name}/{version}")
    X, _ = fetch_covtype(return_X_y=True)
    print(f"{name} version {version} scores one row:", booster.predict(xgb.DMatrix(X[:1].astype("float32"))), flush=True)


if __name__ == "__main__":
    main()
