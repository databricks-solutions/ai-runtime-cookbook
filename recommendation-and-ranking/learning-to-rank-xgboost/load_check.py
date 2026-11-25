"""Load this run's ranker in a fresh process and score one test document.

train.py logs the model to the run before this script runs. The README snippet makes the same calls.
"""

import os

import mlflow
import xgboost as xgb
import yaml

from train import load_data


def main() -> None:
    with open(os.environ["HYPERPARAMETERS_PATH"], encoding="utf-8") as handle:
        p = yaml.safe_load(handle)
    model = mlflow.xgboost.load_model(f"runs:/{os.environ['MLFLOW_RUN_ID']}/model")
    features = load_data(p)["test"][0]
    print("Score for the first test document:", model.predict(xgb.DMatrix(features[:1])), flush=True)


if __name__ == "__main__":
    main()
