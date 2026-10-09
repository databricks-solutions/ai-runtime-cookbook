#!/usr/bin/env python3
"""Train a GPU XGBoost classifier on the public Forest CoverType dataset."""

import os
import time

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
import yaml
from mlflow.models import infer_signature
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(test_size, validation_size, max_samples, seed):
    """Reserve the test set before sampling training data or selecting trees."""
    data = fetch_covtype()
    features = data.data.astype(np.float32)
    labels = (data.target - 1).astype(np.int32)  # UCI labels 1..7 -> XGBoost labels 0..6.
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=validation_size, random_state=seed, stratify=y_train
    )
    if 0 < max_samples < len(x_train):
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size=max_samples, random_state=seed, stratify=y_train
        )
    return x_train, x_val, x_test, y_train, y_val, y_test


class MLflowCurveCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        mlflow.log_metrics(
            {f"{split}_mlogloss": values["mlogloss"][-1] for split, values in evals_log.items()},
            step=epoch,
        )
        return False


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        p["test_size"], p["validation_size"], p["max_samples"], p["random_state"]
    )
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, ref=dtrain)
    params = {
        "objective": "multi:softprob",
        "num_class": 7,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": p["device"],
        "seed": p["random_state"],
        **{key: p[key] for key in ("max_depth", "learning_rate", "subsample", "colsample_bytree")},
    }

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_params({"dataset": "UCI/Covertype", **p})
        mlflow.log_params({"train_rows": len(y_train), "validation_rows": len(y_val), "test_rows": len(y_test)})
        started = time.perf_counter()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=p["n_estimators"],
            evals=[(dtrain, "train"), (dval, "validation")],
            callbacks=[
                MLflowCurveCallback(),
                xgb.callback.EarlyStopping(
                    rounds=p["early_stopping_rounds"], data_name="validation", metric_name="mlogloss", save_best=True,
                ),
            ],
            verbose_eval=20,
        )
        train_seconds = time.perf_counter() - started
        # save_best returns only trees selected on validation; evaluate test data once.
        probabilities = booster.predict(xgb.DMatrix(x_test))
        predicted = probabilities.argmax(axis=1)
        metrics = {
            "test_accuracy": float(accuracy_score(y_test, predicted)),
            "test_f1_macro": float(f1_score(y_test, predicted, average="macro")),
            "test_auc_ovr_macro": float(roc_auc_score(y_test, probabilities, multi_class="ovr")),
            "test_log_loss": float(log_loss(y_test, probabilities)),
            "best_iteration": booster.best_iteration,
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict({str(i): i + 1 for i in range(7)}, "class_labels.json")
        example = x_train[:5]
        # Registration is optional and targets an existing UC schema; never create one.
        registered_name = p["registered_model_name"] or None
        if registered_name:
            mlflow.set_registry_uri("databricks-uc")
        info = mlflow.xgboost.log_model(
            xgb_model=booster,
            name="model",
            signature=infer_signature(example, booster.predict(xgb.DMatrix(example))),
            input_example=example,
            registered_model_name=registered_name,
        )
        print(f"Test metrics: {metrics}\nModel: {info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
