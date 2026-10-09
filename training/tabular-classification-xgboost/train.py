#!/usr/bin/env python3
"""Train a GPU XGBoost fraud classifier on time-ordered credit card transactions."""

import os
import time

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def load_data(data_url, validation_fraction, test_fraction):
    """Split chronologically so validation and test transactions occur after training transactions."""
    if not (0 < validation_fraction and 0 < test_fraction and validation_fraction + test_fraction < 1):
        raise ValueError("Require positive validation/test fractions that sum to less than 1")
    data = pd.read_parquet(data_url).sort_values("Time", kind="stable")
    if data.isna().any().any():
        raise ValueError("Expected no missing values")
    # Time counts seconds from the first transaction, so it does not generalize to later periods.
    features = data.drop(columns=["Time", "Class"]).astype(np.float32)
    labels = data["Class"].astype(int).to_numpy()
    train_end = int(len(data) * (1 - validation_fraction - test_fraction))
    val_end = int(len(data) * (1 - test_fraction))
    splits = {
        "train": slice(0, train_end),
        "validation": slice(train_end, val_end),
        "test": slice(val_end, len(data)),
    }
    return {name: (features.iloc[rows], labels[rows]) for name, rows in splits.items()}


def best_f1_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[f1.argmax()])


class MLflowCurveCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        mlflow.log_metrics(
            {f"{split}_aucpr": values["aucpr"][-1] for split, values in evals_log.items()},
            step=epoch,
        )
        return False


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    splits = load_data(p["data_url"], p["validation_fraction"], p["test_fraction"])
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = splits.values()
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, ref=dtrain)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "device": p["device"],
        "seed": p["random_state"],
        **{key: p[key] for key in ("max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight")},
    }

    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_params({"dataset": "OpenML/creditcard", **p})
        for name, (_, labels) in splits.items():
            mlflow.log_params({f"{name}_rows": len(labels), f"{name}_frauds": int(labels.sum())})
        started = time.perf_counter()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=p["n_estimators"],
            evals=[(dtrain, "train"), (dval, "validation")],
            callbacks=[
                MLflowCurveCallback(),
                xgb.callback.EarlyStopping(
                    rounds=p["early_stopping_rounds"], data_name="validation", metric_name="aucpr", save_best=True,
                ),
            ],
            verbose_eval=20,
        )
        train_seconds = time.perf_counter() - started
        # Pick the alert threshold on validation data, then score the later test period once.
        threshold = best_f1_threshold(y_val, booster.predict(xgb.DMatrix(x_val)))
        probabilities = booster.predict(xgb.DMatrix(x_test))
        predicted = (probabilities >= threshold).astype(int)
        precision = precision_score(y_test, predicted, zero_division=0)
        recall = recall_score(y_test, predicted)
        metrics = {
            "test_average_precision": float(average_precision_score(y_test, probabilities)),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "test_alerts": int(predicted.sum()),
            "decision_threshold": threshold,
            "best_iteration": booster.best_iteration,
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"decision_threshold": threshold, "positive_label": "fraud"}, "decision_threshold.json")
        example = x_train.iloc[:5]
        info = mlflow.xgboost.log_model(
            xgb_model=booster,
            name="model",
            signature=infer_signature(example, booster.predict(xgb.DMatrix(example))),
            input_example=example,
        )
        print(f"Test metrics: {metrics}\nModel: {info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
