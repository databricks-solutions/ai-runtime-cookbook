#!/usr/bin/env python3
"""Train and evaluate a GPU ranker with query-disjoint LETOR partitions."""

import json
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from mlflow.models import infer_signature
from sklearn.datasets import load_svmlight_file


def load_data(p):
    root = Path(p["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "MQ2008.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(p["dataset_url"], timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    splits = {}
    with zipfile.ZipFile(archive) as source:
        for split, filename in (("train", "train"), ("validation", "vali"), ("test", "test")):
            with source.open(f"MQ2008/Fold{p['fold']}/{filename}.txt") as handle:
                features, labels, queries = load_svmlight_file(
                    handle, n_features=46, query_id=True, zero_based=False, dtype=np.float32,
                )
            order = np.argsort(queries, kind="stable")
            splits[split] = features[order].toarray(), labels[order], queries[order]
    query_sets = [set(values[2]) for values in splits.values()]
    if any(query_sets[i] & query_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Query IDs must be disjoint across train, validation, and test")
    return splits


def mean_ndcg(labels, scores, queries, k):
    """Macro NDCG with exponential gains; zero relevance means zero NDCG."""
    if k < 1 or len(labels) == 0 or np.any(queries[1:] < queries[:-1]):
        raise ValueError("Require k >= 1, nonempty input, and rows sorted by query ID")
    boundaries = np.r_[0, np.flatnonzero(queries[1:] != queries[:-1]) + 1, len(queries)]
    ndcgs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        relevance = labels[start:end]
        limit = min(k, end - start)
        discounts = 1.0 / np.log2(np.arange(2, limit + 2))
        predicted_order = np.argsort(-scores[start:end], kind="stable")[:limit]
        ideal_order = np.argsort(-relevance, kind="stable")[:limit]
        ideal = np.sum(np.expm1(relevance[ideal_order] * np.log(2)) * discounts)
        actual = np.sum(np.expm1(relevance[predicted_order] * np.log(2)) * discounts)
        ndcgs.append(float(actual / ideal) if ideal > 0 else 0.0)
    return float(np.mean(ndcgs))


class MLflowCurveCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        mlflow.log_metrics({
            f"{split}_ndcg": next(iter(values.values()))[-1]
            for split, values in evals_log.items()
        }, step=epoch)
        return False


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    if not 1 <= p["fold"] <= 5 or p["ndcg_k"] < 1 or p["n_estimators"] < 1:
        raise ValueError("Require fold in 1..5, ndcg_k >= 1, and n_estimators >= 1")
    splits = load_data(p)
    x_train, y_train, q_train = splits["train"]
    x_val, y_val, q_val = splits["validation"]
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, qid=q_train)
    dval = xgb.QuantileDMatrix(x_val, label=y_val, qid=q_val, ref=dtrain)
    # The trailing '-' assigns zero to queries with no relevant documents.
    metric = f"ndcg@{p['ndcg_k']}-"
    params = {
        "objective": "rank:ndcg", "eval_metric": metric,
        "tree_method": "hist", "device": "cuda", "seed": p["seed"],
        "max_depth": p["max_depth"], "learning_rate": p["learning_rate"],
        "lambdarank_pair_method": "topk", "lambdarank_num_pair_per_sample": p["ndcg_k"],
    }
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_params({"dataset": "LETOR/MQ2008", **p})
        mlflow.log_params({
            key: value for split, (features, _, queries) in splits.items()
            for key, value in ((f"{split}_rows", len(features)), (f"{split}_queries", len(np.unique(queries))))
        })
        started = time.perf_counter()
        booster = xgb.train(
            params, dtrain, num_boost_round=p["n_estimators"],
            evals=[(dtrain, "train"), (dval, "validation")],
            callbacks=[MLflowCurveCallback(), xgb.callback.EarlyStopping(
                rounds=p["early_stopping_rounds"], metric_name=metric,
                data_name="validation", maximize=True, save_best=True,
            )],
            verbose_eval=20,
        )
        train_seconds = time.perf_counter() - started
        x_test, y_test, q_test = splits["test"]
        scores = booster.predict(xgb.DMatrix(x_test))
        baseline = np.random.default_rng(p["seed"]).random(len(y_test))
        metrics = {
            f"test_ndcg_at_{p['ndcg_k']}": mean_ndcg(y_test, scores, q_test, p["ndcg_k"]),
            f"random_baseline_ndcg_at_{p['ndcg_k']}": mean_ndcg(y_test, baseline, q_test, p["ndcg_k"]),
            "best_iteration": booster.best_iteration, "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict({
            str(int(feature[1:]) + 1): float(gain)
            for feature, gain in booster.get_score(importance_type="gain").items()
        }, "feature_importance.json")
        with tempfile.TemporaryDirectory() as directory:
            predictions_path = Path(directory) / "test_predictions.csv"
            pd.DataFrame({"query_id": q_test, "relevance": y_test, "score": scores}).to_csv(
                predictions_path, index=False,
            )
            mlflow.log_artifact(str(predictions_path))
        example = x_train[:5]
        info = mlflow.xgboost.log_model(
            xgb_model=booster, name="model", input_example=example,
            signature=infer_signature(example, booster.predict(xgb.DMatrix(example))),
        )
        print(f"Test metrics: {json.dumps(metrics)}\nModel: {info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
