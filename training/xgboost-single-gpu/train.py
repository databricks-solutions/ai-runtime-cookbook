#!/usr/bin/env python3
"""GPU-accelerated XGBoost multi-class training on Forest CoverType.

Trains an XGBoost classifier on a single GPU (``tree_method="hist"`` + ``device="cuda"``), streams
the config, per-round boosting curves and evaluation metrics to MLflow, and registers the model to
Unity Catalog under the ``@champion`` alias. Hyperparameters come from the workload YAML
``parameters:`` block, which ``air`` materializes to the file named by ``$HYPERPARAMETERS_PATH``.

Run from this recipe's directory:

    databricks air run -f workload.yaml
"""

import logging
import os
import time

import mlflow
import numpy as np
import xgboost as xgb
import yaml
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Quiet benign serverless MLflow/py4j chatter emitted during logging (these are not errors).
for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def load_params() -> dict:
    """Read the ``parameters:`` block that ``air`` materializes from the workload YAML."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_uc_schema(catalog: str, schema: str) -> None:
    """Create the target UC schema if missing, so a fresh catalog works with no manual setup.
    Needs CREATE on the catalog; raises an actionable error otherwise."""
    from databricks.sdk import WorkspaceClient

    try:
        WorkspaceClient().schemas.create(name=schema, catalog_name=catalog)
        print(f"Created schema {catalog}.{schema}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "already exists" in msg:
            return
        if any(t in msg for t in ("permission", "denied", "does not have", "unauthorized")):
            raise RuntimeError(
                f"Cannot create {catalog}.{schema}: grant CREATE on catalog '{catalog}', or set "
                f"uc_catalog/uc_schema to an existing schema you can write to."
            ) from e
        raise


def load_data(test_size: float, max_samples: int, random_state: int):
    """Download Forest CoverType and make a deterministic train/test split.

    581,012 rows x 54 numeric features, 7 classes. Source labels are 1..7; XGBoost multi-class
    expects 0..6, so we shift by one. The split is fixed (``test_size`` + ``random_state``) so a
    downstream batch-inference recipe can reproduce the identical held-out set.
    """
    print("Downloading Forest CoverType (~11 MB, cached under ~/scikit-learn_data)...")
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int32)  # 1..7 -> 0..6
    if 0 < max_samples < len(X):
        idx = np.sort(np.random.default_rng(random_state).permutation(len(X))[:max_samples])
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, classes: {int(y.max()) + 1}")
    return X_train, X_test, y_train, y_test


class _MLflowCurveCallback(xgb.callback.TrainingCallback):
    """Stream each round's train/test mlogloss to MLflow so the curves fill in live."""

    def after_iteration(self, model, epoch, evals_log):
        mlflow.log_metrics(
            {f"{split}_mlogloss": vals["mlogloss"][-1] for split, vals in evals_log.items()},
            step=epoch,
        )
        return False  # never request early stop


def main() -> None:
    p = load_params()
    uc_catalog = p.get("uc_catalog", "main")
    uc_schema = p.get("uc_schema", "default")
    model_fqn = f"{uc_catalog}.{uc_schema}.{p.get('registered_model_name', 'xgboost_covertype')}"

    # XGBoost drives the GPU itself (no torch needed). "cuda" uses the attached GPU_1xA10.
    device = str(p.get("device", "cuda"))
    print(f"Training XGBoost on device={device}")

    X_train, X_test, y_train, y_test = load_data(
        float(p.get("test_size", 0.2)), int(p.get("max_samples", -1)), int(p.get("random_state", 42))
    )
    num_class = int(y_train.max()) + 1

    # QuantileDMatrix is the memory-efficient structure for GPU "hist": it bins on the device.
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train)
    dtest = xgb.QuantileDMatrix(X_test, label=y_test, ref=dtrain)
    params = {
        "objective": "multi:softprob",   # per-class probability matrix (n_rows, num_class)
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": device,                # "cuda" runs the whole boosting on the GPU (XGBoost 2.x+ API)
        "max_depth": int(p.get("max_depth", 8)),
        "learning_rate": float(p.get("learning_rate", 0.1)),
        "subsample": float(p.get("subsample", 0.8)),
        "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
    }
    n_estimators = int(p.get("n_estimators", 200))

    mlflow.set_registry_uri("databricks-uc")
    ensure_uc_schema(uc_catalog, uc_schema)

    # AIR provisions an MLflow run for the job and sets MLFLOW_RUN_ID; resume it so everything lands
    # in the run the CLI shows. (Falls back to a fresh run when executed outside AIR.)
    run_id = os.environ.get("MLFLOW_RUN_ID")
    run = mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run(run_name="xgboost-covertype")
    with run:
        # Log the config UP FRONT so the run has content before training finishes.
        mlflow.log_params({
            "dataset": "sklearn.fetch_covtype",
            "num_features": X_train.shape[1],
            "num_classes": num_class,
            "n_estimators": n_estimators,
            **{k: params[k] for k in
               ("max_depth", "learning_rate", "subsample", "colsample_bytree", "tree_method", "device")},
        })

        t0 = time.time()
        booster = xgb.train(
            params, dtrain, num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "test")],
            callbacks=[_MLflowCurveCallback()], verbose_eval=20,
        )
        train_seconds = time.time() - t0

        proba = booster.predict(xgb.DMatrix(X_test))
        preds = np.argmax(proba, axis=1)
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "auc_ovr_macro": float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro")),
            "log_loss": float(log_loss(y_test, proba)),
            "train_seconds": train_seconds,
        }
        mlflow.log_metrics(metrics)
        print(f"Evaluation: {metrics}")

        # Register with a real sample row so the signature/input example match production inputs.
        example = X_train[:5]
        signature = infer_signature(example, booster.predict(xgb.DMatrix(example)))
        info = mlflow.xgboost.log_model(
            xgb_model=booster, name="model", signature=signature,
            input_example=example, registered_model_name=model_fqn,
        )
        MlflowClient().set_registered_model_alias(model_fqn, "champion", info.registered_model_version)
        print(f"Registered {model_fqn} v{info.registered_model_version} and set @champion "
              f"(trained in {train_seconds:.1f}s).")


if __name__ == "__main__":
    main()
