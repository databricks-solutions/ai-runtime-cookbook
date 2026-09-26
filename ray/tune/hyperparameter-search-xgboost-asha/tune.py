#!/usr/bin/env python3
"""XGBoost hyperparameter search on Forest CoverType with Ray Tune and ASHA.

``ray_bootstrap.sh`` starts a Ray cluster (N x GPU_1xA10, one GPU per node) and runs this on the
head. Ray Tune samples ``num_samples`` configs and runs **one trial per GPU**; each trial reports
its held-out mlogloss after every boosting round, and **ASHA** stops trials that fall behind so GPU
time concentrates on promising configs. The best config is then retrained once on the driver and
registered to Unity Catalog under the ``@champion`` alias.

Hyperparameters/config come from the workload ``parameters:`` block (``$HYPERPARAMETERS_PATH``).
Run from this recipe's directory:

    databricks air run -f workload.yaml
"""

import os

import numpy as np
import ray
import xgboost as xgb
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.datasets import fetch_covtype
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


def load_params() -> dict:
    """Read the ``parameters:`` block that ``air`` materializes from the workload YAML."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def ensure_uc_schema(catalog: str, schema: str) -> None:
    """Create the target UC schema if missing, so a fresh catalog works with no manual setup."""
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


def build_data(test_size: float, max_samples: int, random_state: int):
    """Download Forest CoverType and make a deterministic train/test split (labels 1..7 -> 0..6)."""
    data = fetch_covtype()
    X = data.data.astype(np.float32)
    y = (data.target - 1).astype(np.int32)
    if 0 < max_samples < len(X):
        idx = np.sort(np.random.default_rng(random_state).permutation(len(X))[:max_samples])
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_trial(config, x_train=None, y_train=None, x_test=None, y_test=None,
                num_class=7, n_estimators=200):
    """One Ray Tune trial: train XGBoost on this trial's GPU and report held-out mlogloss per round
    so ASHA can prune. Ray pins one GPU per trial via CUDA_VISIBLE_DEVICES, so device='cuda' is it."""
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train)
    dtest = xgb.QuantileDMatrix(x_test, label=y_test, ref=dtrain)
    params = {
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        **config,
    }

    class _ReportToTune(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            # ASHA ranks/prunes trials on this metric; `iter` (1-based) is the ASHA time axis.
            tune.report({"eval_mlogloss": evals_log["test"]["mlogloss"][-1], "iter": epoch + 1})
            return False

    xgb.train(params, dtrain, num_boost_round=n_estimators,
              evals=[(dtest, "test")], callbacks=[_ReportToTune()], verbose_eval=False)


def main() -> None:
    p = load_params()
    uc_catalog = p.get("uc_catalog", "main")
    uc_schema = p.get("uc_schema", "default")
    model_fqn = f"{uc_catalog}.{uc_schema}.{p.get('registered_model_name', 'xgboost_covertype')}"
    test_size = float(p.get("test_size", 0.2))
    max_samples = int(p.get("max_samples", 100000))
    random_state = int(p.get("random_state", 42))
    num_samples = int(p.get("num_samples", 16))
    n_estimators = int(p.get("n_estimators", 200))
    grace_period = int(p.get("grace_period", 20))

    ray.init(address="auto")
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < 1:
        raise SystemExit("No GPUs registered with Ray; check GPU discovery on the cluster.")
    print(f"Ray cluster ready: {total_gpus} GPU(s); running {num_samples} trials, "
          f"up to {total_gpus} concurrently.", flush=True)

    x_train, x_test, y_train, y_test = build_data(test_size, max_samples, random_state)
    num_class = int(y_train.max()) + 1
    print(f"Train: {x_train.shape}, Test: {x_test.shape}, classes: {num_class}", flush=True)

    param_space = {
        "max_depth": tune.choice([4, 6, 8, 10, 12]),
        "learning_rate": tune.loguniform(0.02, 0.3),
        "subsample": tune.uniform(0.6, 1.0),
        "colsample_bytree": tune.uniform(0.6, 1.0),
        "min_child_weight": tune.choice([1, 3, 5, 7]),
        "reg_lambda": tune.loguniform(0.5, 5.0),
    }
    tuner = tune.Tuner(
        # with_resources gives each trial a whole GPU; with_parameters ships the (already split)
        # data to trials by value through Ray's object store.
        tune.with_resources(
            tune.with_parameters(
                train_trial, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                num_class=num_class, n_estimators=n_estimators,
            ),
            resources={"gpu": 1},
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="eval_mlogloss",
            mode="min",
            scheduler=ASHAScheduler(max_t=n_estimators, grace_period=grace_period, reduction_factor=2),
            num_samples=num_samples,
        ),
    )
    results = tuner.fit()
    if results.num_errors:
        raise RuntimeError(f"{results.num_errors}/{len(results)} trials errored; see the logs above.")
    best = results.get_best_result("eval_mlogloss", "min")
    print(f"\nBest config:        {best.config}", flush=True)
    print(f"Best eval_mlogloss: {best.metrics['eval_mlogloss']:.4f}", flush=True)

    # Retrain the best config once on the driver's GPU, then register it to Unity Catalog.
    import mlflow
    from mlflow.models.signature import infer_signature
    from mlflow.tracking import MlflowClient

    mlflow.set_registry_uri("databricks-uc")
    ensure_uc_schema(uc_catalog, uc_schema)
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train)
    dtest = xgb.QuantileDMatrix(x_test, label=y_test, ref=dtrain)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": num_class, "eval_metric": "mlogloss",
         "tree_method": "hist", "device": "cuda", **best.config},
        dtrain, num_boost_round=n_estimators, evals=[(dtest, "test")], verbose_eval=False,
    )
    proba = booster.predict(xgb.DMatrix(x_test))
    test_accuracy = float(accuracy_score(y_test, np.argmax(proba, axis=1)))
    test_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    print(f"Best model retrained: accuracy={test_accuracy:.4f} auc_ovr_macro={test_auc:.4f}", flush=True)

    # AIR injects MLFLOW_RUN_ID and the databricks tracking URI on the head; gating on the var keeps
    # this runnable off-platform (where it is unset).
    if os.environ.get("MLFLOW_RUN_ID"):
        with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
            mlflow.log_params({
                "dataset": "sklearn.fetch_covtype",
                "num_samples": num_samples,
                "n_estimators": n_estimators,
                "scheduler": "ASHA",
                "asha_max_t": n_estimators,
                "asha_grace_period": grace_period,
                **{f"best_{k}": v for k, v in best.config.items()},
            })
            mlflow.log_metrics({
                "best_eval_mlogloss": float(best.metrics["eval_mlogloss"]),
                "test_accuracy": test_accuracy,
                "test_auc_ovr_macro": test_auc,
            })
            # Every trial's final mlogloss, for comparison in the Experiments UI.
            for i, r in enumerate(results):
                if r.metrics and "eval_mlogloss" in r.metrics:
                    mlflow.log_metric("trial_eval_mlogloss", float(r.metrics["eval_mlogloss"]), step=i)

            booster.set_param({"device": "cpu"})  # portable artifact
            example = x_train[:5]
            signature = infer_signature(example, booster.predict(xgb.DMatrix(example)))
            info = mlflow.xgboost.log_model(
                xgb_model=booster, name="model", signature=signature,
                input_example=example, registered_model_name=model_fqn,
            )
            MlflowClient().set_registered_model_alias(model_fqn, "champion", info.registered_model_version)
            print(f"Registered {model_fqn} v{info.registered_model_version} and set @champion.", flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()
