# Databricks notebook source
# MAGIC %md
# MAGIC # Train Qwen2.5-0.5B in a Lakeflow Job
# MAGIC This notebook reads the same parameters as `databricks air run`, logs to
# MAGIC MLflow, tests the merged model with vLLM, registers it in Unity
# MAGIC Catalog, and passes the registered version to the CPU deployment task.
# COMMAND ----------

# ruff: noqa: F821  # dbutils is injected by the Databricks notebook runtime.
import gc
import sys
from pathlib import Path

import mlflow
import torch
import yaml

recipe_dir = Path.cwd()
config_path = recipe_dir / "workload.yaml"
if not config_path.is_file():
    raise FileNotFoundError(f"Expected workload.yaml beside this notebook: {config_path}")
sys.path.insert(0, str(recipe_dir))

from register import register_model
from train import main

params = yaml.safe_load(config_path.read_text(encoding="utf-8"))["parameters"]
dbutils.widgets.text("uc_catalog", "main")
dbutils.widgets.text("uc_schema", "default")
dbutils.widgets.text("uc_model_name", "qwen2_5_0_5b_sft")
with mlflow.start_run(run_name="qwen2-5-0-5b-lora-sft-job") as run:
    run_id = run.info.run_id

main(params=params, run_id=run_id)
gc.collect()
torch.cuda.empty_cache()
version = register_model(
    output_root=params["output_root"],
    training_run_id=run_id,
    catalog=dbutils.widgets.get("uc_catalog"),
    schema=dbutils.widgets.get("uc_schema"),
    model_name=dbutils.widgets.get("uc_model_name"),
)
dbutils.jobs.taskValues.set(key="training_run_id", value=run_id)
dbutils.jobs.taskValues.set(key="model_version", value=version)
print(f"Training MLflow run ID: {run_id}")
print(f"Registered model version: {version}")
