# Databricks notebook source
# ruff: noqa: F821  # dbutils is injected by the Databricks notebook runtime.
"""Register merged weights produced by an existing AIR training run."""

import json
import sys
from pathlib import Path

import yaml

recipe_dir = Path.cwd()
config_path = recipe_dir / "workload.yaml"
if not config_path.is_file():
    raise FileNotFoundError(f"Expected workload.yaml beside this notebook: {config_path}")
sys.path.insert(0, str(recipe_dir))

from register_model import register_model

params = yaml.safe_load(config_path.read_text(encoding="utf-8"))["parameters"]
dbutils.widgets.text("source_training_run_id", "")
dbutils.widgets.text("uc_catalog", "main")
dbutils.widgets.text("uc_schema", "default")
dbutils.widgets.text("uc_model_name", "qwen2_5_0_5b_sft")

training_run_id = dbutils.widgets.get("source_training_run_id").strip()
if not training_run_id:
    raise ValueError("Set source_training_run_id to a successful AIR MLflow run ID.")

registered_model_name = ".".join(
    [
        dbutils.widgets.get("uc_catalog"),
        dbutils.widgets.get("uc_schema"),
        dbutils.widgets.get("uc_model_name"),
    ]
)
result = register_model(
    output_root=params["output_root"],
    training_run_id=training_run_id,
    registered_model_name=registered_model_name,
)
dbutils.jobs.taskValues.set(key="model_version", value=result.model_version)
result_json = json.dumps(
    {
        "source_training_run_id": training_run_id,
        "registered_model_name": result.model_name,
        "registered_model_version": result.model_version,
        "local_vllm_response": result.local_response,
    },
    sort_keys=True,
)
print("REGISTRATION_RESULT=" + result_json, flush=True)
dbutils.notebook.exit(result_json)
