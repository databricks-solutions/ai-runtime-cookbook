# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy the registered Qwen2.5-0.5B model
# MAGIC Run on CPU serverless. This task calls the Model Serving control plane;
# MAGIC the endpoint provisions its own GPU separately.
# COMMAND ----------

# ruff: noqa: F821  # dbutils is injected by the Databricks notebook runtime.
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    ChatMessage,
    ChatMessageRole,
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

dbutils.widgets.text("uc_catalog", "main")
dbutils.widgets.text("uc_schema", "default")
dbutils.widgets.text("uc_model_name", "qwen2_5_0_5b_sft")
dbutils.widgets.text("model_version", "")
dbutils.widgets.text("endpoint_name", "air-cookbook-qwen05b-lora-serving")

version = dbutils.widgets.get("model_version").strip()
if not version:
    raise ValueError("Set model_version to a registered Unity Catalog model version.")

registered_name = ".".join([
    dbutils.widgets.get("uc_catalog"),
    dbutils.widgets.get("uc_schema"),
    dbutils.widgets.get("uc_model_name"),
])
endpoint_name = dbutils.widgets.get("endpoint_name")
entity = ServedEntityInput(
    entity_name=registered_name,
    entity_version=version,
    workload_type=ServingModelWorkloadType.GPU_MEDIUM,
    workload_size="Small",
    scale_to_zero_enabled=True,
)

client = WorkspaceClient()
try:
    existing_endpoint = client.serving_endpoints.get(endpoint_name)
except ResourceDoesNotExist:
    client.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[entity]),
        timeout=timedelta(minutes=40),
    )
else:
    served_entities = existing_endpoint.config.served_entities if existing_endpoint.config else []
    current_models = {served.entity_name for served in (served_entities or [])}
    if current_models != {registered_name}:
        raise ValueError(
            f"Refusing to replace endpoint {endpoint_name}: it serves {current_models}, "
            f"not only {registered_name}. Choose another endpoint name."
        )
    client.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[entity],
        timeout=timedelta(minutes=40),
    )

prompt = "In machine learning, explain Low-Rank Adaptation (LoRA) in one sentence."
answer = client.serving_endpoints.query(
    name=endpoint_name,
    messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
)
print(f"Deployed {registered_name} version {version} to {endpoint_name}")
print(f"Sample response: {answer.choices[0].message.content}")
