"""Create or update a Custom LLM Serving endpoint without waiting."""

from __future__ import annotations

import argparse
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

DEFAULT_ENDPOINT_NAME = "air-cookbook-qwen05b-lora-serving"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Three-level Unity Catalog model name")
    parser.add_argument("--model-version", required=True, help="Exact numeric model version")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument("--profile", help="Databricks CLI profile")
    return parser.parse_args()


def enum_value(value):
    return value.value if hasattr(value, "value") else value


def entity_matches(entity, desired: ServedEntityInput) -> bool:
    return all(
        [
            entity.entity_name == desired.entity_name,
            str(entity.entity_version) == str(desired.entity_version),
            enum_value(entity.workload_type) == enum_value(desired.workload_type),
            entity.workload_size == desired.workload_size,
            entity.scale_to_zero_enabled == desired.scale_to_zero_enabled,
        ]
    )


def served_entities(config) -> list:
    return list(config.served_entities or []) if config else []


def submit_endpoint(client, endpoint_name: str, desired: ServedEntityInput) -> str:
    try:
        existing = client.serving_endpoints.get(endpoint_name)
    except ResourceDoesNotExist:
        client.serving_endpoints.create(
            name=endpoint_name,
            config=EndpointCoreConfigInput(name=endpoint_name, served_entities=[desired]),
        )
        action = "created"
    else:
        current = served_entities(existing.config)
        pending = served_entities(existing.pending_config)
        config_update = (
            enum_value(existing.state.config_update)
            if existing.state and existing.state.config_update
            else None
        )
        update_in_progress = config_update == "IN_PROGRESS"
        if len(current) > 1:
            raise ValueError(
                f"Refusing to replace endpoint {endpoint_name}: it has multiple active served entities."
            )
        observed_models = {
            entity.entity_name
            for entity in current + (pending if update_in_progress else [])
            if entity.entity_name is not None
        }
        if observed_models and observed_models != {desired.entity_name}:
            raise ValueError(
                f"Refusing to replace endpoint {endpoint_name}: it serves "
                f"{sorted(observed_models)}, not only {desired.entity_name}."
            )
        if update_in_progress:
            if len(pending) == 1 and entity_matches(pending[0], desired):
                action = "already updating"
            else:
                raise RuntimeError(
                    f"Endpoint {endpoint_name} already has a different update in progress. "
                    "Wait for it to finish before retrying."
                )
        elif len(current) == 1 and entity_matches(current[0], desired):
            action = "unchanged"
        else:
            client.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=[desired],
            )
            action = "updated"
    return action


def main() -> None:
    args = parse_args()
    desired = ServedEntityInput(
        entity_name=args.model_name,
        entity_version=str(args.model_version),
        workload_type=ServingModelWorkloadType.GPU_MEDIUM,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    action = submit_endpoint(client, args.endpoint_name, desired)

    result = {
        "action": action,
        "endpoint_name": args.endpoint_name,
        "model_name": args.model_name,
        "model_version": str(args.model_version),
    }
    print("DEPLOYMENT_SUBMITTED=" + json.dumps(result, sort_keys=True))
    print(
        f"Check readiness with: databricks serving-endpoints get {args.endpoint_name}"
        + (f" --profile {args.profile}" if args.profile else "")
    )


if __name__ == "__main__":
    main()
