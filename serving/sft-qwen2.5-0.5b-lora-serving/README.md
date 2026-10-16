# Fine-tune and serve Qwen2.5-0.5B with AI Runtime

This recipe fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on 64 Alpaca examples using one A10 GPU, compares base and fine-tuned loss on 16 held-out examples, registers the merged model in Unity Catalog, and deploys it to Custom LLM Serving.

This intentionally small configuration is an inexpensive integration example, not a production training configuration.

The workflow has three separately runnable stages:

1. The AIR CLI runs training and saves merged Hugging Face weights to a Unity Catalog Volume.
2. A registration-only AI Runtime GPU job loads those weights, smoke-tests them with vLLM, and packages and registers an exact model version.
3. A local script asynchronously creates or updates a GPU endpoint. Standard Databricks CLI commands report readiness and query the endpoint.

The Volume and source MLflow run ID form the handoff between stages, so registration or deployment can be retried without repeating training.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Unity Catalog Volume directory.
- Set the catalog, schema, and model defaults in `databricks.yml`. You need `USE CATALOG`, `USE SCHEMA`, and `CREATE MODEL` on the target.
- Confirm that you can create Model Serving endpoints and use `GPU_MEDIUM` workloads.
- The Alpaca dataset is licensed CC BY-NC 4.0. Use data with suitable rights for your application.

Deploy the registration job and install the local deployment dependencies once:

```bash
databricks bundle deploy -t dev --profile <PROFILE>
python -m pip install -r requirements-deploy.txt
```

Redeploy the bundle after changing its job configuration or registration code.

## Run

### 1. Train with AIR

From this directory, run:

```bash
databricks air run -f workload.yaml --watch --profile <PROFILE>
```

The final `TRAINING_RESULT` contains the MLflow run ID and merged-model path. The weights are saved at `<output_root>/<MLFLOW_RUN_ID>/merged`.

### 2. Register the model

Pass the successful AIR run ID to the registration job:

```bash
databricks bundle run register_qwen_sft -t dev --profile <PROFILE> \
  --params source_training_run_id=<MLFLOW_RUN_ID>
```

The job returns the exact registered model version. Each registration run creates a new version, so retrying after registration succeeds can create a duplicate version for the same AIR run.

### 3. Deploy and query the endpoint

Submit the endpoint create or update using the exact registered version:

```bash
python deploy_endpoint.py \
  --model-name <CATALOG>.<SCHEMA>.<MODEL> \
  --model-version <VERSION> \
  --endpoint-name <ENDPOINT_NAME> \
  --profile <PROFILE>
```

The script exits after submitting the change. Check readiness separately:

```bash
databricks serving-endpoints get <ENDPOINT_NAME> --profile <PROFILE>
```

Wait until `state.ready` is `READY` and `state.config_update` is `NOT_UPDATING`, then confirm that `config.served_entities` contains the requested model version. An endpoint can remain ready on its previous version while an update is still in progress. Then send a sample request:

```bash
databricks serving-endpoints query <ENDPOINT_NAME> \
  --profile <PROFILE> \
  --json '{"messages":[{"role":"user","content":"Explain LoRA in one sentence."}],"max_tokens":64}'
```

## Results

The AIR MLflow run contains training parameters, loss metrics, the LoRA adapter, and lineage to the registered model. The endpoint serves the exact model version supplied to `deploy_endpoint.py` with scale to zero enabled.
