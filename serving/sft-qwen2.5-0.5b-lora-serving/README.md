# Fine-tune and serve Qwen2.5-0.5B with AI Runtime

This recipe fine-tunes `Qwen/Qwen2.5-0.5B-Instruct` with LoRA on 64 Alpaca examples using one A10 GPU, compares base and fine-tuned loss on 16 held-out examples, registers the merged model in Unity Catalog, and deploys it to Custom LLM Serving.

This intentionally small configuration is an inexpensive integration example, not a production training configuration.

The workflow has three separately runnable stages:

1. The AIR CLI runs training and saves merged Hugging Face weights to a Unity Catalog Volume.
2. A registration-only AI Runtime GPU job loads those weights, smoke-tests them with vLLM, and packages and registers an exact model version.
3. A local script creates or updates a GPU endpoint and, with `--wait`, returns once it serves the requested version. A Databricks CLI command queries the endpoint.

The Volume and source MLflow run ID form the handoff between stages, so registration or deployment can be retried without repeating training.

## Before you run

- Set `parameters.output_root` in `workload.yaml` to an existing writable Unity Catalog Volume directory.
- Set the catalog, schema, and model defaults in `databricks.yml`. The bundle passes these values to `register_job.py`; its widget defaults apply only to interactive runs. You need `USE CATALOG`, `USE SCHEMA`, and `CREATE MODEL` on the target.
- Confirm that you can create Model Serving endpoints and use `GPU_MEDIUM` workloads.
- The Alpaca dataset is licensed CC BY-NC 4.0. Use data with suitable rights for your application.

Installing the local deployment dependencies requires Python 3.10 or later with the `venv` module. On Debian or Ubuntu, first run `sudo apt install python3-venv`. Then deploy the registration job and install the dependencies into a virtual environment once:

```bash
databricks bundle deploy -t dev --profile <PROFILE>
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-deploy.txt
```

These commands are written for macOS and Linux. On Windows, the virtual environment's interpreter is `.venv\Scripts\python`; use it in place of `.venv/bin/python` here and in stage 3.

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

Registration uses the training run's `merged_model_path` tag, so an `output_root` override during training carries through automatically. For runs without that tag, it uses `<output_root>/<MLFLOW_RUN_ID>/merged` from the current `workload.yaml`.

The job returns the exact registered model version. Each registration run creates a new version, so retrying after registration succeeds can create a duplicate version for the same AIR run.

### 3. Deploy and query the endpoint

Create or update the endpoint with the exact registered version. Choose an endpoint name that no one else uses in the workspace:

```bash
.venv/bin/python deploy_endpoint.py \
  --model-name <CATALOG>.<SCHEMA>.<MODEL> \
  --model-version <VERSION> \
  --endpoint-name <ENDPOINT_NAME> \
  --profile <PROFILE> \
  --wait
```

With `--wait`, the script returns after the endpoint is ready and serves the requested version, which usually takes several minutes for a new endpoint. It waits up to 45 minutes; if it stops while the update is still in progress, rerun the same command to resume waiting. Without `--wait`, the script exits after submitting the change. In that case, run `databricks serving-endpoints get <ENDPOINT_NAME> --profile <PROFILE>` until `state.ready` is `READY` and `state.config_update` is `NOT_UPDATING`, then confirm that `config.served_entities` contains the requested model version. An endpoint can remain ready on its previous version while an update is still in progress.

Send a sample request. It uses the registration smoke-test prompt, whose answer appears as `local_vllm_response` in the result that the registration job printed. Sampling is not fixed, so the wording can differ, but the answer should be about Low-Rank Adaptation and comparable to that one:

```bash
databricks serving-endpoints query <ENDPOINT_NAME> \
  --profile <PROFILE> \
  --json '{"messages":[{"role":"user","content":"In machine learning, explain Low-Rank Adaptation in one sentence."}],"max_tokens":64}'
```

## Results

The AIR MLflow run contains training parameters, loss metrics, the LoRA adapter, and lineage to the registered model. The endpoint serves the exact model version supplied to `deploy_endpoint.py` with scale to zero enabled.
