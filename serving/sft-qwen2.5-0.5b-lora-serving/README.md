# Fine-tune and serve Qwen2.5-0.5B with AI Runtime

This sample shows how to fine-tune and deploy a small language model on Databricks. We fine-tune `Qwen/Qwen2.5-0.5B-Instruct` on one A10 GPU with [AI Runtime](https://docs.databricks.com/machine-learning/ai-runtime/), track it with [MLflow](https://docs.databricks.com/mlflow/), register it in [Unity Catalog](https://docs.databricks.com/machine-learning/manage-model-lifecycle/), and deploy it with vLLM on a [Custom LLM Serving endpoint](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms). The 64 Alpaca training examples and 16 held-out examples demonstrate the workflow. For a more realistic use case, use task-specific data and evaluate the fine-tuned model against the base model on representative held-out prompts.

## Before you run

- Use an AWS workspace with AI Runtime serverless GPU and Custom LLM Serving. Model logging requires AI v6, MLflow 3.12 or newer, and `databricks-sdk` 0.102.0 or newer.
- Set the catalog, schema, model, and endpoint names in `databricks.yml`. You need permission to register Unity Catalog models and create serving endpoints.
- Make sure `/Volumes/main/default/air_examples/sft-qwen2.5-0.5b-lora-serving` is writable, or change `output_root` in `workload.yaml` to another existing writable [Unity Catalog volume](https://docs.databricks.com/volumes/).
- The Alpaca dataset is licensed **CC BY-NC 4.0**. Use data with suitable rights for other purposes.

## Files

`resources/qwen_sft.job.yml` defines one Lakeflow Job with two notebook tasks. Each task points to a Python notebook file; the training notebook calls the shared training and registration code:

- **`train` (AI v6 GPU):** Runs `train_job.py`. It reads training parameters from `workload.yaml`, calls `train.py` to fine-tune, then calls `register.py` to test with vLLM and register the model. It passes the registered version to the next task.
- **`deploy` (CPU):** Runs `deploy.py` after `train` succeeds. It uses that model version to create or update the `GPU_MEDIUM` Model Serving endpoint and sends a sample request. The endpoint provisions its own GPU.

| File | Used by | Purpose |
| --- | --- | --- |
| `databricks.yml` | [Declarative Automation Bundles](https://docs.databricks.com/dev-tools/bundles/) | Sets model and endpoint names and includes the job definition. |
| `resources/qwen_sft.job.yml` | [Lakeflow Jobs](https://docs.databricks.com/jobs/) | Sets task order, environments, and notebook paths. |
| `workload.yaml` | [AI Runtime CLI](https://docs.databricks.com/machine-learning/ai-runtime/cli/); Lakeflow training task | Its `command` runs `train.py` for a direct AIR run. `train_job.py` reads only its `parameters`; the job sets compute and environment separately. |
| `train.py` | Direct AIR run; called by `train_job.py` | Fine-tunes with LoRA, compares held-out loss, logs to MLflow, and saves weights. |
| `train_job.py` | Lakeflow GPU notebook task | Reads training parameters, calls `train.py` and `register.py`, then passes the model version to deployment. |
| `register.py` | Called by `train_job.py` on GPU | Tests merged weights with vLLM, packages them with MLflow, and registers the model in Unity Catalog. |
| `deploy.py` | Lakeflow CPU notebook task | Uses the Databricks SDK to create or update the serving endpoint and send a sample request. |

## Run

From this directory, replace `<PROFILE>` with a configured [Databricks CLI profile](https://docs.databricks.com/dev-tools/cli/) to run training, registration, and deployment as one Lakeflow Job:

```bash
databricks bundle validate --strict -t dev --profile <PROFILE>
databricks bundle deploy -t dev --profile <PROFILE>
databricks bundle run qwen_sft -t dev --profile <PROFILE>
```

`databricks bundle deploy` uploads these files and creates or updates the job. `databricks bundle run` starts the two tasks in the order shown above.

To run training only with the AI Runtime CLI:

```bash
databricks air run -f workload.yaml --dry-run --profile <PROFILE>
databricks air run -f workload.yaml --watch --profile <PROFILE>
```

This uses the `command` in `workload.yaml` to run `train.py` and save merged weights. Use the Lakeflow Job to register and deploy the model.

## Results

The training MLflow run records parameters, loss metrics, a LoRA adapter, and a result summary. Merged weights are at `<output_root>/<MLFLOW_RUN_ID>/merged`. The job creates a linked packaging run, registers `<catalog>.<schema>.<model_name>`, and deploys its new version to `<endpoint_name>` with scale to zero.

Base and fine-tuned loss use the same 16 held-out examples. These losses and the sample requests check that the workflow runs; they do not establish factual accuracy or a general quality gain.

The full Alpaca dataset has about 52,000 examples. To use more, adjust the example counts, `max_steps`, `max_length`, batch size, and gradient accumulation in `workload.yaml`. The eight-minute cap applies to optimization only.
