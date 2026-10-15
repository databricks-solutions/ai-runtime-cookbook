"""Smoke-test merged weights with vLLM and register Custom LLM Serving model."""

import os
import shutil
import subprocess
import tempfile
import time
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow
import ninja
import requests
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel
from packaging.version import Version

SMOKE_PROMPT = (
    "In machine learning, explain Low-Rank Adaptation (LoRA) for fine-tuning "
    "a large language model in one sentence."
)


class EntryPointModel(ChatModel):
    # Custom LLM serving starts vLLM from metadata, rather than calling predict.
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


def vllm_command(port: int, model_name: str, artifact_name: str) -> list[str]:
    return [
        "env", "VLLM_USE_FLASHINFER_SAMPLER=0",
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", artifact_name,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", "float16",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.7",
    ]


def register_model(
    output_root: str, training_run_id: str, catalog: str, schema: str, model_name: str
) -> str:
    """Package one successful training run and return its UC model version."""
    if not os.environ.get("DATABRICKS_ACCELERATOR"):
        raise RuntimeError("Custom LLM model logging requires AI Runtime serverless GPU.")
    if Version(mlflow.__version__) < Version("3.12"):
        raise RuntimeError("Custom LLM serving requires MLflow 3.12 or newer.")
    if Version(package_version("databricks-sdk")) < Version("0.102.0"):
        raise RuntimeError("Custom LLM serving requires databricks-sdk 0.102.0 or newer.")
    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    if shutil.which("ninja") is None:
        raise RuntimeError("The ninja executable is required by vLLM's FlashInfer JIT.")

    source = Path(output_root) / training_run_id / "merged"
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found at {source}")

    artifact_name = "qwen_sft"
    registered_name = f"{catalog}.{schema}.{model_name}"
    workdir = Path(tempfile.mkdtemp(prefix="qwen-sft-serving-"))
    shutil.copytree(source, workdir / artifact_name)
    original_dir = Path.cwd()
    os.chdir(workdir)
    try:
        log_path = workdir / "vllm.log"
        with log_path.open("w") as log:
            server = subprocess.Popen(
                vllm_command(3080, model_name, artifact_name),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited early:\n{log_path.read_text(errors='replace')[-12000:]}")
                    try:
                        if requests.get("http://127.0.0.1:3080/health", timeout=2).ok:
                            break
                    except requests.RequestException:
                        pass
                    time.sleep(3)
                else:
                    raise RuntimeError(f"vLLM did not become healthy:\n{log_path.read_text(errors='replace')[-12000:]}")

                response = requests.post(
                    "http://127.0.0.1:3080/invocations",
                    json={"messages": [{"role": "user", "content": SMOKE_PROMPT}], "max_tokens": 64},
                    timeout=60,
                )
                response.raise_for_status()
                smoke_answer = response.json()["choices"][0]["message"]["content"]
                print(f"Local vLLM response: {smoke_answer}", flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

        mlflow.set_registry_uri("databricks-uc")
        training_run = mlflow.get_run(training_run_id)
        with mlflow.start_run(
            experiment_id=training_run.info.experiment_id,
            run_name="package-and-register-qwen-sft",
            tags={
                "recipe": "sft-qwen2.5-0.5b-lora-serving",
                "source_training_run_id": training_run_id,
                "source_merged_model_path": str(source),
            },
        ) as packaging_run:
            model_info = mlflow.pyfunc.log_model(
                name=model_name,
                python_model=EntryPointModel(),
                artifacts={"model_dir": artifact_name},
                metadata={
                    "task": "llm/v1/chat",
                    "entrypoint": " ".join(vllm_command(8080, model_name, artifact_name)),
                    "source_training_run_id": training_run_id,
                },
                extra_pip_requirements=[f"mlflow=={mlflow.__version__}", "ninja==1.11.1.4"],
            )
            version = mlflow.register_model(
                model_info.model_uri,
                registered_name,
                env_pack="databricks_model_serving",
            )
            mlflow.set_tags({
                "registered_model": registered_name,
                "registered_model_version": str(version.version),
            })

        with mlflow.start_run(run_id=training_run_id):
            mlflow.set_tags({
                "registered_model": registered_name,
                "registered_model_version": str(version.version),
                "packaging_run_id": packaging_run.info.run_id,
            })
        print(f"Registered {registered_name} version {version.version}", flush=True)
        return str(version.version)
    finally:
        os.chdir(original_dir)
