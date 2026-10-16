"""Smoke-test merged weights and register them for Custom LLM Serving."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path

import mlflow
import ninja
import requests
from mlflow import MlflowClient
from mlflow.pyfunc.model import ChatCompletionResponse, ChatModel
from packaging.version import Version

SMOKE_PROMPT = "In machine learning, explain Low-Rank Adaptation in one sentence."


@dataclass
class RegistrationResult:
    model_name: str
    model_version: str
    local_response: str


class EntrypointModel(ChatModel):
    """Placeholder model; Custom LLM Serving starts the metadata entrypoint."""

    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})


def vllm_command(port: int, model_path: str, served_model_name: str) -> list[str]:
    return [
        "env",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        "python",
        "-u",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.7",
    ]


def wait_for_vllm(server: subprocess.Popen, log_path: Path, port: int) -> None:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if server.poll() is not None:
            details = log_path.read_text(errors="replace")[-12000:]
            raise RuntimeError(f"vLLM exited before becoming healthy:\n{details}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    details = log_path.read_text(errors="replace")[-12000:]
    raise RuntimeError(f"vLLM did not become healthy:\n{details}")


def validate_runtime() -> None:
    if not os.environ.get("DATABRICKS_ACCELERATOR"):
        raise RuntimeError("Custom LLM model packaging requires an AI Runtime GPU task.")
    if Version(mlflow.__version__) < Version("3.12"):
        raise RuntimeError("Custom LLM Serving requires MLflow 3.12 or newer.")
    if Version(package_version("databricks-sdk")) < Version("0.102.0"):
        raise RuntimeError("Custom LLM Serving requires databricks-sdk 0.102.0 or newer.")
    os.environ["PATH"] = str(ninja.BIN_DIR) + os.pathsep + os.environ["PATH"]
    if shutil.which("ninja") is None:
        raise RuntimeError("The ninja executable is required by vLLM's FlashInfer JIT.")


def smoke_test_vllm(work_dir: Path, artifact_name: str, served_model_name: str) -> str:
    port = 3080
    log_path = work_dir / "vllm.log"
    with log_path.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            vllm_command(
                port,
                str((work_dir / artifact_name).resolve()),
                served_model_name,
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_for_vllm(server, log_path, port)
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": SMOKE_PROMPT}],
                    "max_tokens": 64,
                },
                timeout=60,
            )
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"Local vLLM response: {answer}", flush=True)
            return answer
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def package_and_register(
    artifact_name: str,
    served_model_name: str,
    registered_model_name: str,
    training_run_id: str,
    training_experiment_id: str,
    source: Path,
) -> tuple[str, str]:
    entrypoint = " ".join(vllm_command(8080, artifact_name, served_model_name))
    with mlflow.start_run(
        experiment_id=training_experiment_id,
        run_name="package-and-register-qwen-sft",
        tags={
            "recipe": "sft-qwen2.5-0.5b-lora-serving",
            "source_training_run_id": training_run_id,
            "source_merged_model_path": str(source),
        },
    ) as packaging_run:
        model_info = mlflow.pyfunc.log_model(
            name=served_model_name,
            python_model=EntrypointModel(),
            artifacts={"model_dir": artifact_name},
            metadata={
                "task": "llm/v1/chat",
                "entrypoint": entrypoint,
                "source_training_run_id": training_run_id,
            },
            extra_pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                "ninja==1.11.1.4",
            ],
        )
        version = mlflow.register_model(
            model_info.model_uri,
            registered_model_name,
            env_pack="databricks_model_serving",
        )
        mlflow.set_tags(
            {
                "registered_model": registered_model_name,
                "registered_model_version": str(version.version),
            }
        )
        return str(version.version), packaging_run.info.run_id


def register_model(
    output_root: str,
    training_run_id: str,
    registered_model_name: str,
) -> RegistrationResult:
    validate_runtime()
    os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    training_run = client.get_run(training_run_id)

    source = Path(output_root) / training_run_id / "merged"
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found at {source}")

    artifact_name = "qwen_sft"
    served_model_name = registered_model_name.rsplit(".", 1)[-1]
    with tempfile.TemporaryDirectory(prefix="qwen-sft-registration-") as temp_dir:
        work_dir = Path(temp_dir)
        shutil.copytree(source, work_dir / artifact_name)
        original_dir = Path.cwd()
        os.chdir(work_dir)
        try:
            local_response = smoke_test_vllm(work_dir, artifact_name, served_model_name)
            version, packaging_run_id = package_and_register(
                artifact_name=artifact_name,
                served_model_name=served_model_name,
                registered_model_name=registered_model_name,
                training_run_id=training_run_id,
                training_experiment_id=training_run.info.experiment_id,
                source=source,
            )
        finally:
            os.chdir(original_dir)

    training_tags = {
        "registered_model": registered_model_name,
        "registered_model_version": version,
        "packaging_run_id": packaging_run_id,
    }
    for key, value in training_tags.items():
        client.set_tag(training_run_id, key, value)
    return RegistrationResult(
        model_name=registered_model_name,
        model_version=version,
        local_response=local_response,
    )
