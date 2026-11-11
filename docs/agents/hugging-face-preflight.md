# Hugging Face preflight for agents

Audience: coding agents preparing a cookbook recipe. For token creation and secret setup, use the published [Hugging Face token guide](https://huggingface.co/docs/hub/security-tokens), [Databricks secrets](https://docs.databricks.com/aws/en/security/secrets/) ([Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/)), and [AIR secret reference](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/yaml-config#secrets).

Run the helper before the first submission in a workspace or with a model. Repeat after changing the profile, token, Hugging Face account, model, or run-as identity. It uses Python's standard library and the Databricks CLI, makes small API requests, and allocates no GPUs.

## Check the configured models

Confirm the selected workspace with `databricks auth describe --profile "<profile-name>"`. Ensure the workload run-as identity has at least `READ` on the configured secret scope.

From the Llama recipe directory:

```bash
python ../../scripts/hf_preflight.py --profile "<profile-name>" \
  --model meta-llama/Llama-3.1-8B
```

From the audio transcription and diarization recipe directory:

```bash
python ../../scripts/hf_preflight.py --profile "<profile-name>" \
  --model pyannote/speaker-diarization-3.1 \
  --model pyannote/segmentation-3.0
```

Pass `--secret <scope>/<key>` if `secrets.HF_TOKEN` differs from `my_scope/hf_token`. Supply every required model with `--model`, including gated dependencies. The helper uses an interactive hidden token prompt; agents can receive `HF_TOKEN` through their secret environment. Supply the same token stored in Databricks, and keep it out of command arguments, configuration files, logs, and debug output.

The helper validates token identity, model access, and secret-key metadata. It never retrieves the stored value. Separately confirm that the stored token matches the checked token and that the workload run-as identity can read it. Storage permissions, packages, and GPU availability need their own checks.

Proceed only after every check passes; use the same profile for submission:

```bash
databricks air run -f workload.yaml --watch --profile "<profile-name>"
```

## Recover from failures

Follow the helper's error message. Missing or invalid credentials require token/secret setup; model denial requires approval and repository permissions for the token owner's account. Network failures, missing models, and unavailable checks leave access unverified: resolve the reported cause and retry before submission.

Approval is required for [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and both pyannote models: [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0). For an ungated Llama alternative, use the separate [Qwen2.5 TorchTrainer recipe](../../language-models/ray-train-sft-qwen2.5-3b-torchtrainer/) and its configuration.
