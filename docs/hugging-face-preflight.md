# Check Hugging Face access before submitting a recipe

Run the cookbook preflight helper before your first run in a workspace or with a model. Repeat it after changing the CLI profile, token, Hugging Face account, model, or workload run-as identity. It uses Python's standard library and the Databricks CLI, makes small API requests, and allocates no GPUs.

## Prepare the token and workspace secret

Create a [Hugging Face read token](https://huggingface.co/settings/tokens) for the account that will access the model. A fine-grained token must allow reading each required repository, including access to public gated repositories where applicable. For gated models, approval is separate from token creation: visit each model's Hugging Face page with that account, accept the terms, and wait for any required approval.

Select the workspace profile and confirm its host before creating or checking secrets:

```bash
databricks auth describe --profile "<profile-name>"
```

Recipes use `my_scope/hf_token` by default. If the scope does not exist, create it; then enter your token through the CLI's interactive prompt:

```bash
databricks secrets create-scope my_scope --profile "<profile-name>"
databricks secrets put-secret my_scope hf_token --profile "<profile-name>"
databricks secrets list-secrets my_scope --profile "<profile-name>"
```

`put-secret` replaces an existing key, so coordinate with the scope owner before updating a shared token. Skip `create-scope` for an existing scope. Use an existing scope you can manage if workspace policy prevents creating one. Update `secrets.HF_TOKEN` in `workload.yaml` and pass the same `scope/key` as the helper's `--secret` argument. Secret scopes and keys belong to a workspace; verify them again when switching profiles.

The workload's run-as identity needs scope `READ` access. Existing `WRITE` or `MANAGE` access also permits reads. If access is missing, a scope manager can grant it:

```bash
databricks secrets put-acl my_scope "<run-as-principal>" READ --profile "<profile-name>"
```

Use the actual workload run-as user or service principal. Check existing grants first: `put-acl` replaces that principal's permission, so retain an existing `WRITE` or `MANAGE` grant. Scope permissions cover every secret in the scope. See [Databricks secrets](https://docs.databricks.com/aws/en/security/secrets/) for scope permissions and setup. AIR resolves the `secrets.HF_TOKEN` reference in `workload.yaml` and injects the value into the workload environment.

## Run the checks

From the Llama recipe directory:

```bash
python ../../scripts/hf_preflight.py --profile "<profile-name>" \
  --model meta-llama/Llama-3.1-8B
```

From the audio transcription and diarization recipe directory, check both pyannote models:

```bash
python ../../scripts/hf_preflight.py --profile "<profile-name>" \
  --model pyannote/speaker-diarization-3.1 \
  --model pyannote/segmentation-3.0
```

For an interactive run, the helper prompts for the token without echoing it. Supply the same token stored in the workspace secret. Agents can receive `HF_TOKEN` through their secret environment instead. Do not paste token literals into shell commands, configuration files, or logs, and do not enable debug logging while handling credentials.

The helper checks the token's identity, calls each model's access-check API, and lists Databricks secret metadata to confirm the configured key exists. It never retrieves the stored secret value. A passing result leaves two checks for you: confirm that the stored token matches the supplied token and that the workload run-as identity can read the secret.

Wait until every check passes, then submit with the same profile:

```bash
databricks air run -f workload.yaml --watch --profile "<profile-name>"
```

The helper runs locally before submission; it is not an automatic check inside `air run`. It checks the model IDs supplied with `--model`, so update those arguments when changing a recipe's model or gated dependencies. It does not validate storage permissions, training packages, or GPU availability.

## Resolve a failed check

| Result | Next step |
| --- | --- |
| Token missing or malformed | Run interactively for the hidden prompt, or provide `HF_TOKEN` through your agent's secret environment. |
| Token invalid or expired | Create a read token, update the workspace secret, and rerun with that token. |
| Model access denied | Open the printed model URL using the token owner's account, request or accept access, wait for approval, and check the token's repository permissions. |
| Secret key missing | Store the token in the selected workspace and ensure `--secret` matches `secrets.HF_TOKEN`. |
| Cannot list secret metadata | Check the profile's workspace, scope existence, and the CLI identity's scope permissions. Have a scope manager verify the workload run-as identity separately. |
| Check unavailable or access cannot be established | Check connectivity, Hugging Face service status, model ID, and private-repository permissions as appropriate, then retry. Access remains unverified; retry after resolving the reported cause. |

The Llama recipe requires approval at [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). For an ungated model, use the separate [Qwen2.5 TorchTrainer recipe](../language-models/ray-train-sft-qwen2.5-3b-torchtrainer/) and follow its own configuration. The audio recipe needs terms accepted for both [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
