# Hugging Face secret checks for agents

Audience: coding agents preparing a cookbook recipe. Use the published [Hugging Face token guide](https://huggingface.co/docs/hub/security-tokens), [Databricks secret setup](https://docs.databricks.com/aws/en/security/secrets/) ([Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/)), and [AIR secret references](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/connecting/cli/yaml-config#secrets) for setup procedures.

Before the first run in a workspace, or after changing profiles, models, tokens, or run-as identities:

1. Confirm the CLI profile's workspace and identity with `databricks auth describe --profile <profile>`. Secret scopes and keys belong to that workspace.
2. Check each model listed in the recipe. The token owner's Hugging Face account must have accepted the terms and received any required approval. Read access is sufficient; fine-grained tokens also need access to the required repositories, including public gated repositories where applicable.
3. Check `secrets.HF_TOKEN` in `workload.yaml`. The cookbook default is `my_scope/hf_token`; use another accessible existing reference if needed. AIR injects the referenced value as `HF_TOKEN`.
4. List key metadata with `databricks secrets list-secrets my_scope --profile <profile>` (substitute the configured scope). Missing keys require secret setup in that workspace; permission errors require the scope owner's help.
5. Confirm the workload run-as identity has scope `READ` access. Preserve existing `WRITE` or `MANAGE` grants when adjusting permissions; grants cover every secret in the scope.
6. Follow the [preflight guide](hugging-face-preflight.md) to validate the token and every required model before submitting the recipe. Supply the same token stored in Databricks. Metadata checks establish key existence and the CLI identity's ability to list keys; they do not validate the token or another run-as identity's access.

Keep credentials out of source files, YAML, command arguments, shell history, and output. Have the user enter tokens through the hidden `databricks secrets put-secret` prompt; coordinate before replacing a shared key. A local model-access check must use the same token stored in the workspace secret.

For invalid credentials, update the token and secret. For gated-model denial, obtain approval at the exact model URL using the token owner's account and check repository permissions. If a check cannot connect, resolve connectivity and retry before allocating GPUs.
