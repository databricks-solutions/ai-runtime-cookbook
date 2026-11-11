# Hugging Face tokens and Databricks secrets

Complete this setup before the first run of a recipe that declares `secrets.HF_TOKEN`, and repeat the checks when you change workspaces, CLI profiles, run-as identities, or models. A secret belongs to one Databricks workspace; it does not follow your CLI profile to another workspace.

## Create a token with model access

1. Sign in to the Hugging Face account that will own the token. Open each gated model page linked in the recipe, accept its terms, and request access if required. Wait for approval before running the workload. A token alone does not grant gated-model access.
2. Create a [Hugging Face user access token](https://huggingface.co/docs/hub/security-tokens). A read token is sufficient for downloading models. For a fine-grained token, grant read access to the required repositories and permission to read public gated repositories you can access. Use the same account that received model approval; write permission is unnecessary for these recipes.
3. Keep the token available for the interactive secret prompt below. Do not add it to YAML, source files, command-line arguments, or shell history.

## Store the token in the target workspace

Use a Databricks CLI profile authenticated to the workspace where you will submit the recipe. Replace `<profile>` in all commands with that profile name. Confirm the selected host and identity without displaying credentials:

```bash
databricks auth describe --profile <profile>
```

Recipes use the shared secret reference `my_scope/hf_token`. Check whether the scope already exists:

```bash
databricks secrets list-scopes --profile <profile>
```

For a new Databricks-backed scope, create it with the following command. Skip creation if the scope already exists. Ask its owner for access, or use another scope you can manage.

```bash
databricks secrets create-scope my_scope --profile <profile>
```

Store the token using the hidden interactive prompt. This requires `WRITE` or `MANAGE` on the scope. The command replaces an existing `hf_token` value, so coordinate with the scope owner before updating a shared secret.

```bash
databricks secrets put-secret my_scope hf_token --profile <profile>
```

For more details, see [Databricks secret management](https://docs.databricks.com/aws/en/security/secrets/) ([Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/)). Azure Key Vault-backed scopes use their own secret-management procedure.

## Grant the workload identity access

The workload's run-as identity needs at least `READ` on the secret scope. Check the job's run-as setting; it can differ from the CLI user who submits it. A scope owner with `MANAGE` can inspect existing permissions:

```bash
databricks secrets list-acls my_scope --profile <profile>
```

If access is missing, grant `READ` to the run-as user or a group containing it. Replace `<run-as-principal>` with that principal. Retain any existing `WRITE` or `MANAGE` grant; `put-acl` replaces a principal's current permission.

```bash
databricks secrets put-acl my_scope '<run-as-principal>' READ --profile <profile>
```

Secret permissions apply to the entire scope. Choose a scope whose contents can be shared with that identity. See [secret access control](https://docs.databricks.com/aws/en/security/secrets/secret-acl) ([Azure](https://learn.microsoft.com/en-us/azure/databricks/security/secrets/secret-acl)).

## Configure and verify the recipe

Keep the secret reference in `workload.yaml`:

```yaml
secrets:
  HF_TOKEN: my_scope/hf_token
```

AIR resolves the reference and supplies the secret value as the workload's `HF_TOKEN` environment variable. If you use another existing scope or key, change the reference to `<secret-scope>/<secret-key>`.

Verify that `hf_token` appears in the key metadata using the same profile that will submit the workload:

```bash
databricks secrets list-secrets my_scope --profile <profile>
```

This command lists names and timestamps without reading token values. It verifies the CLI identity's ability to list keys, not the token's validity, model approval, or a different run-as identity's permissions. Do not print `HF_TOKEN` or retrieve secret values into terminal output. A local model-access check must use the same token that you stored in Databricks.

After checking the recipe's other prerequisites, submit it from its directory:

```bash
databricks air run -f workload.yaml --watch --profile <profile>
```

## Troubleshoot setup failures

| Failure | What to check |
| --- | --- |
| Scope or key does not exist | Confirm the profile's host, list scopes and keys, and create the missing secret in that workspace or correct `secrets.HF_TOKEN`. |
| Databricks permission denied | Ask the scope owner to verify the submitting identity and workload run-as identity have the required scope permissions. |
| Hugging Face token missing or invalid | Confirm `secrets.HF_TOKEN` is configured, and replace an expired or revoked token using the hidden prompt. |
| Hugging Face gated-model access denied | Open the exact model page, obtain approval for the token owner's account, and check the token's read permissions for that repository. Creating another token does not replace approval. |
| Access check times out or cannot connect | Resolve network access to Hugging Face and repeat the check before submitting a GPU workload. |
