# Contributing

Thank you for contributing to the Databricks AI Runtime Cookbook. This guide defines the standards for cookbook recipes and explains how to submit changes. Instructions for automated coding agents live in `AGENTS.md`.

For instructions on running recipes, see the [README](README.md).

## What belongs in the cookbook

Each recipe should demonstrate one focused AI Runtime use case through the CLI. Keep the code minimal and make the recipe self-contained.

## Recipe standards

### Layout and naming

Place each recipe in a leaf directory under the appropriate workload:

```text
<workload>/<example>/
```

Ray recipes use:

```text
ray/<workload>/<example>/
```

Naming and reuse:

- Reuse an existing workload directory when it fits.
- Use kebab-case for directories and snake_case for Python files.
- Give the leaf recipe directory a descriptive, unambiguous name within its workload path.

Each recipe must include:

- `workload.yaml`
- A primary entry point (typically Python; use the appropriate executable entry point for the recipe)
- A short `README.md`

Keep recipes independent:

- Do not depend on files from another recipe.
- Set `code_source.snapshot.root_path` to the recipe directory, normally `.`.
- Commands must execute against the recipe snapshot and must not depend on files outside the recipe. Reference bundled files through `$CODE_SOURCE_PATH`, or `cd "$CODE_SOURCE_PATH"` and then use relative paths.
- Use absolute paths for external locations such as Unity Catalog volumes.

### Configuration and public safety

Put values that users must configure in `workload.yaml`. Choose where each value goes:

- **`env_variables`**: settings the application or framework consumes as environment variables, including settings users are expected to change. Values must be strings, so quote anything that looks like a number or boolean.
- **`parameters`**: structured recipe configuration that the source reads from the YAML file exposed through `HYPERPARAMETERS_PATH`.
- **`secrets`**: any sensitive value; use these rather than `env_variables` or `parameters`.

When either `env_variables` or `parameters` is reasonable, prefer the interface natural to the underlying tool.

When a recipe requires a Hugging Face token, use the shared cookbook secret reference `my_scope/hf_token`. This secret must already exist and contain a token with the required model access. Users who cannot use this shared default may replace it with another existing accessible secret reference.

Never include the following. For values that have no established cookbook default, use clear placeholders such as `<secret-scope>/<secret-key>`.

- Credentials or personal secret scope names
- Personal or workspace-specific catalogs or schemas
- Private endpoints
- Internal workspace names or identifiers
- Databricks job-run URLs

### Reproducibility and correctness

- Packages inherited from the AIR environment do not need to be pinned. Pin any dependency you add above the AIR environment to an exact, validated version.
- Prefer stable, versioned model and dataset identifiers (for example, `Qwen/Qwen2.5-7B-Instruct` or `tatsu-lab/alpaca`). Pin an immutable revision only when exact reproducibility matters, such as benchmarked results, regression or output assertions, or behavior that depends on a particular upstream snapshot. Avoid floating aliases such as `latest`. When a recipe reports a specific measured metric or comparison, record the revision in the README or a comment as optional provenance rather than in executable code.
- Use evaluation data appropriate to the methodology, and do not tune or select a model on final test data.
- Make distributed downloads, logging, checkpoints, and other side effects rank-safe.
- Set `max_retries` explicitly. When retries are enabled, ensure reruns do not corrupt outputs. For long-running or costly training, save checkpoints and restore them on restart when practical.

### MLflow and model lifecycle

AIR creates an MLflow run for each attempt and provides its ID in `MLFLOW_RUN_ID`. When a recipe adds custom MLflow tracking, log to that run.

- Training recipes generally log metrics and, when useful, a packaged MLflow model.
- Treat registration in Unity Catalog and alias assignment as model lifecycle operations. Include them when they support the recipe's learning objective, and document the registry target and mutations clearly.

### Data and durable storage

For recipe-owned durable outputs and checkpoints, use `/Volumes/main/default/air_examples/<recipe-name>`, where `<recipe-name>` is the leaf recipe directory. This shared volume must already exist and be writable. Contributors must not create catalogs, schemas, or volumes. Users who cannot use this shared location may replace the path in `workload.yaml` with another existing writable volume.

- Create files and subdirectories only below the configured path, never catalogs, schemas, or volumes.
- Pre-existing input data may live in another volume or catalog; document that location as user-configurable rather than copying it into the shared cookbook volume.
- Use `MLFLOW_RUN_ID` for attempt-specific output directories and a separate stable identifier for checkpoints shared across retries.

When a recipe uses this shared location, note it near the setting in `workload.yaml`, for example:

```yaml
# Requires access to this existing volume; replace it with another writable volume if needed.
output_root: /Volumes/main/default/air_examples/<recipe-name>
```

### Documentation

Start each recipe README with one or two sentences stating what the recipe demonstrates. Include fixed details such as the model, dataset, training duration, compute shape, or distributed strategy when they help readers understand the workload's behavior or cost.

Every README must include the `Run` section. The other sections are optional; omit any that add no useful information:

- **`Before you run`**: include only when the reader must complete recipe-specific setup or configuration.
- **`Results`**: include only when the recipe produces persistent artifacts or metrics that the reader needs to locate.

Use this structure:

````markdown
# <Recipe name>

<One or two sentences stating what the recipe demonstrates.>

## Before you run

<Required setup, configuration, permissions, licenses, secrets, or storage. Omit this section when none are required.>

## Run

```bash
databricks air run -f workload.yaml
```

## Results

<Summarize the persistent artifacts or metrics the recipe produces and where to find them. Omit this section when the introduction already describes the observable result.>
````

A few things to avoid:

- Put fixed workload behavior in the introduction rather than `Results`.
- Do not describe implementation steps as results.
- Do not repeat repository-wide requirements such as access to AI Runtime.

## Contribution workflow

### Validate the recipe

For every new recipe or behavior-changing update, run the following command from the recipe directory:

```bash
databricks air run -f workload.yaml --watch
```

Then:

- Confirm that the workload completes successfully and produces the documented result.
- Report the observed workload duration and relevant result in the pull request.
- If you cannot complete this validation before opening the pull request, state that explicitly and explain what remains unvalidated.

The recipe must pass end-to-end validation before it is merged.

### Open a pull request

- Open a pull request to `main`. If you do not have write access, submit it from a fork.
- Keep pull requests cohesive and narrowly scoped.
- Explain what the recipe demonstrates and why it belongs in the cookbook.
- State what you validated: whether the AIR workload completed successfully, its observed duration, and anything you could not validate.
- Make sure the recipe README documents any required setup and permissions.
