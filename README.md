# Databricks AI Runtime Cookbook

The Databricks AI Runtime Cookbook provides runnable examples for building and running machine learning workloads with Databricks AI Runtime.

Each recipe focuses on a specific use case and includes an AIR workload configuration together with the source code needed to run it. Use these recipes as starting points that you can inspect, run, and adapt for your own workloads.

## Repository organization

Recipes are grouped into top-level directories by domain.

There are two exceptions:

- [Ray](ray/) is an index of Ray recipes across the repository. Each recipe remains in its domain directory and uses a `ray-core-`, `ray-data-`, `ray-train-`, or `ray-tune-` prefix.
- [Serving](serving/) contains end-to-end examples that train, register, and deploy a model. Other domain directories generally contain examples for a single training or inference workload.

## Run a recipe

Each recipe is designed to run from its own directory using the same command.

Before running a recipe, set up the Databricks CLI for a workspace with access to AI Runtime:

- Configure the Databricks CLI with AI Runtime ([AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/)).
- Authenticate to your workspace with `databricks auth login` ([AWS](https://docs.databricks.com/aws/en/dev-tools/cli/authentication) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/authentication)).

### Select a workspace profile

Select an existing workspace profile for the current shell, replacing `<profile-name>` with its name:

```bash
databricks auth profiles
export DATABRICKS_CONFIG_PROFILE="<profile-name>"
databricks auth describe
```

Check the host in `auth describe` before submitting a workload. Keep this variable set so submit, status, and log commands use the same workspace. Without a selected profile, the CLI uses the `DEFAULT` profile. See [configuration profiles](https://docs.databricks.com/aws/en/dev-tools/cli/profiles) for profile setup.

To override this selection, pass `--profile <profile-name>`. Use the same override when monitoring that run:

```bash
databricks auth describe --profile <profile-name>
databricks air run -f workload.yaml --watch --profile <profile-name>
```

### Submit and monitor

1. `cd` into the recipe directory.
1. Review `workload.yaml` for values that you need to configure.
1. Run the workload with `databricks air run -f workload.yaml --watch`.

For example:

```bash
cd getting-started/air-cli
databricks air run -f workload.yaml --watch
```

`--watch` streams logs until the run completes. To submit in the background, omit `--watch`, save the job run ID printed by `air run`, and use it to inspect the run or follow its logs:

```bash
databricks air run -f workload.yaml
RUN_ID="<job-run-id>"
databricks air get "$RUN_ID"
databricks air logs "$RUN_ID"
```

These commands use the selected profile. If you submitted with `--profile`, include the same flag on both monitoring commands. `air logs` follows an active run until completion and fetches logs for a completed run. Both monitoring commands also work after a watched command exits.

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidance on adding and validating recipes.
