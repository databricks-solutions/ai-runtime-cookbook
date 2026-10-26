# Databricks AI Runtime Cookbook

The Databricks AI Runtime Cookbook provides runnable examples for building and running machine learning workloads with Databricks AI Runtime.

Each recipe focuses on a specific use case and includes an AIR workload configuration together with the source code needed to run it. Use these recipes as starting points that you can inspect, run, and adapt for your own workloads.

## Repository organization

Recipes are organized by workload directly under the repository root:

```text
<workload>/<example>/
```

Ray is the exception: its recipes are organized by Ray library under `ray/`, with introductory recipes in `ray/getting-started/`:

```text
ray/<library>/<example>/
```

## Run a recipe

Each recipe is designed to run from its own directory using the same command.

Before running a recipe, set up the Databricks CLI for a workspace with access to AI Runtime:

- Configure the Databricks CLI with AI Runtime ([AWS](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/ai-runtime/cli/)).
- Authenticate to your workspace with `databricks auth login` ([AWS](https://docs.databricks.com/aws/en/dev-tools/cli/authentication) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/authentication)).

1. `cd` into the recipe directory.
1. Review `workload.yaml` for values that you need to configure.
1. Run the workload with `databricks air run -f workload.yaml`.

For example:

```bash
cd getting-started/air-cli
databricks air run -f workload.yaml
```

To build and run an AIR workload from a custom container, see the [custom Docker image recipe](getting-started/custom-docker-image/README.md).

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidance on adding and validating recipes.
