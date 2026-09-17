# Databricks AI Runtime Cookbook

The Databricks AI Runtime Cookbook provides runnable examples for building and running machine learning workloads with Databricks AI Runtime.

Each recipe focuses on a specific use case and includes an AIR workload configuration together with the source code needed to run it. Use these recipes as starting points that you can inspect, run, and adapt for your own workloads.

## Repository organization

Recipes are organized by workload directly under the repository root:

```text
<workload>/<example>/
```

Ray is the exception, with recipes organized by workload under `ray/`:

```text
ray/<workload>/<example>/
```

## Run a recipe

Each recipe is designed to run from its own directory using the same command.

1. `cd` into the recipe directory.
1. Review `workload.yaml` for values that you need to configure.
1. Run the workload with `databricks air run -f workload.yaml`.

For example:

```bash
cd getting-started/air-cli
databricks air run -f workload.yaml
```

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidance on adding and validating recipes.
