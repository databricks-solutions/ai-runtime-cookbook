# Contributing

Thank you for contributing to the Databricks AI Runtime Cookbook. For instructions on running recipes, see the [README](README.md).

## Add a recipe

Each recipe should demonstrate a focused use case and be runnable from its own directory with:

```bash
databricks air run -f workload.yaml
```

Keep the files required to run the recipe together in its leaf directory.

## Choose a location

Organize recipes by workload directly under the repository root:

```text
<workload>/<example>/
```

For example:

```text
getting-started/air-cli/
training/sft-llama-3.1-8b-fsdp-multinode/
```

Ray is the exception. Organize Ray recipes by workload within the `ray/` directory:

```text
ray/<workload>/<example>/
```

Reuse an existing workload directory when it describes the recipe.

## Name directories and files

- Use kebab-case for directory names.
- Give each example a name that distinguishes it from other recipes in the same workload.
- Include details such as the model, method, or topology when they help identify the example.
- Name the AIR workload configuration `workload.yaml`.
- Use snake case for Python filenames. For example, use `train.py` or `batch_inference.py`.

## Configure the recipe

- Use paths relative to the recipe directory.
- Pin dependencies to versions that have been validated.
- Make required user configuration, such as secrets or output locations, clear in `workload.yaml`.

## Validate the recipe

Before opening a pull request:

- Check the Python and YAML syntax.
- Run `databricks air run -f workload.yaml` from the recipe directory and confirm that the workload completes successfully.

## Open a pull request

Open a pull request to `main`. If you do not have write access, submit it from a fork.

Include:

- What the pull request adds or changes and why.
- How you validated the change.
