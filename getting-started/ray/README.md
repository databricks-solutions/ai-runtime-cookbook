# Ray getting started

These recipes introduce each Ray library on AI Runtime. For Ray recipes in other domains, see the [Ray recipe index](../../ray/).

| Recipe | Demonstrates | Compute |
| --- | --- | --- |
| [Ray Core](core/) | One Ray task per GPU across the cluster | 2 nodes, 1 A10 each |
| [Ray Data](data/) | A distributed Ray Data pipeline over synthetic data | 2 nodes, 1 A10 each |
| [Ray Train](train/) | Distributed PyTorch training with `TorchTrainer` | 2 nodes, 1 A10 each |
| [Ray Tune](tune/) | An ASHA search over eight hyperparameter configurations | 4 nodes, 1 A10 each |

Run each recipe from its leaf directory with the standard command:

```bash
databricks air run -f workload.yaml --watch
```

The recipes share `ray_bootstrap.sh`, which starts a Ray cluster across the workload's nodes and runs `RAY_ENTRYPOINT` on the head node. Each workload snapshots this directory with `root_path: ..` and uses `include_paths` to package only the shared bootstrap and its recipe directory.

## Adapt a recipe

- To add nodes, override the GPU count: `databricks air run -f workload.yaml --watch --override compute.num_accelerators=<N>`, where `<N>` is the total number of GPUs.
- To run your own script, copy a recipe directory inside `getting-started/ray/`. In the copy's `workload.yaml`, update three values together: `experiment_name`, the directory in `code_source.include_paths`, and `RAY_ENTRYPOINT` in `command`. If the snapshot doesn't contain the entry point, the bootstrap stops with `RAY_ENTRYPOINT file not found` before Ray starts.
- To run a script outside this directory, follow the Ray recipes in other domains:
  1. Put `workload.yaml`, your script, and a copy of `ray_bootstrap.sh` in one directory.
  2. Set `code_source.root_path` to `.` and remove `include_paths`.
  3. Set `RAY_ENTRYPOINT` to the script's file name.
- To run a package, set `RAY_ENTRYPOINT` to a directory that contains `__main__.py`. The bootstrap runs it like `python <dir>`, as a script, so `__main__.py` can't use relative imports.
- To pass arguments to your script, add them after the bootstrap in `command`, for example `RAY_ENTRYPOINT=my-job/main.py bash ray_bootstrap.sh --epochs 3`. Pass credentials through `secrets` rather than arguments.
