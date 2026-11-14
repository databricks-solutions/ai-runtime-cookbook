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

- To add nodes, override the GPU count, for example `databricks air run -f workload.yaml --watch --override compute.num_accelerators=4`.
- To run your own script, copy a recipe directory and update `experiment_name`, the recipe directory in `code_source.include_paths`, and `RAY_ENTRYPOINT` together. If the snapshot does not contain the entry point, the bootstrap stops with `RAY_ENTRYPOINT file not found` before it starts Ray.
- To pass arguments to your script, add them after the bootstrap: `RAY_ENTRYPOINT=my-job/main.py bash ray_bootstrap.sh --epochs 3`.
