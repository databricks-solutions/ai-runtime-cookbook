# Ray getting started

These recipes share `ray_bootstrap.sh`, which starts and coordinates a multi-node Ray cluster on AI Runtime. Each workload snapshots this directory with `root_path: ..` and uses `include_paths` to package only the shared bootstrap and its recipe directory.

Run each recipe from its leaf directory with the standard command:

```bash
databricks air run -f workload.yaml
```
