# Ray Core hello world

This recipe starts a Ray cluster on two A10 GPUs and schedules one Ray Core task per GPU. Each task prints its node rank, Ray GPU ID, visible CUDA device, and GPU model.

## Run

```bash
databricks air run -f workload.yaml
```
