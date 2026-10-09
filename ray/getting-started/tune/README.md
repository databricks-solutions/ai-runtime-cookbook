# Ray Tune hello world

This recipe starts a four-GPU Ray cluster and uses Ray Tune with ASHA to search eight configurations for a small PyTorch model. It prints the best configuration and loss.

## Run

```bash
databricks air run -f workload.yaml
```
