# ResNet-50 fine-tuning on Imagenette

This recipe uses Ray Train to fine-tune a pretrained ResNet-50 on Imagenette for five epochs. Ray launches one DDP worker per A10 GPU across two nodes and reports training and validation metrics to the AIR MLflow run.

## Run

```bash
databricks air run -f workload.yaml
```
