# Graph node classification with GraphSAGE

Train a two-layer GraphSAGE model on the public [ogbn-arxiv citation graph](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) for 50 epochs on one A10 GPU. This is graph node classification: each paper has 128 supplied text-derived features, and the model combines those features with citation neighbors to predict one of 40 topics.

The official paper-year split supplies training, validation, and test nodes. This is a **transductive** benchmark: features and citation edges for all nodes are visible during training, but only training labels contribute to the loss. Validation accuracy selects the checkpoint; test labels are used once afterward. The recipe uses full-graph mean aggregation implemented with PyTorch sparse matrices, treating citations as undirected; it does not tokenize papers or train a text encoder.

## Before you run

The dataset downloads automatically and is distributed under ODC-BY. Review its linked usage terms. Set `parameters.output_root` to another existing writable Unity Catalog volume directory if needed.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains loss/accuracy curves, final test accuracy and macro F1, and copies of the artifacts in `<output_root>/<MLFLOW_RUN_ID>/`: `best_model.pt`, `model_config.json`, `class_labels.json`, `metrics.json`, and `test_predictions.csv` with paper IDs and predicted topics.

The checkpoint is a state dictionary. To reuse it, construct `GraphSAGE` from `train.py` with the saved dimensions, load the weights with `weights_only=True`, and pass node features and an adjacency matrix produced by `mean_adjacency`. Predictions require graph context as well as features; this is not a standalone text-classification model.
