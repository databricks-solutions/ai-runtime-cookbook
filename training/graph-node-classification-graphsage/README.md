# Graph node classification with GraphSAGE

Train a two-layer GraphSAGE model to sort research papers into 40 topics, for 50 epochs on one A10 GPU. The model uses each paper's own features plus the papers linked to it by citations.

The example uses the [ogbn-arxiv citation graph](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv):

- Each node is a paper with 128 features derived from its title and abstract. The recipe doesn't process raw text.
- Each edge is a citation. The recipe treats citations as undirected.
- Papers are split by publication year into training, validation, and test sets.

This is supervised classification. The model learns from papers with known topics and predicts topics for newer papers. All features and citations are visible during training, but only training labels count toward the loss. Validation accuracy picks the best checkpoint. Test labels are used once at the end.

## Before you run

The dataset downloads automatically and is licensed under ODC-BY. Review its usage terms on the linked page. To write outputs elsewhere, set `parameters.output_root` to an existing Unity Catalog volume you can write to.

## Run

```bash
databricks air run -f workload.yaml
```

## Results

The AIR MLflow run contains loss and accuracy curves, test accuracy, and macro F1. These files are written to `<output_root>/<MLFLOW_RUN_ID>/` and logged to the run:

- `best_model.pt`: model weights (a state dictionary)
- `model_config.json`: model dimensions
- `class_labels.json`: topic ID to arXiv category
- `metrics.json`: final metrics
- `test_predictions.csv`: paper IDs and predicted topics

To reuse the model, build `GraphSAGE` from `train.py` with the saved dimensions and load the weights with `weights_only=True`. Predictions need the graph as well as the features. Pass node features and an adjacency matrix built with `mean_adjacency`.
