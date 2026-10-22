#!/usr/bin/env python3
"""Classify paper topics using text-derived features and a citation graph."""

import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score


def load_data(p):
    root = Path(p["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "arxiv.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(p["dataset_url"], timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        def read_csv(name, header=None):
            with source.open(f"arxiv/{name}.csv.gz") as handle:
                return pd.read_csv(handle, compression="gzip", header=header)

        features = torch.from_numpy(read_csv("raw/node-feat").to_numpy(dtype=np.float32))
        labels = torch.from_numpy(read_csv("raw/node-label").to_numpy(dtype=np.int64, copy=True).ravel())
        edges = torch.from_numpy(read_csv("raw/edge").to_numpy(dtype=np.int64).T.copy())
        splits = {
            split: torch.from_numpy(read_csv(f"split/time/{filename}").to_numpy(dtype=np.int64, copy=True).ravel())
            for split, filename in (("train", "train"), ("validation", "valid"), ("test", "test"))
        }
        categories = read_csv("mapping/labelidx2arxivcategeory", header=0)
        paper_ids = read_csv("mapping/nodeidx2paperid", header=0)["paper id"].to_numpy()
    # Every paper must belong to exactly one official split.
    indices = torch.cat(list(splits.values()))
    if not torch.equal(indices.sort().values, torch.arange(len(labels))):
        raise ValueError("Expected disjoint official splits covering every node")
    class_labels = dict(zip(categories["label idx"].astype(str), categories["arxiv category"]))
    return features, labels, edges, splits, class_labels, paper_ids


def mean_adjacency(edges, num_nodes, device):
    """Deduplicate undirected neighbors and average each node's neighbor features."""
    edges = edges[:, edges[0] != edges[1]]  # The self branch is learned separately.
    indices = torch.cat((edges, edges.flip(0)), dim=1).to(device)
    adjacency = torch.sparse_coo_tensor(
        indices, torch.ones(indices.shape[1], device=device), (num_nodes, num_nodes)
    ).coalesce()
    indices = adjacency.indices()
    degrees = torch.bincount(indices[0], minlength=num_nodes).clamp_min(1)
    weights = degrees[indices[0]].float().reciprocal()
    return torch.sparse_coo_tensor(indices, weights, adjacency.shape).coalesce()


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout):
        super().__init__()
        self.self_in = torch.nn.Linear(in_channels, hidden_channels)
        self.neighbor_in = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.self_out = torch.nn.Linear(hidden_channels, num_classes)
        self.neighbor_out = torch.nn.Linear(hidden_channels, num_classes, bias=False)
        self.dropout = dropout

    def forward(self, features, adjacency):
        hidden = F.relu(self.self_in(features) + self.neighbor_in(torch.sparse.mm(adjacency, features)))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.self_out(hidden) + self.neighbor_out(torch.sparse.mm(adjacency, hidden))


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    if p["epochs"] < 1 or not 0 <= p["dropout"] < 1:
        raise ValueError("Require epochs >= 1 and 0 <= dropout < 1")
    torch.manual_seed(p["seed"])
    device = torch.device("cuda")
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    features, labels, edges, splits, class_labels, paper_ids = load_data(p)
    adjacency = mean_adjacency(edges, len(labels), device)
    features, labels = features.to(device), labels.to(device)
    splits = {key: value.to(device) for key, value in splits.items()}
    model_config = {
        "in_channels": features.shape[1], "hidden_channels": p["hidden_channels"],
        "num_classes": len(class_labels), "dropout": p["dropout"],
    }
    model = GraphSAGE(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best_accuracy, best_epoch = -1.0, 0
    checkpoint = output_dir / "best_model.pt"

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"dataset": "ogbn-arxiv", "protocol": "transductive", **p})
        mlflow.log_params({f"{key}_nodes": len(value) for key, value in splits.items()})
        mlflow.log_params({"nodes": len(labels), "undirected_edges": adjacency._nnz()})
        started = time.perf_counter()
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, adjacency)
            loss = F.cross_entropy(logits[splits["train"]], labels[splits["train"]])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                predictions = model(features, adjacency).argmax(dim=1)
                metrics = {
                    "train_loss": loss.item(),
                    **{f"{split}_accuracy": (predictions[splits[split]] == labels[splits[split]]).float().mean().item()
                       for split in ("train", "validation")},
                }
            mlflow.log_metrics(metrics, step=epoch)
            if metrics["validation_accuracy"] > best_accuracy:
                best_accuracy, best_epoch = metrics["validation_accuracy"], epoch
                torch.save(model.state_dict(), checkpoint)
            print(f"Epoch {epoch}/{p['epochs']}: {metrics}", flush=True)

        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            test_predictions = model(features, adjacency)[splits["test"]].argmax(dim=1).cpu().numpy()
        test_labels = labels[splits["test"]].cpu().numpy()
        metrics = {
            "test_accuracy": float((test_predictions == test_labels).mean()),
            "test_f1_macro": float(f1_score(test_labels, test_predictions, labels=np.arange(len(class_labels)),
                                           average="macro", zero_division=0)),
            "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
            "train_and_eval_seconds": time.perf_counter() - started,
        }
        test_nodes = splits["test"].cpu().numpy()
        pd.DataFrame({
            "node_index": test_nodes, "paper_id": paper_ids[test_nodes],
            "true_label": test_labels, "predicted_label": test_predictions,
            "predicted_topic": [class_labels[str(label)] for label in test_predictions],
        }).to_csv(output_dir / "test_predictions.csv", index=False)
        for name, value in (("model_config", model_config), ("class_labels", class_labels), ("metrics", metrics)):
            (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir))
        print(f"Test metrics: {metrics}\nArtifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
