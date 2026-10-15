#!/usr/bin/env python3
"""Train a two-tower retriever and evaluate warm-start full-catalog retrieval."""

import io
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec


def load_ratings(p):
    root = Path(p["data_root"])
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "ml-1m.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(p["dataset_url"], timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source, source.open("ml-1m/ratings.dat") as handle:
        return pd.read_csv(handle, sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp"])


def split_ratings(ratings, train_fraction, validation_fraction):
    if not 0 < train_fraction < train_fraction + validation_fraction < 1:
        raise ValueError("Require positive train/validation fractions summing to less than one")
    train_cutoff, validation_cutoff = [
        int(ratings.timestamp.quantile(fraction))
        for fraction in (train_fraction, train_fraction + validation_fraction)
    ]
    splits = (
        ratings[ratings.timestamp <= train_cutoff].copy(),
        ratings[(ratings.timestamp > train_cutoff) & (ratings.timestamp <= validation_cutoff)].copy(),
        ratings[ratings.timestamp > validation_cutoff].copy(),
    )
    if any(frame.empty for frame in splits):
        raise ValueError("Timestamp cutoffs produced an empty split")
    return splits, {"train_cutoff_timestamp": train_cutoff, "validation_cutoff_timestamp": validation_cutoff}


def encode_pairs(frame, user_ids, item_ids):
    users = pd.Index(user_ids).get_indexer(frame.user_id)
    items = pd.Index(item_ids).get_indexer(frame.item_id)
    known = (users >= 0) & (items >= 0)
    return torch.from_numpy(np.column_stack((users[known], items[known])).astype(np.int64))


def interaction_mask(pairs, num_users, num_items, device):
    mask = torch.zeros((num_users, num_items), dtype=torch.bool, device=device)
    pairs = pairs.to(device)
    mask[pairs[:, 0], pairs[:, 1]] = True
    return mask


def evaluation_data(frame, history, user_ids, item_ids, p, device):
    positive = frame[frame.rating >= p["positive_rating"]]
    pairs = encode_pairs(positive, user_ids, item_ids)
    seen = interaction_mask(encode_pairs(history, user_ids, item_ids), len(user_ids), len(item_ids), device)
    targets = interaction_mask(pairs, len(user_ids), len(item_ids), device) & ~seen
    users = torch.where(targets.any(dim=1))[0]
    if len(users) == 0:
        raise ValueError("No eligible warm-start evaluation users")
    counts = {
        "evaluated_users": len(users), "eligible_positives": targets.sum().item(),
        "total_positives": len(positive), "positive_coverage": targets.sum().item() / len(positive),
    }
    return users, targets, seen, counts


class TwoTower(torch.nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_tower = self.tower(num_users, embedding_dim, hidden_dim)
        self.item_tower = self.tower(num_items, embedding_dim, hidden_dim)

    @staticmethod
    def tower(size, embedding_dim, hidden_dim):
        embedding = torch.nn.Embedding(size, embedding_dim)
        torch.nn.init.normal_(embedding.weight, std=0.1)
        return torch.nn.Sequential(embedding, torch.nn.Linear(embedding_dim, hidden_dim),
                                   torch.nn.ReLU(), torch.nn.Linear(hidden_dim, embedding_dim))

    def forward(self, pairs):
        return (self.user_tower(pairs[:, 0]) * self.item_tower(pairs[:, 1])).sum(dim=1)


@torch.no_grad()
def evaluate(model, data, p, popularity=None):
    model.eval()
    users, targets, seen, _ = data
    device = seen.device
    item_vectors = model.item_tower(torch.arange(seen.shape[1], device=device))
    k = min(p["top_k"], seen.shape[1])
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
    ideal_discounts = discounts.cumsum(dim=0)
    recall, ndcg = 0.0, 0.0
    for batch in users.split(p["eval_batch_size"]):
        scores = (model.user_tower(batch) @ item_vectors.T if popularity is None
                  else popularity.unsqueeze(0).expand(len(batch), -1).clone())
        scores.masked_fill_(seen[batch], -torch.inf)
        top_items = scores.topk(k, dim=1).indices
        hits = targets[batch].gather(1, top_items).float()
        relevant = targets[batch].sum(dim=1)
        recall += (hits.sum(dim=1) / relevant).sum().item()
        ndcg += ((hits * discounts).sum(dim=1) / ideal_discounts[relevant.clamp(max=k) - 1]).sum().item()
    return {f"recall_at_{k}": recall / len(users), f"ndcg_at_{k}": ndcg / len(users)}


def sample_negatives(users, seen):
    """Sample uniformly from items not rated by each user in training."""
    negatives = torch.randint(seen.shape[1], users.shape, device=users.device)
    rejected = seen[users, negatives]
    while rejected.any():
        negatives[rejected] = torch.randint(seen.shape[1], (int(rejected.sum()),), device=users.device)
        rejected = seen[users, negatives]
    return negatives


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    # The CLI can serialize small numbers as 1e-06, which PyYAML reads as a string.
    for key in ("train_fraction", "validation_fraction", "learning_rate", "weight_decay"):
        p[key] = float(p[key])
    if min(p["epochs"], p["batch_size"], p["eval_batch_size"], p["top_k"]) < 1:
        raise ValueError("Epochs, batch sizes, and top_k must be positive")
    torch.manual_seed(p["seed"])
    device = torch.device("cuda")
    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (train, validation, test), cutoffs = split_ratings(load_ratings(p), p["train_fraction"], p["validation_fraction"])
    positive = train[train.rating >= p["positive_rating"]]
    user_ids, item_ids = np.sort(positive.user_id.unique()), np.sort(positive.item_id.unique())
    if len(user_ids) == 0 or len(item_ids) < p["top_k"]:
        raise ValueError("Not enough training users/items for the configured top_k")
    pairs = encode_pairs(positive, user_ids, item_ids).to(device)
    seen = interaction_mask(encode_pairs(train, user_ids, item_ids), len(user_ids), len(item_ids), device)
    if seen.all(dim=1).any():
        raise ValueError("Every training user must have an unobserved item for negative sampling")
    validation_data = evaluation_data(validation, train, user_ids, item_ids, p, device)
    config = {"num_users": len(user_ids), "num_items": len(item_ids),
              "embedding_dim": p["embedding_dim"], "hidden_dim": p["hidden_dim"]}
    model = TwoTower(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"], weight_decay=p["weight_decay"])
    best_recall, best_epoch = -1.0, 0
    checkpoint = output_dir / "best_model.pt"

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params({"dataset": "MovieLens/1M", **p, **cutoffs, **config, "train_positives": len(pairs)})
        mlflow.log_metrics({f"validation_{key}": value for key, value in validation_data[3].items()})
        started = time.perf_counter()
        for epoch in range(1, p["epochs"] + 1):
            model.train()
            total_loss = 0.0
            order = torch.randperm(len(pairs), device=device)
            for indices in order.split(p["batch_size"]):
                batch = pairs[indices]
                users, items = batch[:, 0], batch[:, 1]
                negatives = sample_negatives(users, seen)
                user_vectors = model.user_tower(users)
                margin = (user_vectors * (model.item_tower(items) - model.item_tower(negatives))).sum(dim=1)
                loss = F.softplus(-margin).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            validation_metrics = evaluate(model, validation_data, p)
            metrics = {"train_bpr_loss": total_loss / len(pairs),
                       **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            mlflow.log_metrics(metrics, step=epoch)
            recall = validation_metrics[f"recall_at_{p['top_k']}"]
            if recall > best_recall:
                best_recall, best_epoch = recall, epoch
                torch.save(model.state_dict(), checkpoint)
            print(f"Epoch {epoch}/{p['epochs']}: {metrics}", flush=True)

        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_data = evaluation_data(test, pd.concat([train, validation]), user_ids, item_ids, p, device)
        test_metrics = evaluate(model, test_data, p)
        popularity = torch.bincount(pairs[:, 1], minlength=len(item_ids)).float()
        baseline = evaluate(model, test_data, p, popularity=popularity)
        metrics = {
            **{f"test_{key}": value for key, value in {**test_metrics, **test_data[3]}.items()},
            **{f"popularity_{key}": value for key, value in baseline.items()},
            "best_epoch": best_epoch, "best_validation_recall": best_recall,
            "train_and_eval_seconds": time.perf_counter() - started,
        }
        model = model.cpu().eval()
        with torch.no_grad():
            # ZIP creation seeks within the file; UC volumes require sequential writes.
            with io.BytesIO() as buffer:
                np.savez_compressed(buffer,
                                    user_embeddings=model.user_tower(torch.arange(len(user_ids))).numpy(),
                                    item_embeddings=model.item_tower(torch.arange(len(item_ids))).numpy())
                (output_dir / "embeddings.npz").write_bytes(buffer.getvalue())
            example = pairs[:5].cpu()
        mappings = {"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist()}
        for name, value in (("model_config", config), ("id_mappings", mappings), ("metrics", metrics)):
            (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(output_dir))
        info = mlflow.pytorch.log_model(
            pytorch_model=model, name="model", input_example=example.numpy(),
            signature=ModelSignature(
                inputs=Schema([TensorSpec(np.dtype("int64"), (-1, 2))]),
                outputs=Schema([TensorSpec(np.dtype("float32"), (-1,))]),
            ),
        )
        print(f"Test metrics: {metrics}\nArtifacts: {output_dir}\nModel: {info.model_uri}", flush=True)


if __name__ == "__main__":
    main()
