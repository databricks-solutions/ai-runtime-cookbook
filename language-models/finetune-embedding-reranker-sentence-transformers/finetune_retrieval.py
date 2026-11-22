#!/usr/bin/env python3
"""Fine-tune a bi-encoder and a cross-encoder reranker on FiQA and compare test retrieval quality."""

import csv
import json
import os
import random
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    mine_hard_negatives,
)
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss

# Text-only EmbeddingGemma 2 loads 270M of its 740M parameters; the vision and audio encoders are not needed here.
TEXT_ONLY_CONFIG = {"vision_config": None, "audio_config": None}
# Query prefix from the EmbeddingGemma 2 model card. Documents use "title: none" when a title is missing.
QUERY_TEMPLATE = "task: search result | query: {text}"
DOCUMENT_TEMPLATE = "title: {title} | text: {text}"
RECALL_K = 100
NDCG_K = 10
SPLITS = ("train", "dev", "test")


def require_int(name, value, minimum):
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def load_qrels(repo_name, split):
    """Returns {query_id: set(corpus_id)} of positive judgments. BEIR qrels are binary."""
    path = hf_hub_download(repo_name, f"{split}.tsv", repo_type="dataset")
    qrels = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if float(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return qrels


def limit_queries(query_ids, limit):
    query_ids = sorted(query_ids)
    return query_ids[:limit] if limit else query_ids


def encode(model, texts, batch_size):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def top_candidates(query_emb, corpus_emb, k):
    """Exact cosine search. Embeddings are normalized, so a dot product is the cosine similarity."""
    corpus = torch.from_numpy(corpus_emb).to("cuda")
    indices = []
    for start in range(0, len(query_emb), 256):
        queries = torch.from_numpy(query_emb[start:start + 256]).to("cuda")
        _, top = torch.topk(queries @ corpus.T, k=k, dim=1)
        indices.append(top.cpu().numpy())
    del corpus
    return np.concatenate(indices)


def mean_and_standard_error(values):
    """Standard error of the mean over queries, so metric differences can be compared to it."""
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def paired_standard_error(values_a, values_b):
    """Standard error of the mean per-query difference of two systems on the same queries."""
    differences = np.asarray(values_a) - np.asarray(values_b)
    return float(np.std(differences, ddof=1) / np.sqrt(len(differences)))


def evaluate_rankings(ranked_ids, query_ids, qrels):
    """Each ranked list holds corpus IDs in rank order.

    Returns the mean nDCG@10 and Recall@100, each with its standard error over queries,
    plus the raw per-query values for paired comparisons between systems.
    """
    ndcg, recall = [], []
    for query_id, ranked in zip(query_ids, ranked_ids):
        relevant = qrels[query_id]
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, doc_id in enumerate(ranked[:NDCG_K])
            if doc_id in relevant
        )
        ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(NDCG_K, len(relevant))))
        ndcg.append(dcg / ideal)
        recall.append(sum(doc_id in relevant for doc_id in ranked[:RECALL_K]) / len(relevant))
    ndcg_mean, ndcg_se = mean_and_standard_error(ndcg)
    recall_mean, recall_se = mean_and_standard_error(recall)
    summary = {
        f"ndcg_at_{NDCG_K}": ndcg_mean,
        f"ndcg_at_{NDCG_K}_se": ndcg_se,
        f"recall_at_{RECALL_K}": recall_mean,
        f"recall_at_{RECALL_K}_se": recall_se,
    }
    return summary, {f"ndcg_at_{NDCG_K}": ndcg, f"recall_at_{RECALL_K}": recall}


def retrieve_and_score(model, corpus_texts, corpus_ids, query_texts, query_ids, qrels, batch_size):
    corpus_emb = encode(model, corpus_texts, batch_size)
    query_emb = encode(model, query_texts, batch_size)
    candidates = top_candidates(query_emb, corpus_emb, RECALL_K)
    ranked = [corpus_ids[row].tolist() for row in candidates]
    summary, per_query = evaluate_rankings(ranked, query_ids, qrels)
    return summary, per_query, candidates


def load_embedder(path, dtype, max_seq_length):
    model = SentenceTransformer(path, device="cuda", model_kwargs={"dtype": dtype}, config_kwargs=TEXT_ONLY_CONFIG)
    model.max_seq_length = max_seq_length
    return model


def rerank(reranker, query_texts, doc_texts, candidates, depths, batch_size):
    """Rescores the first max(depths) candidates. Returns ranked corpus indices for each depth.

    Candidates below a depth keep their first-stage order, so Recall@100 is comparable across depths.
    """
    max_depth = max(depths)
    pairs = [
        (query_texts[row], doc_texts[col])
        for row in range(len(candidates))
        for col in candidates[row, :max_depth]
    ]
    scores = reranker.predict(
        pairs, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
    ).reshape(len(candidates), max_depth)
    results = {}
    for depth in depths:
        reordered = []
        for row in range(len(candidates)):
            order = np.argsort(-scores[row, :depth], kind="stable")
            reordered.append(np.concatenate([candidates[row, :depth][order], candidates[row, depth:]]))
        results[depth] = np.stack(reordered)
    return results


def log_to_mlflow(run_id, params, metrics, artifact_paths):
    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="results")


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)

    # Zero means "all queries" for the query limits and is otherwise invalid for these settings.
    for name in ("max_seq_length", "embed_batch_size", "mini_batch_size", "embedder_train_batch_size",
                 "reranker_train_batch_size", "reranker_max_length", "rerank_batch_size",
                 "negatives_per_positive", "embedder_epochs", "reranker_epochs"):
        require_int(name, p[name], minimum=1)
    for name in ("max_train_queries", "max_eval_queries", "negative_range_min"):
        require_int(name, p[name], minimum=0)
    require_int("seed", p["seed"], minimum=0)
    if not p["rerank_depths"] or any(not 1 <= depth <= RECALL_K for depth in p["rerank_depths"]):
        raise ValueError(f"rerank_depths must be between 1 and {RECALL_K}")
    if p["negative_range_max"] <= p["negative_range_min"]:
        raise ValueError("negative_range_max must be greater than negative_range_min")
    embedder_lr = float(p["embedder_learning_rate"])
    reranker_lr = float(p["reranker_learning_rate"])
    if not torch.cuda.is_available():
        raise RuntimeError("This recipe requires a CUDA GPU")

    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])
    started = time.perf_counter()
    timings = {}

    # Load the corpus, the queries, and the relevance labels for each split.
    corpus = load_dataset(p["dataset_name"], "corpus", split="corpus")
    queries = load_dataset(p["dataset_name"], "queries", split="queries")
    corpus_ids = np.array(corpus["_id"])
    doc_titles = corpus["title"]
    doc_texts = corpus["text"]
    doc_index = {doc_id: i for i, doc_id in enumerate(corpus["_id"])}
    query_text = dict(zip(queries["_id"], queries["text"]))
    qrels = {split: load_qrels(p["qrels_dataset_name"], split) for split in SPLITS}

    # Model inputs use the EmbeddingGemma 2 formats. Reranker inputs use plain title and text.
    embedder_docs = [
        DOCUMENT_TEMPLATE.format(title=(title.strip() or "none"), text=text.strip())
        for title, text in zip(doc_titles, doc_texts)
    ]
    plain_docs = [f"{title} {text}".strip() for title, text in zip(doc_titles, doc_texts)]

    train_ids = limit_queries(qrels["train"], p["max_train_queries"])
    dev_ids = limit_queries(qrels["dev"], p["max_eval_queries"])
    test_ids = limit_queries(qrels["test"], p["max_eval_queries"])

    def split_inputs(query_ids):
        return (
            [QUERY_TEMPLATE.format(text=query_text[q].strip()) for q in query_ids],
            [query_text[q].strip() for q in query_ids],
        )

    dev_queries, dev_plain = split_inputs(dev_ids)
    test_queries, test_plain = split_inputs(test_ids)
    print(
        f"Corpus: {len(embedder_docs)} documents. Queries: train {len(train_ids)}, "
        f"dev {len(dev_ids)}, test {len(test_ids)}.",
        flush=True,
    )

    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="finetune_"))

    # Stage 1: baseline retrieval with the pretrained embedder.
    stage_start = time.perf_counter()
    base = load_embedder(p["embedder_model"], torch.bfloat16, p["max_seq_length"])
    base_dev, _, _ = retrieve_and_score(base, embedder_docs, corpus_ids, dev_queries, dev_ids, qrels["dev"], p["embed_batch_size"])
    base_test, base_test_queries, _ = retrieve_and_score(base, embedder_docs, corpus_ids, test_queries, test_ids, qrels["test"], p["embed_batch_size"])
    print(f"Base embedder dev: {base_dev}\nBase embedder test: {base_test}", flush=True)

    # Stage 2: mine hard negatives for training queries with the pretrained embedder.
    train_pairs = [
        (query_text[q].strip(), q, doc)
        for q in train_ids
        for doc in sorted(qrels["train"][q])
        if doc in doc_index
    ]
    mining_dataset = Dataset.from_dict({
        "anchor": [QUERY_TEMPLATE.format(text=text) for text, _, _ in train_pairs],
        "positive": [embedder_docs[doc_index[doc]] for _, _, doc in train_pairs],
    })
    triplets = mine_hard_negatives(
        mining_dataset,
        base,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=embedder_docs,
        range_min=p["negative_range_min"],
        range_max=p["negative_range_max"],
        num_negatives=p["negatives_per_positive"],
        sampling_strategy="random",
        output_format="triplet",
        batch_size=p["embed_batch_size"],
        use_faiss=False,
        verbose=False,
    )
    timings["baseline_and_mining_seconds"] = time.perf_counter() - stage_start
    print(f"Mined {len(triplets)} (query, positive, hard negative) triplets.", flush=True)
    del base
    torch.cuda.empty_cache()

    # Stage 3: fine-tune the bi-encoder. The cached loss gives a large in-batch negative pool without holding every activation.
    stage_start = time.perf_counter()
    trainee = load_embedder(p["embedder_model"], torch.float32, p["max_seq_length"])
    embedder_args = SentenceTransformerTrainingArguments(
        output_dir=str(scratch / "embedder"),
        num_train_epochs=p["embedder_epochs"],
        per_device_train_batch_size=p["embedder_train_batch_size"],
        learning_rate=embedder_lr,
        warmup_ratio=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=p["seed"],
    )
    embedder_trainer = SentenceTransformerTrainer(
        model=trainee,
        args=embedder_args,
        train_dataset=triplets,
        loss=CachedMultipleNegativesRankingLoss(trainee, mini_batch_size=p["mini_batch_size"]),
    )
    embedder_trainer.train()
    embedder_dir = output_dir / "embedder"
    trainee.save(str(embedder_dir))
    del trainee, embedder_trainer
    torch.cuda.empty_cache()
    timings["embedder_training_seconds"] = time.perf_counter() - stage_start

    # Reload the saved model so the evaluated weights are the ones written to the volume.
    stage_start = time.perf_counter()
    tuned = load_embedder(str(embedder_dir), torch.bfloat16, p["max_seq_length"])
    tuned_dev, _, dev_candidates = retrieve_and_score(tuned, embedder_docs, corpus_ids, dev_queries, dev_ids, qrels["dev"], p["embed_batch_size"])
    tuned_test, tuned_test_queries, test_candidates = retrieve_and_score(tuned, embedder_docs, corpus_ids, test_queries, test_ids, qrels["test"], p["embed_batch_size"])
    print(f"Fine-tuned embedder dev: {tuned_dev}\nFine-tuned embedder test: {tuned_test}", flush=True)
    del tuned
    torch.cuda.empty_cache()
    timings["tuned_embedder_eval_seconds"] = time.perf_counter() - stage_start

    # Stage 4: fine-tune the cross-encoder on the same mined triplets, as labeled pairs (1 for positive, 0 for negative).
    stage_start = time.perf_counter()
    plain_query_for = {QUERY_TEMPLATE.format(text=text): text for text, _, _ in train_pairs}
    plain_doc_for = dict(zip(embedder_docs, plain_docs))
    pair_queries, pair_docs, pair_labels = [], [], []
    for anchor, positive, negative in zip(triplets["anchor"], triplets["positive"], triplets["negative"]):
        pair_queries += [plain_query_for[anchor]] * 2
        pair_docs += [plain_doc_for[positive], plain_doc_for[negative]]
        pair_labels += [1.0, 0.0]
    pair_dataset = Dataset.from_dict({"query": pair_queries, "document": pair_docs, "label": pair_labels})

    reranker = CrossEncoder(
        p["reranker_model"], num_labels=1, max_length=p["reranker_max_length"], device="cuda"
    )
    reranker_args = CrossEncoderTrainingArguments(
        output_dir=str(scratch / "reranker"),
        num_train_epochs=p["reranker_epochs"],
        per_device_train_batch_size=p["reranker_train_batch_size"],
        learning_rate=reranker_lr,
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=p["seed"],
    )
    reranker_trainer = CrossEncoderTrainer(
        model=reranker,
        args=reranker_args,
        train_dataset=pair_dataset,
        loss=BinaryCrossEntropyLoss(reranker),
    )
    reranker_trainer.train()
    reranker_dir = output_dir / "reranker"
    reranker.save(str(reranker_dir))
    timings["reranker_training_seconds"] = time.perf_counter() - stage_start

    # Stage 5: choose the rerank depth on dev, then report test once.
    stage_start = time.perf_counter()
    reranker = CrossEncoder(str(reranker_dir), device="cuda", max_length=p["reranker_max_length"])
    depths = sorted(set(p["rerank_depths"]))
    dev_reranked = rerank(reranker, dev_plain, plain_docs, dev_candidates, depths, p["rerank_batch_size"])
    dev_by_depth, dev_per_query = {}, {}
    for depth in depths:
        dev_by_depth[depth], dev_per_query[depth] = evaluate_rankings(
            [corpus_ids[row].tolist() for row in dev_reranked[depth]], dev_ids, qrels["dev"]
        )
    best_depth = max(depths, key=lambda depth: (dev_by_depth[depth][f"ndcg_at_{NDCG_K}"], -depth))
    test_reranked = rerank(reranker, test_plain, plain_docs, test_candidates, depths, p["rerank_batch_size"])
    test_by_depth, test_per_query = {}, {}
    for depth in depths:
        test_by_depth[depth], test_per_query[depth] = evaluate_rankings(
            [corpus_ids[row].tolist() for row in test_reranked[depth]], test_ids, qrels["test"]
        )
    timings["reranking_seconds"] = time.perf_counter() - stage_start
    dev_ndcg_by_depth = {depth: round(scores[f"ndcg_at_{NDCG_K}"], 4) for depth, scores in dev_by_depth.items()}
    print(f"Dev nDCG@10 by rerank depth: {dev_ndcg_by_depth}. Chosen depth: {best_depth}.", flush=True)

    results = {
        "test": {
            "base_embedder": base_test,
            "fine_tuned_embedder": tuned_test,
            "fine_tuned_embedder_and_reranker": test_by_depth[best_depth],
        },
        "dev": {
            "base_embedder": base_dev,
            "fine_tuned_embedder": tuned_dev,
            "fine_tuned_embedder_and_reranker": dev_by_depth[best_depth],
        },
        "reranker_depth_dev_sweep": {str(depth): dev_by_depth[depth] for depth in depths},
        "reranker_depth_test_sweep": {str(depth): test_by_depth[depth] for depth in depths},
        "selected_rerank_depth": best_depth,
    }
    # Both systems score the same test queries, so the paired standard error of the
    # per-query difference is the right uncertainty for each improvement claim.
    paired = {
        "fine_tuned_minus_base": {
            f"ndcg_at_{NDCG_K}_se": paired_standard_error(
                tuned_test_queries[f"ndcg_at_{NDCG_K}"], base_test_queries[f"ndcg_at_{NDCG_K}"]
            ),
            f"recall_at_{RECALL_K}_se": paired_standard_error(
                tuned_test_queries[f"recall_at_{RECALL_K}"], base_test_queries[f"recall_at_{RECALL_K}"]
            ),
        },
        "reranked_minus_fine_tuned": {
            f"ndcg_at_{NDCG_K}_se": paired_standard_error(
                test_per_query[best_depth][f"ndcg_at_{NDCG_K}"], tuned_test_queries[f"ndcg_at_{NDCG_K}"]
            ),
            f"recall_at_{RECALL_K}_se": paired_standard_error(
                test_per_query[best_depth][f"recall_at_{RECALL_K}"], tuned_test_queries[f"recall_at_{RECALL_K}"]
            ),
        },
    }
    results["paired_standard_errors"] = paired
    counts = {
        "corpus_documents": len(embedder_docs),
        "train_queries": len(train_ids),
        "dev_queries": len(dev_ids),
        "test_queries": len(test_ids),
        "mined_triplets": len(triplets),
        "reranker_training_pairs": len(pair_dataset),
    }
    metrics = {**counts, **timings, "total_seconds": time.perf_counter() - started, "results": results}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    parameters = {**p, "embedder_learning_rate": embedder_lr, "reranker_learning_rate": reranker_lr}
    (output_dir / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")

    # MLflow metric names cannot contain "@", so the metric keys use "at".
    flat_metrics = {}
    for split in ("dev", "test"):
        for system, scores in results[split].items():
            if isinstance(scores, dict) and system != "selected_rerank_depth":
                for key, value in scores.items():
                    flat_metrics[f"{split}_{system}_{key}"] = value
    for comparison, se_scores in paired.items():
        for key, value in se_scores.items():
            flat_metrics[f"test_{comparison}_{key}"] = value
    flat_metrics.update(timings)
    flat_metrics.update({"total_seconds": metrics["total_seconds"], "selected_rerank_depth": best_depth, **counts})
    log_params = {key: (json.dumps(value) if isinstance(value, list) else str(value)) for key, value in parameters.items()}
    log_to_mlflow(run_id, log_params, flat_metrics, [output_dir / "metrics.json", output_dir / "parameters.json"])

    print(json.dumps(results, indent=2), flush=True)
    print(f"Saved models and metrics under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
