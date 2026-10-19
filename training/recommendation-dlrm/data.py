"""List, download, and preprocess Criteo click-log Parquet files for DLRM training."""

import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DENSE_COLUMNS = [f"integer_feature_{i}" for i in range(1, 14)]
SPARSE_COLUMNS = [f"categorical_feature_{i}" for i in range(1, 27)]


def list_day_files(repo, day):
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main/data/{day}"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    return sorted(entry["path"] for entry in entries if entry["path"].endswith(".parquet"))


def take(files, start, count, name):
    if count < 1 or start + count > len(files):
        raise ValueError(f"{name} needs {count} files starting at {start}, but the day has {len(files)}")
    return files[start:start + count]


def split_files(train_day_files, eval_day_files, p, rank, world):
    """Shard train files across ranks; validation and test are disjoint files from a later day."""
    if p["train_day"] == p["eval_day"]:
        raise ValueError("train_day and eval_day must be different days")
    if p["train_files"] < world:
        raise ValueError(f"train_files ({p['train_files']}) must be at least the number of ranks ({world})")
    train = take(train_day_files, 0, p["train_files"], "train_files")[rank::world]
    validation = take(eval_day_files, 0, p["validation_files"], "validation_files")
    test = take(eval_day_files, p["validation_files"], p["test_files"], "test_files")
    return train, validation, test


def download(repo, path, data_root):
    target = Path(data_root) / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial")
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(target)
    return target


def encode_dense(table):
    """log(1 + x), with missing and negative counts set to 0 (DLRM's Criteo transform)."""
    columns = [pc.fill_null(table[name], 0).to_numpy() for name in DENSE_COLUMNS]
    return np.log1p(np.maximum(np.stack(columns, axis=1), 0)).astype(np.float32)


def encode_sparse(table, hash_size):
    """Hash hex categories into rows 1..hash_size-1; row 0 is reserved for missing values."""
    out = np.empty((table.num_rows, len(SPARSE_COLUMNS)), dtype=np.int64)
    for j, name in enumerate(SPARSE_COLUMNS):
        # Parse each distinct string once instead of once per row.
        encoded = pc.dictionary_encode(table[name].combine_chunks())
        hashes = np.array([int(value, 16) if value else -1 for value in encoded.dictionary.to_pylist()],
                          dtype=np.int64)
        ids = np.append(np.where(hashes >= 0, 1 + hashes % (hash_size - 1), 0), 0)
        out[:, j] = ids[pc.fill_null(encoded.indices, len(hashes)).to_numpy()]
    return out


def row_slice(num_rows, rank, world):
    """Equal contiguous slices so every rank runs the same number of eval batches."""
    length = num_rows // world
    return rank * length, length


def load_split(paths, hash_size, rank=0, world=1):
    columns = ["label", *DENSE_COLUMNS, *SPARSE_COLUMNS]
    table = pa.concat_tables(pq.read_table(path, columns=columns) for path in paths)
    table = table.slice(*row_slice(table.num_rows, rank, world))
    labels = table["label"].to_numpy().astype(np.float32)
    return encode_dense(table), encode_sparse(table, hash_size), labels


def num_steps(rows, batch_size):
    steps = rows // batch_size
    if steps < 1:
        raise ValueError(f"{rows} rows is fewer than one batch of {batch_size}; "
                         "use more files or a smaller batch_size")
    return steps
