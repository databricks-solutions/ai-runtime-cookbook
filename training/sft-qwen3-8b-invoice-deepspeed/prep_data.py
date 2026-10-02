#!/usr/bin/env python3
"""Stage the invoice SFT dataset as HF datasets on a Unity Catalog Volume.

One-time prerequisite for ``train.py``. The AIR GPU worker has no Spark, so this
script pulls the public **Winuim/invoice-sft-dataset-v2** dataset (OCR text paired
with ground-truth extraction JSON in chat-message format), builds a conversational
``messages`` column, carves out a train/eval split, and writes both to a UC Volume
that ``train.py`` reads with ``load_from_disk``.

Run it once from any environment with write access to the target volume (for
example a Databricks cluster web terminal) before ``databricks air run``:

    pip install datasets huggingface_hub
    python prep_data.py --output-root /Volumes/main/default/air_examples/sft-qwen3-8b-invoice-deepspeed

Pass the same ``--output-root`` you set for ``output_root`` in ``workload.yaml``;
the datasets are written under ``<output-root>/data``.
"""

import argparse
import json
import os

# The HF `datasets` cache defaults to /root/.cache, which may be read-only. Point
# it at a writable local path before importing datasets.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_home/datasets")

from datasets import load_dataset, load_from_disk


def to_example(row):
    """Fold system+user into one user turn and keep the assistant JSON as the target."""
    system_prompt, ocr_text, extraction_json = "", None, None
    for msg in row["messages"]:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            ocr_text = msg["content"]
        elif msg["role"] == "assistant":
            extraction_json = msg["content"]
    # Guard before touching the strings: a row missing the user or assistant
    # message would otherwise crash the whole map. Invalid rows are emitted with
    # a consistent schema and dropped by the filter below.
    if ocr_text is None or extraction_json is None:
        return {"messages": [{"role": "user", "content": ""},
                             {"role": "assistant", "content": ""}], "_valid": False}
    # Separate the system prompt from the OCR text so their tokens don't glue.
    user_content = f"{system_prompt.rstrip()}\n\n{ocr_text}".strip() if system_prompt else ocr_text.strip()
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": extraction_json.strip()},
        ],
        "_valid": True,
    }


def _is_valid_json(row):
    if not row["_valid"]:
        return False
    try:
        json.loads(row["messages"][1]["content"])
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-root",
        default="/Volumes/main/default/air_examples/sft-qwen3-8b-invoice-deepspeed",
        help="UC Volume path; datasets are written under <output-root>/data. "
             "Must match output_root in workload.yaml.",
    )
    parser.add_argument("--hf-dataset", default="Winuim/invoice-sft-dataset-v2",
                        help="Public Hugging Face dataset to stage.")
    parser.add_argument("--test-size", type=float, default=0.05,
                        help="Fraction of rows held out for eval.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Split seed (fixed for reproducible re-runs).")
    args = parser.parse_args()

    data_dir = f"{args.output_root}/data"
    train_path = f"{data_dir}/invoice_train_dataset"
    eval_path = f"{data_dir}/invoice_eval_dataset"
    print(f"HF dataset: {args.hf_dataset}")
    print(f"Output:     {train_path}, {eval_path}")

    # Each row is a list of {role, content} chat messages: system + user (OCR text)
    # + assistant (ground-truth extraction JSON).
    raw = load_dataset(args.hf_dataset, split="train", cache_dir=os.environ["HF_DATASETS_CACHE"])
    parsed = raw.map(to_example, remove_columns=raw.column_names)

    before = len(parsed)
    parsed = parsed.filter(_is_valid_json).remove_columns(["_valid"])
    print(f"Kept {len(parsed)}/{before} rows (dropped empty / invalid-JSON rows).")

    # The public dataset is a single `train` split, so carve the eval set out here.
    split = parsed.train_test_split(test_size=args.test_size, seed=args.seed)
    split["train"].save_to_disk(train_path)
    split["test"].save_to_disk(eval_path)
    print(f"Train: {len(split['train'])} samples -> {train_path}")
    print(f"Eval:  {len(split['test'])} samples -> {eval_path}")

    # Verify the staged data loads back and show one example.
    train_check = load_from_disk(train_path)
    print(f"\nVerify: {len(train_check)} train samples, columns={train_check.column_names}")
    if len(train_check):
        for msg in train_check[0]["messages"]:
            print(f"  [{msg['role']}]: {msg['content'][:200]}...")
    print("\nDone. Data is staged and ready for `databricks air run -f workload.yaml`.")


if __name__ == "__main__":
    main()
