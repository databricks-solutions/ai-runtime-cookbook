#!/usr/bin/env python3
"""Stage the invoice SFT dataset as HF datasets on a Unity Catalog Volume.

One-time prerequisite for ``train.py``. The AIR GPU worker has no Spark, so this
script pulls the public **Winuim/invoice-sft-dataset-v2** dataset (OCR text paired
with ground-truth extraction JSON in chat-message format), keeps only answers that
match the JSON schema the prompt requests, writes each example as a conversational
``prompt`` / ``completion`` pair (so ``completion_only_loss`` trains on the
assistant JSON only), carves out a train/eval split, and writes both to a UC Volume
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
import re

# The HF `datasets` cache defaults to /root/.cache, which may be read-only. Point
# it at a writable local path before importing datasets.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_home/datasets")

from datasets import load_dataset, load_from_disk


# The JSON schema the dataset's system prompt asks for. Answers that do not match
# it exactly are dropped rather than coerced: a portion of the public dataset
# carries raw OCR strings ("$1182,50", "$106.98 $106.98"), non-ISO dates, and other
# item layouts, and guessing their locale format would put wrong labels in training.
SCHEMA_KEYS = ["doc_type", "invoice_no", "date", "seller", "client", "subtotal", "tax", "total", "items"]
DOC_TYPES = {"invoice", "receipt", "credit_note"}
ITEM_KEYS = ["name", "quantity", "price"]
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def matches_schema(ans) -> bool:
    """True if the answer has exactly the prompt's keys and value types."""
    if not isinstance(ans, dict) or set(ans) != set(SCHEMA_KEYS):
        return False
    if ans["doc_type"] not in DOC_TYPES or not _is_num(ans["total"]):
        return False
    if any(ans[k] is not None and not isinstance(ans[k], str) for k in ("invoice_no", "seller", "client")):
        return False
    if ans["date"] is not None and not (isinstance(ans["date"], str) and ISO_DATE.fullmatch(ans["date"])):
        return False
    if any(ans[k] is not None and not _is_num(ans[k]) for k in ("subtotal", "tax")):
        return False
    if not isinstance(ans["items"], list):
        return False
    for item in ans["items"]:
        if not isinstance(item, dict) or set(item) != set(ITEM_KEYS) or not isinstance(item["name"], str):
            return False
        if any(item[k] is not None and not _is_num(item[k]) for k in ("quantity", "price")):
            return False
    return True


def to_example(row):
    """Build a conversational prompt/completion pair from a system+user+assistant row.

    The system prompt and OCR text are folded into one user turn (the prompt); the
    assistant extraction JSON is the completion. Rows whose answer is missing, not
    JSON, or off-schema are flagged and dropped by the filter in main().
    """
    system_prompt, ocr_text, extraction_json = "", None, None
    for msg in row["messages"]:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            ocr_text = msg["content"]
        elif msg["role"] == "assistant":
            extraction_json = msg["content"]
    ans = None
    if ocr_text is not None and extraction_json is not None:
        try:
            ans = json.loads(extraction_json)
        except json.JSONDecodeError:
            pass
    valid = matches_schema(ans)
    # Separate the system prompt from the OCR text so their tokens don't glue.
    user_content = f"{system_prompt.rstrip()}\n\n{ocr_text}".strip() if system_prompt else (ocr_text or "").strip()
    # Re-serialize valid answers in schema key order so every target is formatted
    # the same way. Invalid rows keep a consistent column schema for the filter.
    completion = json.dumps({k: ans[k] for k in SCHEMA_KEYS}, ensure_ascii=False) if valid else ""
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": completion}],
        "_valid": valid,
    }


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
    parsed = parsed.filter(lambda row: row["_valid"]).remove_columns(["_valid"])
    print(f"Kept {len(parsed)}/{before} rows (dropped rows whose answer is not schema-valid JSON).")

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
        for msg in train_check[0]["prompt"] + train_check[0]["completion"]:
            print(f"  [{msg['role']}]: {msg['content'][:200]}...")
    print("\nDone. Data is staged and ready for `databricks air run -f workload.yaml`.")


if __name__ == "__main__":
    main()
