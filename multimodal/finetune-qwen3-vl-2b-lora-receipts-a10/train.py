#!/usr/bin/env python3
"""LoRA fine-tuning of Qwen3-VL-2B-Instruct for receipt field extraction on one A10.

Scores the plain base model and a schema-prompted base model, trains a LoRA adapter on the CORD-v2
train split, then scores the tuned model. All three rows share greedy decoding and the scorer on the
CORD-v2 test split; the plain base and tuned rows also share the same prompt, so the tuned-versus-base
difference isolates the effect of training, while the schema baseline lists the train split's field
paths, the same paths the tuned model is trained to emit. Validation loss is reported before and after
training as a sanity check; nothing is selected on any split.

Run from this recipe's directory:

    databricks air run -f workload.yaml --watch
"""

import io
import json
import logging
import math
import os
from collections import Counter

import mlflow
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)

for _name in ("mlflow.tracking.context.registry", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

PROMPT = "Extract the fields of this receipt as a JSON object. Return only the JSON."
# LoRA on the language model's attention and MLP projections only. The vision tower (model.visual)
# and lm_head stay frozen.
LORA_TARGET_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
)


def load_params() -> dict:
    """Read the ``parameters:`` block that ``air`` materializes from the workload YAML."""
    path = os.environ.get("HYPERPARAMETERS_PATH")
    if path and os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_processor(model_name: str, min_pixels: int, max_pixels: int):
    # Bounds the visual tokens per receipt (each token covers 28x28 pixels after the 2x2 merge).
    # Passing size here keeps the bound on every call, including the one used during training.
    processor = AutoProcessor.from_pretrained(model_name)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        model_name, size={"shortest_edge": min_pixels, "longest_edge": max_pixels}
    )
    return processor


def as_rgb(image) -> Image.Image:
    """The dataset decodes images to PIL only when the feature type is declared; decode raw bytes otherwise."""
    if isinstance(image, dict):
        image = Image.open(io.BytesIO(image["bytes"]))
    return image.convert("RGB")


def normalize_value(value) -> str:
    text = " ".join(str(value).split()).lower()
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text
    return f"{number:.4f}" if math.isfinite(number) else text


def flatten(obj, path: str = "") -> list[tuple[str, str]]:
    """Leaf (field path, value) pairs. List items share their parent's path, so order does not matter."""
    if isinstance(obj, dict):
        items = []
        for key, value in obj.items():
            items += flatten(value, f"{path}.{key}" if path else str(key))
        return items
    if isinstance(obj, list):
        items = []
        for value in obj:
            items += flatten(value, path)
        return items
    return [(path, normalize_value(obj))]


def schema_prompt(train_rows) -> str:
    """Base-model baseline prompt. It lists the field paths seen in the train split, never the test split."""
    paths = sorted({path for row in train_rows for path, _ in flatten(json.loads(row["ground_truth"])["gt_parse"])})
    return PROMPT + "\nFields (dot-separated paths; a dot separates nested keys):\n" + "\n".join(paths)


def field_scores(pred: dict, gold: dict) -> tuple[float, bool]:
    """Field-level F1 and exact match over multisets of (path, value) pairs."""
    p, g = Counter(flatten(pred)), Counter(flatten(gold))
    tp = sum((p & g).values())
    n_pred, n_gold = sum(p.values()), sum(g.values())
    if n_pred == 0 or n_gold == 0 or tp == 0:
        return 0.0, p == g
    precision, recall = tp / n_pred, tp / n_gold
    return 2 * precision * recall / (precision + recall), p == g


def parse_prediction(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def mean_se(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


class ReceiptDataset(torch.utils.data.Dataset):
    """Each item is one receipt image with the gold JSON as the assistant response (prompt tokens masked)."""

    def __init__(self, rows, processor, prompt_text: str):
        self.rows = rows
        self.processor = processor
        self.prompt_text = prompt_text

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        enc = self.processor(text=[self.prompt_text], images=[as_rgb(row["image"])], return_tensors="pt")
        gold = json.loads(row["ground_truth"])["gt_parse"]
        target_ids = torch.tensor(
            self.processor.tokenizer(
                json.dumps(gold, ensure_ascii=False) + "<|im_end|>", add_special_tokens=False
            )["input_ids"]
        )
        prompt_ids = enc["input_ids"][0]
        return {
            "input_ids": torch.cat([prompt_ids, target_ids]),
            "labels": torch.cat([torch.full_like(prompt_ids, -100), target_ids]),
            # Qwen3-VL needs this to compute multimodal RoPE; 0 marks text tokens.
            "mm_token_type_ids": torch.cat([enc["mm_token_type_ids"][0], torch.zeros_like(target_ids)]),
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    width = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, fill: int) -> torch.Tensor:
        out = torch.full((len(batch), width), fill, dtype=batch[0][key].dtype)
        for i, b in enumerate(batch):
            out[i, : len(b[key])] = b[key]
        return out

    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, b in enumerate(batch):
        attention_mask[i, : len(b["input_ids"])] = 1
    return {
        "input_ids": pad("input_ids", pad_token_id),
        "attention_mask": attention_mask,
        "labels": pad("labels", -100),
        "mm_token_type_ids": pad("mm_token_type_ids", 0),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
    }


@torch.no_grad()
def validation_loss(model, dataset: ReceiptDataset, pad_token_id: int, batch_size: int) -> float:
    model.eval()
    losses = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        batch = {k: v.to(model.device) for k, v in collate(items, pad_token_id).items()}
        loss = model(**batch).loss.item()
        if not math.isfinite(loss):
            raise RuntimeError(f"Non-finite validation loss: {loss}")
        losses.append(loss)
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_texts(model, processor, rows, prompt_text: str, batch_size: int, max_new_tokens: int) -> list[str]:
    model.eval()
    processor.tokenizer.padding_side = "left"
    texts = []
    for start in range(0, len(rows), batch_size):
        # Slicing a datasets.Dataset returns a dict of columns, so take rows one by one.
        chunk = [rows[i] for i in range(start, min(start + batch_size, len(rows)))]
        enc = processor(
            text=[prompt_text] * len(chunk),
            images=[as_rgb(r["image"]) for r in chunk],
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        texts += processor.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"generated {len(texts)}/{len(rows)}", flush=True)
    return texts


def score(texts: list[str], rows) -> list[dict]:
    scored = []
    for text, row in zip(texts, rows):
        gold = json.loads(row["ground_truth"])["gt_parse"]
        pred = parse_prediction(text)
        f1, exact = field_scores(pred or {}, gold)
        scored.append({"prediction": text, "parsed": pred is not None, "field_f1": f1, "exact": exact})
    return scored


def summarize(name: str, scored: list[dict]) -> dict:
    f1_mean, f1_se = mean_se([s["field_f1"] for s in scored])
    em_mean, em_se = mean_se([float(s["exact"]) for s in scored])
    parse_mean, _ = mean_se([float(s["parsed"]) for s in scored])
    return {
        f"{name}_field_f1": f1_mean,
        f"{name}_field_f1_se": f1_se,
        f"{name}_exact_match": em_mean,
        f"{name}_exact_match_se": em_se,
        f"{name}_parse_rate": parse_mean,
    }


def write_predictions(path: str, scored: list[dict]) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(scored):
            f.write(json.dumps({"index": i, **s}, ensure_ascii=False) + "\n")


def main() -> None:
    p = load_params()
    model_name = p.get("model_name", "Qwen/Qwen3-VL-2B-Instruct")
    dataset_name = p.get("dataset_name", "naver-clova-ix/cord-v2")
    min_pixels = int(p.get("min_pixels", 65536))
    max_pixels = int(p.get("max_pixels", 768 * 28 * 28))
    max_train_samples = int(p.get("max_train_samples", -1))
    max_eval_samples = int(p.get("max_eval_samples", -1))
    max_steps = int(p.get("max_steps", -1))
    epochs = float(p.get("epochs", 2))
    per_device_batch_size = int(p.get("per_device_batch_size", 2))
    grad_accum = int(p.get("grad_accum", 4))
    learning_rate = float(p.get("learning_rate", 2e-4))
    lora_r = int(p.get("lora_r", 16))
    lora_alpha = int(p.get("lora_alpha", 32))
    lora_dropout = float(p.get("lora_dropout", 0.05))
    max_new_tokens = int(p.get("max_new_tokens", 512))
    eval_batch_size = int(p.get("eval_batch_size", 16))
    seed = int(p.get("seed", 42))
    output_root = p["output_root"]

    set_seed(seed)
    print("CUDA:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", flush=True)

    # AIR creates this run for the attempt. Pop it so the Trainer's MLflow callback logs into the
    # run we open below instead of starting one; no fallback run is created.
    run_id = os.environ.pop("MLFLOW_RUN_ID")
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    train_rows = load_dataset(dataset_name, split="train")
    val_rows = load_dataset(dataset_name, split="validation")
    test_rows = load_dataset(dataset_name, split="test")
    # Smoke-test subsets take the first N rows; the default run uses every row.
    if max_train_samples > 0:
        train_rows = train_rows.select(range(min(max_train_samples, len(train_rows))))
    if max_eval_samples > 0:
        test_rows = test_rows.select(range(min(max_eval_samples, len(test_rows))))
        val_rows = val_rows.select(range(min(max_eval_samples, len(val_rows))))
    print(f"train={len(train_rows)} validation={len(val_rows)} test={len(test_rows)}", flush=True)

    processor = load_processor(model_name, min_pixels, max_pixels)
    print("image size bounds:", processor.image_processor.size, flush=True)
    def build_prompt(text: str) -> str:
        return processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Training and the tuned evaluation use PROMPT. The schema prompt is used only by the base baseline.
    prompt_text = build_prompt(PROMPT)
    schema_prompt_text = build_prompt(schema_prompt(train_rows))
    pad_token_id = processor.tokenizer.pad_token_id

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_params(
            {
                "recipe_model": model_name,
                "recipe_dataset": dataset_name,
                "recipe_min_pixels": min_pixels,
                "recipe_max_pixels": max_pixels,
                "recipe_lora_r": lora_r,
                "recipe_lora_alpha": lora_alpha,
                "recipe_lora_target": "language_model attention+MLP",
                "recipe_n_train": len(train_rows),
                "recipe_n_validation": len(val_rows),
                "recipe_n_test": len(test_rows),
                "recipe_max_new_tokens": max_new_tokens,
            }
        )

        # Before: the base model with the same prompt, decoding and scorer as the tuned model.
        base_val_loss = validation_loss(
            model, ReceiptDataset(val_rows, processor, prompt_text), pad_token_id, eval_batch_size // 4
        )
        base_texts = generate_texts(model, processor, test_rows, prompt_text, eval_batch_size, max_new_tokens)
        base_scored = score(base_texts, test_rows)
        write_predictions(os.path.join(output_dir, "predictions_base.jsonl"), base_scored)
        metrics = summarize("base", base_scored)
        metrics["base_val_loss"] = base_val_loss
        mlflow.log_metrics(metrics)
        print("base:", metrics, flush=True)

        # Base with the schema prompt: the fairer baseline, since it names the fields the tuned model learns.
        schema_texts = generate_texts(model, processor, test_rows, schema_prompt_text, eval_batch_size, max_new_tokens)
        schema_scored = score(schema_texts, test_rows)
        write_predictions(os.path.join(output_dir, "predictions_base_schema.jsonl"), schema_scored)
        schema_metrics = summarize("base_schema", schema_scored)
        mlflow.log_metrics(schema_metrics)
        print("base_schema:", schema_metrics, flush=True)

        # Train: LoRA on the language model. Gradient checkpointing with use_reentrant=False works
        # with frozen vision inputs, so no enable_input_require_grads() call is needed.
        model.config.use_cache = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=LORA_TARGET_REGEX,
                bias="none",
            ),
        )
        model.print_trainable_parameters()

        train_dataset = ReceiptDataset(train_rows, processor, prompt_text)
        args = TrainingArguments(
            output_dir="/tmp/qwen3vl-lora-trainer",
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=0.05,  # float in [0,1) = ratio of total steps (transformers 5.x dropped warmup_ratio)
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5,
            save_strategy="no",
            eval_strategy="no",
            report_to=["mlflow"],
            remove_unused_columns=False,
            dataloader_num_workers=2,
            seed=seed,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=lambda batch: collate(batch, pad_token_id),
        )
        trainer.train()

        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise RuntimeError(f"Training losses are missing or non-finite: {losses}")
        print(f"train loss first={losses[0]:.4f} last={losses[-1]:.4f} logged={len(losses)}", flush=True)

        adapter_dir = os.path.join(output_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        print("saved adapter to", adapter_dir, flush=True)

        # After: the same evaluation on the same test rows.
        model.config.use_cache = True
        tuned_val_loss = validation_loss(
            model, ReceiptDataset(val_rows, processor, prompt_text), pad_token_id, eval_batch_size // 4
        )
        tuned_texts = generate_texts(model, processor, test_rows, prompt_text, eval_batch_size, max_new_tokens)
        tuned_scored = score(tuned_texts, test_rows)
        write_predictions(os.path.join(output_dir, "predictions_tuned.jsonl"), tuned_scored)
        metrics = summarize("tuned", tuned_scored)
        metrics["tuned_val_loss"] = tuned_val_loss
        mlflow.log_metrics(metrics)
        print("tuned:", metrics, flush=True)

        # Paired difference: each receipt is scored under both models, so the SE of the difference
        # is the SE of the per-receipt differences, not a combination of two independent SEs.
        diff_f1 = [t["field_f1"] - b["field_f1"] for t, b in zip(tuned_scored, base_scored)]
        diff_em = [float(t["exact"]) - float(b["exact"]) for t, b in zip(tuned_scored, base_scored)]
        diff_f1_mean, diff_f1_se = mean_se(diff_f1)
        diff_em_mean, diff_em_se = mean_se(diff_em)
        schema_diff_f1 = [t["field_f1"] - b["field_f1"] for t, b in zip(tuned_scored, schema_scored)]
        schema_diff_em = [float(t["exact"]) - float(b["exact"]) for t, b in zip(tuned_scored, schema_scored)]
        schema_diff_f1_mean, schema_diff_f1_se = mean_se(schema_diff_f1)
        schema_diff_em_mean, schema_diff_em_se = mean_se(schema_diff_em)
        final = {
            "diff_field_f1": diff_f1_mean,
            "diff_field_f1_se": diff_f1_se,
            "diff_exact_match": diff_em_mean,
            "diff_exact_match_se": diff_em_se,
            "diff_schema_field_f1": schema_diff_f1_mean,
            "diff_schema_field_f1_se": schema_diff_f1_se,
            "diff_schema_exact_match": schema_diff_em_mean,
            "diff_schema_exact_match_se": schema_diff_em_se,
            "diff_val_loss": tuned_val_loss - base_val_loss,
        }
        mlflow.log_metrics(final)
        print("difference (tuned - base):", final, flush=True)

        mlflow.log_artifact(os.path.join(output_dir, "predictions_base.jsonl"))
        mlflow.log_artifact(os.path.join(output_dir, "predictions_base_schema.jsonl"))
        mlflow.log_artifact(os.path.join(output_dir, "predictions_tuned.jsonl"))


if __name__ == "__main__":
    main()
