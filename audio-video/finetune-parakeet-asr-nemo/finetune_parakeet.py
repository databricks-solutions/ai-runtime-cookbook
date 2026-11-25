"""Fine-tune NVIDIA Parakeet TDT 0.6B v3 on AMI meeting speech with NVIDIA NeMo Speech.

The script samples AMI individual-headset utterances into NeMo JSON-lines manifests, scores
the pretrained model, fine-tunes it for a fixed number of steps while selecting the checkpoint
by validation WER, and scores the selected checkpoint on a held-out test subset. Every score
uses the same text normalizer, so the before and after numbers are comparable.
"""

import io
import json
import os
import random
import re
import statistics
import time
from pathlib import Path

import lightning.pytorch as pl
import mlflow
import pyarrow.parquet as pq
import soundfile as sf
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from kaldialign import edit_distance
from nemo.collections.asr.metrics.wer import word_error_rate as nemo_word_error_rate
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf

DATASET_ID = "edinburghcstr/ami"
DATASET_CONFIG = "ihm"  # individual headset microphone: one speaker per channel
# AMI's Parquet files are published on the Hub's auto-converted branch.
PARQUET_REVISION = "refs/convert/parquet"
SAMPLE_RATE = 16000
NON_TEXT = re.compile(r"[^a-z0-9' ]+")


def normalize(text):
    # AMI transcripts are uppercase without punctuation; Parakeet emits cased, punctuated text.
    # Lowercasing and dropping everything but letters, digits and apostrophes lets both sides meet.
    return " ".join(NON_TEXT.sub(" ", text.lower()).split())


def per_utterance_errors(refs, hyps):
    # Kept per utterance so the bootstrap standard error can resample utterances; this is the
    # same measure NeMo's word_error_rate aggregates, so the two stay consistent.
    return [edit_distance(r.split(), h.split())["total"] for r, h in zip(refs, hyps)]


def wer_standard_errors(baseline_errors, finetuned_errors, words, n_resamples=1000, seed=0):
    """Bootstrap standard errors of the baseline WER, the fine-tuned WER and their difference.

    Both models are scored on the same utterances, so resampling is paired: each resample
    draws utterance indices once and scores both models on them, which keeps the utterance
    difficulty that cancels in the difference.
    """
    rng = random.Random(seed)
    n = len(words)
    baselines, finetuned, differences = [], [], []
    for _ in range(n_resamples):
        index = [rng.randrange(n) for _ in range(n)]
        total_words = sum(words[i] for i in index)
        if not total_words:
            continue
        baseline_wer = sum(baseline_errors[i] for i in index) / total_words
        finetuned_wer = sum(finetuned_errors[i] for i in index) / total_words
        baselines.append(baseline_wer)
        finetuned.append(finetuned_wer)
        differences.append(baseline_wer - finetuned_wer)
    return statistics.stdev(baselines), statistics.stdev(finetuned), statistics.stdev(differences)


def list_shards(split):
    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset", revision=PARQUET_REVISION)
    pattern = re.compile(rf"{DATASET_CONFIG}/{split}/\d{{4}}\.parquet")
    shards = sorted(name for name in files if pattern.fullmatch(name))
    if not shards:
        raise RuntimeError(f"No {DATASET_CONFIG}/{split} Parquet shards found in {DATASET_ID}")
    return shards


def build_split(split, num_shards, num_utterances, min_duration, max_duration, data_dir, rng):
    shard_names = rng.sample(list_shards(split), num_shards)
    tables = [
        pq.read_table(
            hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=PARQUET_REVISION),
            columns=["text", "audio"],
        )
        for name in shard_names
    ]
    texts = [table.column("text").to_pylist() for table in tables]
    audio = [table.column("audio").to_pylist() for table in tables]
    pool = [
        (s, r)
        for s, column in enumerate(texts)
        for r in range(len(column))
        if normalize(column[r])
    ]
    rng.shuffle(pool)

    split_dir = data_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for s, r in pool:
        if len(records) == num_utterances:
            break
        samples, rate = sf.read(io.BytesIO(audio[s][r]["bytes"]), dtype="float32")
        duration = len(samples) / rate
        if not min_duration <= duration <= max_duration:
            continue
        path = split_dir / f"{len(records):05d}.wav"
        sf.write(path, samples, rate, subtype="PCM_16")
        records.append(
            {"audio_filepath": str(path), "duration": round(duration, 3), "text": normalize(texts[s][r])}
        )
    if len(records) < num_utterances:
        raise RuntimeError(f"{split}: only {len(records)} usable utterances of {num_utterances} requested")
    return records


def write_manifest(records, path):
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def load_model(name):
    return ASRModel.from_pretrained(model_name=name).to("cuda")


def score(model, records, batch_size):
    model.eval()
    with torch.inference_mode():
        outputs = model.transcribe(
            [record["audio_filepath"] for record in records], batch_size=batch_size, verbose=False
        )
    # NeMo 3.0.0's transcribe returns a list of Hypothesis objects.
    hyps = [normalize(output.text) for output in outputs]
    refs = [record["text"] for record in records]
    return nemo_word_error_rate(hyps, refs), hyps


class BestValidationWeights(pl.Callback):
    """Keeps an in-memory copy of the weights with the lowest normalized validation WER.

    NeMo's logged val_wer compares cased, punctuated hypotheses with normalized references, so it
    overstates errors; selection uses the same score as the reported metrics instead.
    """

    def __init__(self, validation_records, batch_size):
        self.validation_records = validation_records
        self.batch_size = batch_size
        self.best_wer = float("inf")
        self.best_step = None
        self.history = []
        self.state = None

    def on_validation_end(self, trainer, pl_module):
        was_training = pl_module.training
        wer, _ = score(pl_module, self.validation_records, self.batch_size)
        pl_module.train(was_training)
        self.history.append({"step": trainer.global_step, "validation_wer": wer})
        print(f"Step {trainer.global_step}: normalized validation WER {wer:.4f}", flush=True)
        if wer < self.best_wer:
            self.best_wer = wer
            self.best_step = trainer.global_step
            self.state = {key: value.detach().to("cpu", copy=True) for key, value in pl_module.state_dict().items()}


def train(p, model, train_manifest, val_manifest, validation_records, exp_dir):
    best_weights = BestValidationWeights(validation_records, p["eval_batch_size"])
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        max_steps=p["max_steps"],
        max_epochs=-1,
        val_check_interval=p["val_check_interval"],
        check_val_every_n_epoch=None,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[best_weights],
    )
    # exp_manager writes NeMo logs under exp_dir (local scratch). Its checkpoint callback is off:
    # best weights are kept in memory instead, so no checkpoint is written or reloaded during training.
    # TensorBoard and W&B are off; metrics go to MLflow after scoring.
    exp_manager(
        trainer,
        OmegaConf.create(
            {
                "exp_dir": str(exp_dir),
                "name": "parakeet_ami",
                "create_tensorboard_logger": False,
                "create_wandb_logger": False,
                "create_checkpoint_callback": False,
                "resume_if_exists": False,
            }
        ),
    )
    model.set_trainer(trainer)
    model.setup_training_data(
        OmegaConf.create(
            {
                "manifest_filepath": train_manifest,
                "sample_rate": SAMPLE_RATE,
                "batch_size": p["batch_size"],
                "shuffle": True,
                "num_workers": p["num_workers"],
                "pin_memory": True,
                "min_duration": p["min_duration"],
                "max_duration": p["max_duration"],
            }
        )
    )
    model.setup_multiple_validation_data(
        OmegaConf.create(
            {
                "manifest_filepath": val_manifest,
                "sample_rate": SAMPLE_RATE,
                "batch_size": p["eval_batch_size"],
                "shuffle": False,
                "num_workers": p["num_workers"],
                "pin_memory": True,
            }
        )
    )
    model.setup_optimization(
        OmegaConf.create(
            {
                "name": "adamw",
                "lr": p["learning_rate"],
                "betas": [0.9, 0.98],
                "weight_decay": 1e-3,
                "sched": {
                    "name": "CosineAnnealing",
                    "warmup_steps": p["warmup_steps"],
                    "warmup_ratio": None,
                    "min_lr": p["min_learning_rate"],
                },
            }
        )
    )
    trainer.fit(model)
    # Lightning moves the model back to the CPU in teardown; final scoring runs on the GPU.
    model.to("cuda")
    return best_weights


def log_to_mlflow(metrics, artifacts):
    # Reopens the AIR-provided run; a logging failure raises and fails the workload.
    with mlflow.start_run(run_id=os.environ["MLFLOW_RUN_ID"]):
        mlflow.log_params(
            {key: metrics[key] for key in ("model", "dataset", "learning_rate", "max_steps", "batch_size")}
        )
        mlflow.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if key.endswith(("_wer", "_stderr", "_seconds"))
            }
        )
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))


def main():
    with open(os.environ["HYPERPARAMETERS_PATH"]) as handle:
        p = yaml.safe_load(handle)
    # Small floats may arrive as "5e-05", which PyYAML parses as a string.
    for key in ("learning_rate", "min_learning_rate", "min_duration", "max_duration"):
        p[key] = float(p[key])
    rng = random.Random(p["seed"])
    torch.manual_seed(p["seed"])

    run_id = os.environ["MLFLOW_RUN_ID"]
    output_dir = Path(p["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/tmp/ami_parakeet_data")
    exp_dir = Path("/tmp/parakeet_ami_nemo_experiments")

    print("Sampling AMI splits", flush=True)
    validation = build_split(
        "validation", p["validation_shards"], p["validation_utterances"],
        p["min_duration"], p["max_duration"], data_dir, rng,
    )
    test = build_split(
        "test", p["test_shards"], p["test_utterances"], p["min_duration"], p["max_duration"], data_dir, rng,
    )
    train_records = build_split(
        "train", p["train_shards"], p["train_utterances"], p["min_duration"], p["max_duration"], data_dir, rng,
    )
    manifests = {}
    for name, records in (("train", train_records), ("validation", validation), ("test", test)):
        manifests[name] = data_dir / f"{name}_manifest.json"
        write_manifest(records, manifests[name])
    print(
        f"Utterances: train={len(train_records)} validation={len(validation)} test={len(test)}",
        flush=True,
    )

    # One model object serves baseline scoring, fine-tuning and final scoring, so the .nemo
    # archive is never reloaded (see BestValidationWeights).
    print("Scoring the pretrained model", flush=True)
    model = load_model(p["model_name"])
    baseline_validation_wer, baseline_validation_hyps = score(model, validation, p["eval_batch_size"])
    baseline_test_wer, baseline_hyps = score(model, test, p["eval_batch_size"])
    print(f"Baseline WER: validation={baseline_validation_wer:.4f} test={baseline_test_wer:.4f}", flush=True)

    print("Fine-tuning", flush=True)
    started = time.perf_counter()
    best_weights = train(
        p, model, str(manifests["train"]), str(manifests["validation"]), validation, exp_dir
    )
    train_seconds = time.perf_counter() - started
    print(
        f"Selected step {best_weights.best_step} with validation WER {best_weights.best_wer:.4f}",
        flush=True,
    )
    if best_weights.state is None:
        raise RuntimeError("Training finished without a validation pass to select weights from")
    model.load_state_dict(best_weights.state)

    finetuned_path = output_dir / "finetuned.nemo"
    model.save_to(str(finetuned_path))
    finetuned_validation_wer, finetuned_validation_hyps = score(model, validation, p["eval_batch_size"])
    finetuned_test_wer, tuned_hyps = score(model, test, p["eval_batch_size"])
    print(f"Fine-tuned WER: validation={finetuned_validation_wer:.4f} test={finetuned_test_wer:.4f}", flush=True)

    # Standard errors by bootstrap over utterances, for the sampled splits the README reports.
    standard_errors = {}
    for name, refs, baseline_hyps_split, finetuned_hyps_split in (
        ("validation", [record["text"] for record in validation], baseline_validation_hyps, finetuned_validation_hyps),
        ("test", [record["text"] for record in test], baseline_hyps, tuned_hyps),
    ):
        words = [len(record.split()) for record in refs]
        baseline_errors = per_utterance_errors(refs, baseline_hyps_split)
        finetuned_errors = per_utterance_errors(refs, finetuned_hyps_split)
        baseline_se, finetuned_se, difference_se = wer_standard_errors(
            baseline_errors, finetuned_errors, words, seed=p["seed"]
        )
        standard_errors[f"baseline_{name}_wer_stderr"] = baseline_se
        standard_errors[f"finetuned_{name}_wer_stderr"] = finetuned_se
        standard_errors[f"{name}_wer_difference_stderr"] = difference_se
    test_improvement = baseline_test_wer - finetuned_test_wer
    print(
        f"Test WER improvement {test_improvement:.4f} vs standard error of the difference "
        f"{standard_errors['test_wer_difference_stderr']:.4f}",
        flush=True,
    )

    predictions = output_dir / "test_predictions.jsonl"
    with open(predictions, "w") as handle:
        for record, before, after in zip(test, baseline_hyps, tuned_hyps):
            handle.write(json.dumps({"reference": record["text"], "baseline": before, "finetuned": after}) + "\n")

    metrics = {
        "model": p["model_name"],
        "dataset": f"{DATASET_ID} ({DATASET_CONFIG})",
        "text_normalization": "lowercase; punctuation other than apostrophes removed; hyphens become spaces",
        "train_utterances": len(train_records),
        "validation_utterances": len(validation),
        "test_utterances": len(test),
        "max_steps": p["max_steps"],
        "batch_size": p["batch_size"],
        "learning_rate": p["learning_rate"],
        "selected_step": best_weights.best_step,
        "validation_wer_history": best_weights.history,
        "baseline_validation_wer": baseline_validation_wer,
        "baseline_test_wer": baseline_test_wer,
        "finetuned_validation_wer": finetuned_validation_wer,
        "finetuned_test_wer": finetuned_test_wer,
        **standard_errors,
        "train_seconds": train_seconds,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    log_to_mlflow(metrics, [metrics_path, predictions])


if __name__ == "__main__":
    main()
