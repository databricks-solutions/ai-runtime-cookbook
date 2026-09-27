#!/usr/bin/env python3
"""Distributed audio transcription + speaker diarization with Ray Data.

Runs on a Ray cluster that `ray_bootstrap.sh` forms across the workload's GPU
node(s). Ray Data's `map_batches` launches one GPU actor per GPU; each actor
loads faster-whisper (ASR) and pyannote (diarization) once, then pulls audio
clips off Ray's shared work queue. That dynamic scheduling rebalances stragglers
across GPUs on its own — long clips don't leave fast GPUs idle.

Pipeline per clip: faster-whisper transcribes -> pyannote assigns speaker turns.
Results (transcript + speaker-turn count + timing) are written to a Unity Catalog
volume as Parquet.

Input audio is staged once on the Ray head from the public, ungated
`diarizers-community/voxconverse` multi-speaker dataset (CC-BY-4.0), written as
WAVs to a shared UC volume so every node can read them.

`pyannote/speaker-diarization-3.1` is gated: set HF_TOKEN (via a Databricks
secret in workload.yaml) to a token that has accepted the model's terms.
"""

import os
import time

import ray

# HF caches must live on a writable path (the default /root/.cache is read-only
# on serverless/AIR workers). Set before any HF import.
os.environ.setdefault("HF_HOME", "/tmp/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_home/datasets")

MODEL_SOURCE = os.environ.get("WHISPER_MODEL", "large-v3")
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
NUM_CLIPS = int(os.environ.get("NUM_CLIPS", "24"))
TARGET_SR = 16000
# Shared UC volume paths. Set these in workload.yaml; the volumes must exist.
INPUT_DIR = os.environ.get("INPUT_DIR", "/Volumes/main/default/air_examples/voxconverse_wavs")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/Volumes/main/default/air_examples/audio_diarization_results")
HF_TOKEN = os.environ.get("HF_TOKEN")


def stage_audio() -> list:
    """Download a slice of the public dataset and write clips as WAVs to INPUT_DIR.

    Runs once on the Ray head. Idempotent: clips already present are not rewritten.
    Returns the list of {"wav_path", "call_id"} items for the Ray Dataset.
    """
    from datasets import Audio, load_dataset

    os.makedirs(INPUT_DIR, exist_ok=True)
    ds = load_dataset(
        "diarizers-community/voxconverse",
        split=f"dev[:{NUM_CLIPS}]",
        cache_dir="/tmp/hf_home/datasets",
    )
    # decode=False keeps the audio as {"path", "bytes"} so we can write the
    # original file straight to disk — no torchcodec/ffmpeg decode step needed.
    # faster-whisper and pyannote decode the files themselves.
    ds = ds.cast_column("audio", Audio(decode=False))
    import shutil

    items = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        src_name = audio.get("path") or f"clip_{i:04d}.wav"
        ext = os.path.splitext(src_name)[1] or ".wav"
        base = os.path.splitext(os.path.basename(src_name))[0] or "clip"
        # Prefix with the index so clips that share a basename don't collide.
        call_id = f"{i:04d}_{base}"
        wav_path = f"{INPUT_DIR}/{call_id}{ext}"
        if not os.path.exists(wav_path):
            if audio.get("bytes") is not None:
                with open(wav_path, "wb") as f:
                    f.write(audio["bytes"])
            elif audio.get("path") and os.path.exists(audio["path"]):
                # decode=False can return a path reference with bytes=None.
                shutil.copyfile(audio["path"], wav_path)
            else:
                print(f"  skipping {call_id}: no audio bytes or readable path", flush=True)
                continue
        items.append({"wav_path": wav_path, "call_id": call_id})
    print(f"Staged {len(items)} clips to {INPUT_DIR}", flush=True)
    return items


class WhisperDiarizer:
    """GPU actor: transcribe with faster-whisper, then diarize with pyannote."""

    def __init__(self):
        import torch
        from faster_whisper import BatchedInferencePipeline, WhisperModel
        from pyannote.audio import Pipeline as DiarPipeline

        # Actors run in their own process (and, on multi-node, other hosts), so
        # re-check the token here rather than trusting the driver-only guard.
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN is not set in this actor's environment; ensure the "
                "workload secret is injected on every node."
            )

        self.asr = BatchedInferencePipeline(
            WhisperModel(MODEL_SOURCE, device="cuda", compute_type="float16")
        )
        # pyannote 3.1 checkpoints load pickled objects; torch>=2.6 defaults
        # weights_only=True, which rejects them. Scope the override to this load
        # and restore it, rather than weakening torch.load for the whole process.
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        try:
            self.diarizer = DiarPipeline.from_pretrained(
                DIARIZATION_MODEL, use_auth_token=HF_TOKEN
            ).to(torch.device("cuda"))
        finally:
            torch.load = _orig_load

    def __call__(self, batch: dict) -> dict:
        # batch_size=1 -> one clip per call; columns arrive as length-1 arrays.
        wav_path = batch["wav_path"][0]
        call_id = batch["call_id"][0]
        t0 = time.time()

        segments, info = self.asr.transcribe(wav_path, batch_size=16, vad_filter=True)
        segments = list(segments)
        transcript = " ".join(s.text.strip() for s in segments).strip()

        diarization = self.diarizer(wav_path)
        turns = list(diarization.itertracks(yield_label=True))
        speakers = sorted({label for _, _, label in turns})

        return {
            "call_id": [call_id],
            "audio_duration_s": [round(float(info.duration), 2)],
            "num_segments": [len(segments)],
            "num_speakers": [len(speakers)],
            "num_speaker_turns": [len(turns)],
            "transcript": [transcript],
            "processing_secs": [round(time.time() - t0, 2)],
        }


def main():
    if not HF_TOKEN:
        raise SystemExit(
            "HF_TOKEN is not set. Provide a Hugging Face token (via a Databricks "
            "secret in workload.yaml) that has accepted the terms for "
            f"{DIARIZATION_MODEL}."
        )

    ray.init(address="auto")
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus < 1:
        raise SystemExit("No GPUs found in the Ray cluster; cannot run inference.")
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    items = stage_audio()
    ds = ray.data.from_items(items)

    # One actor per GPU; Ray Data's queue hands each free actor the next clip,
    # so total time tracks the busiest GPU, not the slowest static shard.
    out = ds.map_batches(
        WhisperDiarizer,
        concurrency=total_gpus,
        num_gpus=1,
        batch_size=1,
    ).materialize()

    out.write_parquet(OUTPUT_PATH)
    print(f"Wrote {out.count()} rows to {OUTPUT_PATH}", flush=True)
    for row in out.take(2):
        print(
            f"[{row['call_id']}] {row['num_speakers']} speakers, "
            f"{row['num_segments']} segments, {row['processing_secs']}s: "
            f"{row['transcript'][:160]}...",
            flush=True,
        )

    ray.shutdown()


if __name__ == "__main__":
    main()
