# Audio transcription + diarization (Whisper + pyannote, Ray Data)

Distributed speech-to-text with speaker diarization on a Ray cluster. Ray Data runs one
GPU actor per GPU — each loads faster-whisper (ASR) and pyannote (diarization) once and
pulls clips off Ray's shared queue, so long clips don't strand fast GPUs. Results are
written to a Unity Catalog volume as Parquet.

Input is the public, ungated `diarizers-community/voxconverse` multi-speaker dataset
(CC-BY-4.0); the entrypoint stages `NUM_CLIPS` clips to `INPUT_DIR` on the Ray head, so
no separate prep step is needed. Point `INPUT_DIR` at your own WAVs to run on other audio.

## Prerequisites

1. Writable Unity Catalog volumes for `INPUT_DIR` (staged WAVs) and `OUTPUT_PATH`
   (result Parquet); they must already exist and may be the same volume.
2. **Hugging Face token for pyannote.** `pyannote/speaker-diarization-3.1` is gated:
   accept its terms once at https://huggingface.co/pyannote/speaker-diarization-3.1,
   store the token in a Databricks secret, and set `secrets.HF_TOKEN` in `workload.yaml`
   to that `scope/key`.

## Run

```bash
databricks air run -f workload.yaml
```

Configure `NUM_CLIPS`, `INPUT_DIR`, `OUTPUT_PATH`, and the `HF_TOKEN` secret in
`workload.yaml`. Add nodes by raising `compute.num_accelerators` — `ray_bootstrap.sh`
forms the cluster and Ray Data scales the actor pool to the available GPUs.
