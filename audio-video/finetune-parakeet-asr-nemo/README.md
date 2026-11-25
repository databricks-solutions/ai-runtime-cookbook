# Fine-tune Parakeet speech recognition with NVIDIA NeMo Speech

This recipe fine-tunes [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a 600-million-parameter automatic speech recognition (ASR) model, on conversational English from the [AMI Meeting Corpus](https://huggingface.co/datasets/edinburghcstr/ami) with [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) 3.0.0 on one 24 GB A10 GPU. It measures word error rate (WER) on a held-out test subset before and after fine-tuning, using the same text normalization for both, and selects the checkpoint on a separate validation subset.

Parakeet is trained on a large general-purpose corpus, but meeting speech contains disfluencies (such as "um" and "mm-hmm"), overlapping and casual phrasing, and transcript conventions the pretrained model does not reproduce. Its WER on this data is typically higher than on read speech. The recipe fine-tunes for 900 steps on 3,000 utterances, a demonstration budget chosen to fit the time limit rather than to reach the best achievable WER.

## Before you run

- **Licenses.** The model is licensed under [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) and the AMI corpus under [CC-BY-4.0](https://huggingface.co/datasets/edinburghcstr/ami). Both are public and ungated. No Hugging Face token is needed. Keep the attribution when you reuse the outputs.
- **Text style.** AMI transcripts are uppercase and have no punctuation, while Parakeet emits cased, punctuated text. The recipe lowercases all text, removes punctuation except apostrophes, and scores both before and after fine-tuning on that normalized text. As a consequence, the fine-tuned model produces lowercase text without punctuation. Change `normalize` in `finetune_parakeet.py` if your application needs cased output.
- **Data.** The recipe downloads only the Parquet shards it samples (about 2 GB) from the Hub's auto-converted branch, then decodes and writes WAV files to local scratch space. Nothing is copied into the output volume except the results.
- **Storage.** Results go to `parameters.output_root` in `workload.yaml`, which points to the shared `/Volumes/main/default/air_examples/finetune-parakeet-asr-nemo` volume. Replace it with another writable volume if you cannot use that one. The fine-tuned checkpoint is about 2.5 GB.

To write to another volume, override `parameters.output_root` when you submit the run:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

Each run writes to `<output_root>/<MLFLOW_RUN_ID>/`:

- `finetuned.nemo`: the model with the lowest normalized validation WER, saved with `save_to`. Load it in a fresh process with `ASRModel.restore_from`.
- `metrics.json`: utterance counts, hyperparameters, the selected step, the validation WER at each check, and baseline and fine-tuned validation and test WER with their bootstrap standard errors.
- `test_predictions.jsonl`: each test reference alongside the baseline and fine-tuned transcriptions.

After training, `load_finetuned.py` restores `finetuned.nemo` in a new process and transcribes one test utterance. To load the checkpoint yourself, use the same code:

```python
"""Load the fine-tuned Parakeet checkpoint in a fresh process and transcribe one test clip."""

import json
import os

import torch
import yaml
from nemo.collections.asr.models import ASRModel

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

checkpoint = f"{P['output_root']}/{os.environ['MLFLOW_RUN_ID']}/finetuned.nemo"
model = ASRModel.restore_from(checkpoint, map_location=torch.device("cuda")).eval()

# The first test utterance from the training run's scratch manifest, with its normalized reference.
with open("/tmp/ami_parakeet_data/test_manifest.json") as f:
    record = json.loads(f.readline())
with torch.inference_mode():
    hypothesis = model.transcribe([record["audio_filepath"]], batch_size=1, verbose=False)[0]
print("Reference:", record["text"])
print("Fine-tuned transcription:", hypothesis.text)
```

The AIR MLflow run logs the training loss at each step and the WER values with their standard errors as metrics. It also stores `metrics.json`, `test_predictions.jsonl`, and the NeMo training logs as artifacts.

On the normalized scores in the validated run (`GPU_1xA10`, about 15 minutes in the container), the test improvement is well above its standard error:

| Split | Pretrained | Fine-tuned (step 900) |
| --- | --- | --- |
| Validation (500 utterances) | 0.1025 ± 0.0080 | 0.0812 ± 0.0059 |
| Test (1,000 utterances) | 0.0928 ± 0.0049 | 0.0758 ± 0.0037 |

The test WER improvement, 0.0170, is about 4.5 times the standard error of the paired difference, 0.0037. Validation WER fell from 0.0937 at step 300 to 0.0846 at step 600, and then to 0.0812 at step 900, the last check and therefore the selected checkpoint. Standard errors are bootstrap estimates over utterances. They reflect the sampled subsets rather than the full AMI splits. The validation and test sets are random subsets, capped at 500 and 1,000 utterances to keep scoring within the time limit.
