# Fine-tune the SmolVLA robot policy with LeRobot

This recipe fine-tunes `lerobot/smolvla_base`, a 450M-parameter vision-language-action (VLA) policy, with [Hugging Face LeRobot](https://github.com/huggingface/lerobot) on one H100. It trains on `lerobot/svla_so100_pickplace`: 50 teleoperated pick-and-place episodes from an SO-100 robot arm, with top and wrist cameras. The policy learns to predict chunks of arm actions from camera images, the robot state and a language instruction. `smolvla_base` expects three cameras named `camera1` to `camera3`. The recipe maps the dataset's `top` and `wrist` cameras to `camera1` and `camera2`, and marks `camera3` as empty.

The run trains for 2,000 steps at batch size 64 and takes about 17 minutes.

## Before you run

The recipe writes its outputs to `/Volumes/main/default/air_examples/lerobot-smolvla-finetune`. The volume must exist and be writable. To write to another volume, override `parameters.output_root`:

```bash
databricks air run -f workload.yaml --override parameters.output_root=/Volumes/<catalog>/<schema>/<volume>/<dir> --watch
```

## Run

```bash
databricks air run -f workload.yaml --watch
```

## Results

The run prints the training-loss curve. It writes `summary.json` and the fine-tuned policy (`pretrained_model/`) to `<output_root>/<MLflow run ID>/`, and logs the loss summary to MLflow.

In a validation run on an H100, training loss fell from 0.358 at step 50 to 0.082 at step 2,000. The recipe doesn't evaluate the policy on a robot or in simulation.

Notes:

- PyAV's wheel bundles a RHEL build of OpenSSL that aborts with "FATAL FIPS SELFTEST FAILURE" on AI Runtime when `av` is imported. `train.sh` replaces it with the OpenSSL 1.1 libraries from the manylinux2014 OpenCV wheel. Only PyAV's SRT streaming library uses those libraries.
