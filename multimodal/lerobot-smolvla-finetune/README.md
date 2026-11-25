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

The run prints the training-loss curve. It writes `summary.json` and the fine-tuned policy (`pretrained_model/`, loadable with `SmolVLAPolicy.from_pretrained`) to `<output_root>/<MLflow run ID>/`, and logs the loss summary to MLflow.

In a validation run on an H100, training loss fell from 0.358 at step 50 to 0.082 at step 2,000. The recipe doesn't evaluate the policy on a robot or in simulation.

After training, `load_policy.py` loads the saved policy in a new process and predicts one action for the first training frame. To load the policy yourself, use the same code:

```python
"""Load the saved SmolVLA policy in a fresh process and predict one action."""

import os

import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

policy_dir = f"{P['output_root']}/{os.environ['MLFLOW_RUN_ID']}/pretrained_model"
policy = SmolVLAPolicy.from_pretrained(policy_dir).to("cuda").eval()
preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=policy_dir)

# The first training frame, with the cameras renamed to the names the policy expects.
frame = LeRobotDataset(P["dataset_repo_id"], video_backend="pyav")[0]
observation = {
    "observation.images.camera1": frame["observation.images.top"],
    "observation.images.camera2": frame["observation.images.wrist"],
    "observation.state": frame["observation.state"],
    "task": frame["task"],
}
policy.reset()
action = postprocessor(policy.select_action(preprocessor(observation)))
assert torch.isfinite(action).all(), "The loaded policy predicted a non-finite action"
print("Loaded policy predicted an action with shape", tuple(action.shape))
```
