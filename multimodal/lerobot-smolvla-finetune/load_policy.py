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
