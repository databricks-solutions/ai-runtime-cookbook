"""Fine-tune SmolVLA with LeRobot's lerobot-train on the workload parameters."""

import os
import subprocess

import av
import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_DIR = "/tmp/lerobot/smolvla"
# Log about 40 points per run. summarize.py uses the same interval.
LOG_FREQ = max(P["steps"] // 40, 1)
# smolvla_base expects camera1 to camera3; this dataset has top and wrist cameras.
RENAME_MAP = ('{"observation.images.top": "observation.images.camera1", '
              '"observation.images.wrist": "observation.images.camera2"}')


def main():
    print("PyAV", av.__version__)

    subprocess.run([
        "lerobot-train",
        f"--policy.path={P['policy_path']}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--policy.empty_cameras=1",
        f"--dataset.repo_id={P['dataset_repo_id']}",
        "--dataset.video_backend=pyav",
        f"--rename_map={RENAME_MAP}",
        f"--batch_size={P['batch_size']}",
        f"--steps={P['steps']}",
        f"--save_freq={P['steps']}",
        f"--log_freq={LOG_FREQ}",
        "--num_workers=8",
        f"--output_dir={RUN_DIR}",
        "--job_name=smolvla_so100",
    ], check=True)


if __name__ == "__main__":
    main()
