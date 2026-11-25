"""Fine-tune SmolVLA with LeRobot's lerobot-train on the workload parameters."""

import glob
import os
import shutil
import subprocess
import sysconfig

import yaml

with open(os.environ["HYPERPARAMETERS_PATH"]) as f:
    P = yaml.safe_load(f)

RUN_DIR = "/tmp/lerobot/smolvla"
# Log about 40 points per run. summarize.py uses the same interval.
LOG_FREQ = max(P["steps"] // 40, 1)
# smolvla_base expects camera1 to camera3; this dataset has top and wrist cameras.
RENAME_MAP = ('{"observation.images.top": "observation.images.camera1", '
              '"observation.images.wrist": "observation.images.camera2"}')


def swap_openssl():
    # PyAV's wheel bundles a RHEL build of OpenSSL 1.1.1k, used only by its SRT streaming library,
    # that aborts with "FATAL FIPS SELFTEST FAILURE" on AI Runtime as soon as `av` is imported.
    # Swap in the OpenSSL 1.1 libraries from the manylinux2014 OpenCV wheel, which load cleanly.
    site = sysconfig.get_paths()["purelib"]
    for lib in ["ssl", "crypto"]:
        src = glob.glob(f"{site}/opencv_python_headless.libs/lib{lib}-*.so.1.1")[0]
        shutil.copy(src, glob.glob(f"{site}/av.libs/lib{lib}-*.so.1.1.1k")[0])
        shutil.copy(src, f"{site}/av.libs/")  # the replacement libssl looks up libcrypto by its own name


def main():
    swap_openssl()
    import av  # Imported after the swap, because importing it first aborts on AI Runtime.
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
