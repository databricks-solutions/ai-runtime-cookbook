#!/usr/bin/env python3
"""Install NVIDIA Transformer Engine (TE), which provides the FP8 and NVFP4 kernels bench.py uses.

The databricks_ai_v6 environment provides PyTorch (CUDA 13) but not TE. TE's CUDA core library
installs from a prebuilt wheel; only its PyTorch binding, transformer_engine_torch, compiles from
source (about 2 minutes on an 8xB300 node).

Optional: set TE_WHEEL_CACHE to a Unity Catalog volume directory. The first run saves the compiled
binding there and later runs install it in seconds. Wheels are kept in a subdirectory keyed on the
Python, PyTorch, and CUDA versions and NVTE_CUDA_ARCHS, so a changed environment builds a fresh wheel.
"""

import glob
import importlib
import importlib.metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nvidia
import torch

# 2.17.1 matches the cuBLAS in databricks_ai_v6; newer TE releases need a newer cuBLASLt.
TE_VERSION = os.environ.get("TE_VERSION", "2.17.1")


def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", *args], check=True)


def gpu0_compute_capability():
    # For example 103 on B300 or 90 on H100.
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout
    return out.splitlines()[0].strip().replace(".", "")


def is_installed():
    try:
        import transformer_engine.pytorch  # noqa: F401
    except Exception:
        return False
    return True


def report_ready():
    importlib.invalidate_caches()
    import transformer_engine.pytorch  # noqa: F401
    print("Transformer Engine ready:", importlib.metadata.version("transformer_engine"), flush=True)


def save_binding(cache):
    # pip keeps the wheel it just built in its cache; copy it to the volume for reuse. Best-effort:
    # TE is already installed, so a failed copy doesn't fail the run.
    try:
        cache.mkdir(parents=True, exist_ok=True)
        pip_cache = subprocess.run(
            [sys.executable, "-m", "pip", "cache", "dir"], check=True, capture_output=True, text=True,
        ).stdout.strip()
        for wheel in Path(pip_cache, "wheels").rglob(f"transformer_engine_torch-{TE_VERSION}-*.whl"):
            shutil.copy(wheel, cache)
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"Could not copy the compiled binding to {cache}: {err}", flush=True)
    if glob.glob(f"{cache}/transformer_engine_torch-{TE_VERSION}-*.whl"):
        print(f"Cached the compiled binding in {cache}.", flush=True)
    else:
        print("Could not find the compiled wheel in the pip cache; later runs will rebuild.", flush=True)


def main():
    archs = os.environ.get("NVTE_CUDA_ARCHS") or gpu0_compute_capability()
    os.environ["NVTE_CUDA_ARCHS"] = archs

    cache = None
    cache_root = os.environ.get("TE_WHEEL_CACHE", "").rstrip("/")
    if cache_root:
        # For example .../py3.12-torch2.11.0+cu130-sm90 (the torch version includes CUDA).
        py = f"py{sys.version_info.major}.{sys.version_info.minor}"
        cache = Path(f"{cache_root}/{py}-torch{torch.__version__}-sm{archs.replace(';', '_')}")

    if is_installed():
        print("Transformer Engine is already installed.", flush=True)
        return

    if cache and glob.glob(f"{cache}/transformer_engine_torch-{TE_VERSION}-*.whl"):
        print(f"Installing Transformer Engine {TE_VERSION} with the cached binding from {cache}...", flush=True)
        pip("install", "--find-links", str(cache), f"transformer_engine[pytorch]=={TE_VERSION}")
        report_ready()
        return

    print(f"Building Transformer Engine {TE_VERSION} for CUDA archs {archs}...", flush=True)
    # The CUDA math-library headers (cusparse.h, nvtx3, ...) are not under /usr/local/cuda/include in
    # this environment; they ship inside the nvidia-* pip wheels that PyTorch depends on.
    nv_includes = sorted({d for p in nvidia.__path__ for d in glob.glob(f"{p}/*/include")})
    os.environ["CPATH"] = ":".join([*nv_includes, "/usr/local/cuda/include", os.environ.get("CPATH", "")])
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("MAX_JOBS", str(len(os.sched_getaffinity(0))))
    pip("install", "--no-build-isolation", f"transformer_engine[pytorch]=={TE_VERSION}")

    # Only node 0 writes the cache, so the nodes of a multi-node run don't write the same file at once.
    if cache and os.environ.get("NODE_RANK", "0") == "0":
        save_binding(cache)
    report_ready()


if __name__ == "__main__":
    main()
