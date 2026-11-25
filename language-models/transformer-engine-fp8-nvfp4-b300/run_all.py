#!/usr/bin/env python3
"""Train one precision per GPU in parallel, then print the bf16-relative comparison."""

import os
import subprocess
import sys
from pathlib import Path

RECIPE_DIR = Path(__file__).resolve().parent
LOG_DIR = Path("/tmp/te-bench")


def main():
    precisions = os.environ.get("PRECISIONS", "bf16 fp8_current mxfp8 nvfp4").split()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for gpu, precision in enumerate(precisions):
        log_path = LOG_DIR / f"log_{precision}.txt"
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                [sys.executable, str(RECIPE_DIR / "bench.py"), "train", precision],
                cwd=RECIPE_DIR,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        jobs.append((precision, proc, log_path))

    status = 0
    for precision, proc, log_path in jobs:
        if proc.wait() != 0:
            print(f"=== {precision} FAILED; last log lines:")
            print("".join(log_path.read_text().splitlines(keepends=True)[-40:]))
            status = 1
    for precision in precisions:
        lines = [line for line in (LOG_DIR / f"log_{precision}.txt").read_text().splitlines() if line.startswith("[")]
        print("\n".join(lines[-3:]))

    subprocess.run([sys.executable, str(RECIPE_DIR / "summarize.py")], check=True)
    sys.exit(status)


if __name__ == "__main__":
    main()
