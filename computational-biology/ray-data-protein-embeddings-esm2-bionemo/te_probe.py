"""Probes used by install_transformer_engine.sh.

Each command prints or exits with the value the shell script needs.
"""

import glob
import importlib.metadata
import sys


def cuda_include_dirs():
    """CUDA headers ship inside the nvidia-* pip wheels that PyTorch depends on."""
    import nvidia

    return ":".join(sorted({d for p in nvidia.__path__ for d in glob.glob(p + "/*/include")}))


def main():
    command = sys.argv[1]
    if command == "import":
        # Exits non-zero when Transformer Engine is not installed.
        import transformer_engine.pytorch  # noqa: F401
    elif command == "include-dirs":
        print(cuda_include_dirs())
    elif command == "version":
        import transformer_engine.pytorch  # noqa: F401

        print("Transformer Engine ready:", importlib.metadata.version("transformer_engine"))
    else:
        raise SystemExit(f"Unknown probe: {command}")


if __name__ == "__main__":
    main()
