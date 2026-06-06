"""Shared Modal utilities: the App and the container Image.

This module is intentionally minimal — it only defines infrastructure that is
shared across entrypoints. The actual benchmark function and CLI entrypoint live
in the root-level ``benchmark.py``.
"""

from pathlib import Path

import modal

# GPU type to provision. Swap for "A100", "A10G", "L4", "H200", etc.
GPU = "H200"

app = modal.App("cs336-benchmarking")

REPO_ROOT = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    # NOTE: uv_sync() does not work here — Modal only copies pyproject.toml + uv.lock
    # into the build context, so the editable local path dep `./cs336-basics` cannot
    # be resolved ("Distribution not found at file:///.uv/cs336-basics"). We install
    # the runtime deps explicitly instead.
    .uv_pip_install(
        "torch~=2.11.0",
        "numpy>=2.4",
        "einops>=0.8",
        "einx>=0.4",
        "jaxtyping>=0.3",
    )
    .add_local_dir(
        str(REPO_ROOT / "cs336-basics" / "cs336_basics"),
        remote_path="/root/cs336_basics",
    )
    .add_local_dir(
        str(REPO_ROOT / "cs336_systems"),
        remote_path="/root/cs336_systems",
    )
)
