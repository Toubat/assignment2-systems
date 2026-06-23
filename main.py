"""Generic Modal runner: ship the codebase to a GPU container and run a command.

Useful for anything that can't run locally (e.g. Triton kernels / tests on
macOS, which has no Triton wheels). The command runs in ``/root`` with the repo
on ``PYTHONPATH``, so imports like ``cs336_systems`` / ``cs336_basics`` work.

Examples:

    modal run main.py --command "pytest tests/test_attention.py -k pytorch"
    modal run main.py --command "pytest tests/test_weighted_sum.py -v"
    modal run main.py --command "python -c 'import torch; print(torch.cuda.get_device_name(0))'"
"""

from cs336_systems.modal import GPU, REPO_ROOT, app, image

# Reuse the shared image (deps incl. torch+Triton, and the cs336_systems /
# cs336_basics packages already mounted); add the test suite and pytest config
# so the cloud run behaves like a local one. Stacking add_local_* is allowed.
runner_image = image.add_local_dir(
    str(REPO_ROOT / "tests"), remote_path="/root/tests"
).add_local_file(str(REPO_ROOT / "pyproject.toml"), remote_path="/root/pyproject.toml")


@app.function(image=runner_image, gpu=GPU, timeout=60 * 60)
def run_command(command: str) -> int:
    import os
    import subprocess

    import torch

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    print(f"[main] running on {device}")
    print(f"[main] $ {command}")

    # Put the repo root on PYTHONPATH so `import cs336_systems` works regardless
    # of how the command launches Python (bare `pytest`, `python script.py`, ...).
    env = {**os.environ, "PYTHONPATH": "/root"}
    return subprocess.run(command, shell=True, cwd="/root", env=env).returncode


@app.local_entrypoint()
def main(command: str):
    code = run_command.remote(command)
    if code != 0:
        raise SystemExit(f"command exited with code {code}")
    print("[main] command succeeded")
