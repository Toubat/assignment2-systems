"""Run the Triton kernel tests on a Modal GPU.

Triton kernels can't run on macOS, so this ships the test suite to a Modal GPU
container and runs pytest there.

    modal run run_kernel_tests.py
"""

from cs336_systems.modal import GPU, REPO_ROOT, app, image

# Reuse the shared image (now includes pytest); just mount the test suite on top.
# Stacking another add_local_* onto the base image is allowed (the constraint is
# only that no build step may follow an add_local_* step).
test_image = image.add_local_dir(str(REPO_ROOT / "tests"), remote_path="/root/tests")


@app.function(image=test_image, gpu=GPU, timeout=60 * 20)
def run_tests() -> int:
    import subprocess
    import sys

    import torch

    assert (
        torch.cuda.is_available()
    ), "expected a CUDA device inside the Modal GPU container"
    print(f"running kernel tests on {torch.cuda.get_device_name(0)}")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "/root/tests/test_weighted_sum.py",
            "-v",
            "-s",
        ],
        cwd="/root",
    )
    return proc.returncode


@app.local_entrypoint()
def main():
    code = run_tests.remote()
    if code != 0:
        raise SystemExit(f"pytest failed with exit code {code}")
    print("All kernel tests passed on GPU.")
