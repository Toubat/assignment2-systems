"""Benchmark parallel training strategies on Modal GPUs (single node).

Problems (cs336_assignment2_systems.pdf, pp. 33-36):
- naive_ddp_benchmarking: per-parameter all-reduce after backward
- minimal_ddp_flat_benchmarking: single all-reduce on flattened gradients
- ddp_overlap_individual_parameters_benchmarking(a): async all-reduce per
  parameter, overlapped with the backward pass

Named generically: the same harness will later benchmark FSDP / sharded
optimizers / 2D parallelism -- anything that plugs in via `wrap_model` /
`make_optimizer` in cs336_systems.benchmarking.benchmark_parallel_training.

Usage:
    # smoke test (small model, cheap GPUs)
    GPU_TYPE=L4 modal run benchmark_parallelism.py --size small

    # the assignment setting: xl model, 1 node x 2 GPUs
    modal run benchmark_parallelism.py --size xl

    # options
    modal run benchmark_parallelism.py --size xl --impls naive,flat,overlap \
        --world-size 2 --num-warmups 5 --num-trials 20

GPU type/count are baked into the Modal function at import time, so they are
set via env vars: GPU_TYPE (default H200; xl + AdamW needs ~55GB/rank, too big
for L4) and GPU_COUNT (default = 2).

Writes charts/parallelism_<size>_ws<n>.json and .png locally, and prints a table.
"""

import json
import os

import modal

app = modal.App("parallelism-benchmark")

GPU_TYPE = os.environ.get("GPU_TYPE", "H200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "numpy", "einops", "einx", "jaxtyping", "matplotlib")
    .add_local_dir("cs336-basics/cs336_basics", remote_path="/root/cs336_basics")
    .add_local_dir("cs336_systems", remote_path="/root/cs336_systems")
)

# These are only importable inside the container image, not locally.
with image.imports():
    import torch
    import torch.distributed as dist
    from torch.multiprocessing.spawn import spawn

    from cs336_systems.benchmarking import LMConfig, benchmark_parallel_training
    from cs336_systems.dist.ddp import DDP, NaiveDDP

# Section 2.1.2: vocab 10,000, batch size 4, context length 512.
VOCAB_SIZE = 10_000
DEFAULT_GLOBAL_BATCH_SIZE = 4
CONTEXT_LENGTH = 512

# size -> (d_model, d_ff, num_layers, num_heads)
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
}

IMPLS = ["naive", "flat", "overlap"]


def get_config(size: str) -> "LMConfig":
    d_model, d_ff, num_layers, num_heads = MODEL_SIZES[size]
    return LMConfig(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def train_worker(
    rank: int,
    world_size: int,
    size: str,
    impl: str,
    global_batch_size: int,
    num_warmups: int,
    num_trials: int,
    port: int,
    result_path: str,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    wrappers = {
        "naive": lambda m: NaiveDDP(m, batch_all_reduce=False),
        "flat": lambda m: NaiveDDP(m, batch_all_reduce=True),
        "overlap": DDP,
    }

    results = benchmark_parallel_training(
        config=get_config(size),
        wrap_model=wrappers[impl],
        global_batch_size=global_batch_size,
        num_warmups=num_warmups,
        num_trials=num_trials,
    )

    if rank == 0:
        with open(result_path, "w") as f:
            json.dump(results, f)

    dist.destroy_process_group()


def plot_results(
    results: dict[str, dict[str, float]],
    size: str,
    world_size: int,
    global_batch_size: int,
) -> bytes:
    """Grouped bar chart of step vs comm time per impl. Rendered remotely
    (matplotlib lives in the image, not necessarily locally)."""
    from cs336_systems.plots import grouped_bar_chart

    impls = list(results)
    path = grouped_bar_chart(
        f"parallelism_{size}_ws{world_size}_gb{global_batch_size}.png",
        f"DDP training step breakdown ({size}, {world_size}x {GPU_TYPE}, global batch {global_batch_size})",
        "implementation",
        "time per iteration (ms)",
        categories=impls,
        groups={
            "total step": [results[i]["step_ms_mean"] for i in impls],
            "grad comm (non-overlapped)": [results[i]["comm_ms_mean"] for i in impls],
        },
    )
    return path.read_bytes()


@app.function(image=image, gpu=f"{GPU_TYPE}:{GPU_COUNT}", timeout=3600)
def run_benchmark(
    size: str,
    impls: list[str],
    world_size: int,
    global_batch_size: int,
    num_warmups: int,
    num_trials: int,
) -> tuple[dict, bytes]:
    print(f"CUDA devices available: {torch.cuda.device_count()}")
    assert world_size <= torch.cuda.device_count(), f"world_size={world_size} > available GPUs; set GPU_COUNT>={world_size}"

    results: dict[str, dict[str, float]] = {}
    for i, impl in enumerate(impls):
        print(f"--- impl={impl}, size={size}, world_size={world_size}, global_batch_size={global_batch_size} ---")
        result_path = f"/tmp/results_{impl}.json"
        spawn(
            fn=train_worker,
            # Unique port per run so successive process groups don't collide.
            args=(
                world_size,
                size,
                impl,
                global_batch_size,
                num_warmups,
                num_trials,
                29600 + i,
                result_path,
            ),
            nprocs=world_size,
            join=True,
        )
        with open(result_path) as f:
            results[impl] = json.load(f)
        print(results[impl])

    return results, plot_results(results, size, world_size, global_batch_size)


@app.local_entrypoint()
def main(
    size: str = "xl",
    impls: str = "naive,flat,overlap",
    world_size: int = 2,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    num_warmups: int = 5,
    num_trials: int = 20,
):
    impl_list = [i.strip() for i in impls.split(",")]
    unknown = set(impl_list) - set(IMPLS)
    assert not unknown, f"unknown impls {unknown}; choose from {IMPLS}"
    assert size in MODEL_SIZES, f"unknown size {size!r}; choose from {list(MODEL_SIZES)}"

    results, png = run_benchmark.remote(size, impl_list, world_size, global_batch_size, num_warmups, num_trials)

    os.makedirs("charts", exist_ok=True)
    stem = f"charts/parallelism_{size}_ws{world_size}_gb{global_batch_size}"
    with open(f"{stem}.png", "wb") as f:
        f.write(png)
    with open(f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'impl':>8} {'step (ms)':>18} {'comm (ms)':>18} {'comm %':>8}")
    for impl, r in results.items():
        print(f"{impl:>8} {r['step_ms_mean']:>11.2f} ±{r['step_ms_std']:>5.2f} {r['comm_ms_mean']:>11.2f} ±{r['comm_ms_std']:>5.2f} {r['comm_fraction'] * 100:>7.1f}%")
    print(f"\nSaved {stem}.png and .json")
