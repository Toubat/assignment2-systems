"""Modal benchmark for optimizer-state-sharding accounting (parts a and b).

Compares ordinary AdamW with ``ShardedOptimizer`` using the same DDP model on
one node with two GPUs. The assignment configuration is the default:

    modal run benchmark_optimizer_state.py

For a cheaper smoke test:

    GPU_TYPE=L4 modal run benchmark_optimizer_state.py --size small \
        --num-warmups 1 --num-trials 2

Writes a JSON report and a training-speed chart under ``charts/``.
"""

import json
import os

import modal

app = modal.App("optimizer-state-sharding-accounting")

GPU_TYPE = os.environ.get("GPU_TYPE", "H200")
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch~=2.11.0",
        "numpy",
        "einops",
        "einx",
        "jaxtyping",
        "matplotlib",
    )
    .add_local_dir("cs336-basics/cs336_basics", remote_path="/root/cs336_basics")
    .add_local_dir("cs336_systems", remote_path="/root/cs336_systems")
)

with image.imports():
    import torch
    import torch.distributed as dist
    from torch.multiprocessing.spawn import spawn

    from cs336_basics.optimizer import AdamW
    from cs336_systems.benchmarking import LMConfig, benchmark_parallel_training
    from cs336_systems.dist.ddp import DDP
    from cs336_systems.dist.optim import ShardedOptimizer
    from cs336_systems.plots import grouped_bar_chart

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
DEFAULT_GLOBAL_BATCH_SIZE = 4

# Section 2.1.2: d_model, d_ff, num_layers, num_heads.
MODEL_SIZES = {
    "small": (768, 3072, 12, 12),
    "xl": (2560, 10240, 32, 32),
}

IMPLEMENTATIONS = ("unsharded", "sharded")


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


def benchmark_worker(
    rank: int,
    world_size: int,
    size: str,
    implementation: str,
    global_batch_size: int,
    num_warmups: int,
    num_trials: int,
    port: int,
    result_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if implementation == "unsharded":
        make_optimizer = lambda params: AdamW(params, lr=1e-3)
    elif implementation == "sharded":
        make_optimizer = lambda params: ShardedOptimizer(params, AdamW, lr=1e-3)
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    results = benchmark_parallel_training(
        config=get_config(size),
        wrap_model=DDP,
        make_optimizer=make_optimizer,
        global_batch_size=global_batch_size,
        num_warmups=num_warmups,
        num_trials=num_trials,
    )

    if rank == 0:
        with open(result_path, "w") as file:
            json.dump(results, file)

    dist.destroy_process_group()


def plot_speed_results(
    results: dict[str, dict[str, float]],
    size: str,
    world_size: int,
) -> bytes:
    implementations = list(IMPLEMENTATIONS)
    gpu_name = torch.cuda.get_device_name(0)
    path = grouped_bar_chart(
        f"optimizer_state_sharding_speed_{size}_ws{world_size}.png",
        f"Optimizer state sharding training speed ({size}, {world_size}x {gpu_name})",
        "optimizer",
        "mean time per iteration (ms)",
        categories=implementations,
        groups={
            "training step": [
                results[implementation]["step_ms_mean"]
                for implementation in implementations
            ]
        },
    )
    return path.read_bytes()


def plot_memory_results(
    results: dict[str, dict[str, float]],
    size: str,
    world_size: int,
) -> bytes:
    gpu_name = torch.cuda.get_device_name(0)
    metrics = (
        ("after model init", "memory_after_model_bytes_max"),
        ("before opt step", "memory_before_optimizer_step_bytes_max"),
        ("after opt step", "memory_after_optimizer_step_bytes_max"),
        ("peak fwd/bwd", "peak_forward_backward_bytes_max"),
        ("peak opt step", "peak_optimizer_step_bytes_max"),
        ("parameters", "parameter_bytes_max"),
        ("gradients", "gradient_bytes_max"),
        ("optimizer state", "optimizer_state_bytes_max"),
    )
    path = grouped_bar_chart(
        f"optimizer_state_sharding_memory_{size}_ws{world_size}.png",
        f"Optimizer state sharding memory ({size}, {world_size}x {gpu_name})",
        "checkpoint / component",
        "maximum per-rank allocated memory (GiB)",
        categories=[label for label, _ in metrics],
        groups={
            implementation: [
                gibibytes(results[implementation][key]) for _, key in metrics
            ]
            for implementation in IMPLEMENTATIONS
        },
    )
    return path.read_bytes()


@app.function(image=image, gpu=f"{GPU_TYPE}:{GPU_COUNT}", timeout=60 * 60)
def run_benchmark(
    size: str,
    world_size: int,
    global_batch_size: int,
    num_warmups: int,
    num_trials: int,
) -> tuple[dict[str, dict[str, float]], bytes, bytes]:
    assert world_size <= torch.cuda.device_count()

    results: dict[str, dict[str, float]] = {}
    for index, implementation in enumerate(IMPLEMENTATIONS):
        print(f"--- {implementation}: {size}, world_size={world_size} ---")
        result_path = f"/tmp/optimizer_state_{implementation}.json"
        spawn(
            fn=benchmark_worker,
            args=(
                world_size,
                size,
                implementation,
                global_batch_size,
                num_warmups,
                num_trials,
                29700 + index,
                result_path,
            ),
            nprocs=world_size,
            join=True,
        )
        with open(result_path) as file:
            results[implementation] = json.load(file)

    return (
        results,
        plot_memory_results(results, size, world_size),
        plot_speed_results(results, size, world_size),
    )


def gibibytes(byte_count: float) -> float:
    return byte_count / 2**30


@app.local_entrypoint()
def main(
    size: str = "xl",
    world_size: int = 2,
    global_batch_size: int = DEFAULT_GLOBAL_BATCH_SIZE,
    num_warmups: int = 5,
    num_trials: int = 10,
) -> None:
    assert size in MODEL_SIZES, f"choose from {list(MODEL_SIZES)}"
    assert world_size == 2, "the assignment setting requires two GPUs"
    assert GPU_COUNT >= world_size, f"set GPU_COUNT>={world_size}"

    results, memory_chart, speed_chart = run_benchmark.remote(
        size, world_size, global_batch_size, num_warmups, num_trials
    )

    os.makedirs("charts", exist_ok=True)
    output_stem = f"charts/optimizer_state_sharding_{size}_ws{world_size}"
    with open(f"{output_stem}.json", "w") as file:
        json.dump(results, file, indent=2)
    with open(f"{output_stem}_memory.png", "wb") as file:
        file.write(memory_chart)
    with open(f"{output_stem}_speed.png", "wb") as file:
        file.write(speed_chart)

    metrics = (
        ("after model init", "memory_after_model_bytes_max"),
        ("before optimizer step", "memory_before_optimizer_step_bytes_max"),
        ("after optimizer step", "memory_after_optimizer_step_bytes_max"),
        ("peak forward/backward", "peak_forward_backward_bytes_max"),
        ("peak optimizer step", "peak_optimizer_step_bytes_max"),
        ("parameters", "parameter_bytes_max"),
        ("gradients", "gradient_bytes_max"),
        ("optimizer states", "optimizer_state_bytes_max"),
    )

    print(
        f"\n{'metric':>24} {'unsharded (GiB)':>18} "
        f"{'sharded (GiB)':>16} {'reduction':>12}"
    )
    for label, key in metrics:
        unsharded = gibibytes(results["unsharded"][key])
        sharded = gibibytes(results["sharded"][key])
        reduction = 1 - sharded / unsharded if unsharded else 0.0
        print(f"{label:>24} {unsharded:>18.2f} " f"{sharded:>16.2f} {reduction:>11.1%}")

    print(f"\n{'optimizer':>12} {'step time (ms)':>20}")
    for implementation in IMPLEMENTATIONS:
        result = results[implementation]
        print(
            f"{implementation:>12} "
            f"{result['step_ms_mean']:>12.2f} ±{result['step_ms_std']:>5.2f}"
        )
    speed_overhead = (
        results["sharded"]["step_ms_mean"] / results["unsharded"]["step_ms_mean"] - 1
    )
    print(f"Sharding time overhead: {speed_overhead:.1%}")
    print(
        f"\nSaved {output_stem}.json, "
        f"{output_stem}_memory.png, and {output_stem}_speed.png"
    )
