"""Modal BF16 benchmark for optimizer/FSDP sharding accounting.

Compares DDP with ordinary AdamW, DDP with ``ShardedOptimizer``, and FSDP with
ordinary AdamW on one node with two GPUs. The assignment configuration is the
default:

    modal run benchmark_optimizer_state.py

For a cheaper smoke test:

    GPU_TYPE=L4 modal run benchmark_optimizer_state.py --size small \
        --num-warmups 1 --num-trials 2

Writes a JSON report and a training-speed chart under ``charts/``.
"""

import json
import os

import modal

app = modal.App("fsdp-accounting")

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
    from cs336_systems.dist.ddp import DDP, FSDP
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

IMPLEMENTATIONS = ("unsharded", "sharded", "fsdp")


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
    autocast_bfloat16: bool,
    port: int,
    result_path: str,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if implementation == "unsharded":
        wrap_model = DDP
        make_optimizer = lambda params: AdamW(params, lr=1e-3)
    elif implementation == "sharded":
        wrap_model = DDP
        make_optimizer = lambda params: ShardedOptimizer(params, AdamW, lr=1e-3)
    elif implementation == "fsdp":
        wrap_model = FSDP
        make_optimizer = lambda params: AdamW(params, lr=1e-3)
    else:
        raise ValueError(f"unknown implementation: {implementation}")

    results = benchmark_parallel_training(
        config=get_config(size),
        wrap_model=wrap_model,
        make_optimizer=make_optimizer,
        global_batch_size=global_batch_size,
        num_warmups=num_warmups,
        num_trials=num_trials,
        autocast_bfloat16=autocast_bfloat16,
    )

    if rank == 0:
        with open(result_path, "w") as file:
            json.dump(results, file)

    dist.destroy_process_group()


def plot_speed_results(
    results: dict[str, dict[str, float]],
    size: str,
    world_size: int,
    autocast_bfloat16: bool,
) -> bytes:
    implementations = list(IMPLEMENTATIONS)
    gpu_name = torch.cuda.get_device_name(0)
    precision = "BF16 autocast" if autocast_bfloat16 else "FP32"
    path = grouped_bar_chart(
        f"fsdp_accounting_speed_{size}_ws{world_size}.png",
        f"FSDP comparison speed ({size}, {precision}, {world_size}x {gpu_name})",
        "implementation",
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
    autocast_bfloat16: bool,
) -> bytes:
    gpu_name = torch.cuda.get_device_name(0)
    precision = "BF16 autocast" if autocast_bfloat16 else "FP32"
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
        f"fsdp_accounting_memory_{size}_ws{world_size}.png",
        f"FSDP comparison memory ({size}, {precision}, {world_size}x {gpu_name})",
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
    autocast_bfloat16: bool,
) -> tuple[dict[str, dict[str, float]], bytes, bytes]:
    assert world_size <= torch.cuda.device_count()

    results: dict[str, dict[str, float]] = {}
    for index, implementation in enumerate(IMPLEMENTATIONS):
        print(f"--- {implementation}: {size}, world_size={world_size} ---")
        result_path = f"/tmp/fsdp_accounting_{implementation}.json"
        spawn(
            fn=benchmark_worker,
            args=(
                world_size,
                size,
                implementation,
                global_batch_size,
                num_warmups,
                num_trials,
                autocast_bfloat16,
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
        plot_memory_results(results, size, world_size, autocast_bfloat16),
        plot_speed_results(results, size, world_size, autocast_bfloat16),
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
    autocast_bfloat16: bool = True,
) -> None:
    assert size in MODEL_SIZES, f"choose from {list(MODEL_SIZES)}"
    assert world_size >= 1, "world_size must be positive"
    assert (
        global_batch_size % world_size == 0
    ), "global_batch_size must be divisible by world_size"
    assert GPU_COUNT >= world_size, f"set GPU_COUNT>={world_size}"

    results, memory_chart, speed_chart = run_benchmark.remote(
        size,
        world_size,
        global_batch_size,
        num_warmups,
        num_trials,
        autocast_bfloat16,
    )

    os.makedirs("charts", exist_ok=True)
    output_stem = f"charts/fsdp_accounting_{size}_ws{world_size}"
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
        f"{'sharded (GiB)':>16} {'fsdp (GiB)':>14}"
    )
    for label, key in metrics:
        unsharded = gibibytes(results["unsharded"][key])
        sharded = gibibytes(results["sharded"][key])
        fsdp = gibibytes(results["fsdp"][key])
        print(f"{label:>24} {unsharded:>18.2f} " f"{sharded:>16.2f} {fsdp:>14.2f}")

    print(f"\n{'implementation':>14} {'step time (ms)':>20}")
    for implementation in IMPLEMENTATIONS:
        result = results[implementation]
        print(
            f"{implementation:>12} "
            f"{result['step_ms_mean']:>12.2f} ±{result['step_ms_std']:>5.2f}"
        )
    for implementation in ("sharded", "fsdp"):
        speed_overhead = (
            results[implementation]["step_ms_mean"]
            / results["unsharded"]["step_ms_mean"]
            - 1
        )
        print(f"{implementation} time overhead: {speed_overhead:.1%}")
    print(
        f"\nSaved {output_stem}.json, "
        f"{output_stem}_memory.png, and {output_stem}_speed.png"
    )
