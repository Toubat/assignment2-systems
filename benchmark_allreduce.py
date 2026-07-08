"""Benchmark single-node all-reduce on Modal L4 GPUs.

Problem (distributed_communication_single_node): time NCCL all-reduce of
float32 tensors (1MB, 10MB, 100MB, 1GB) across 2, 4, and 6 GPUs/processes.

Usage:
    modal run dist_example.py

Writes charts/allreduce_single_node.png and .json locally, and prints a table.
"""

import io
import json
import os

import modal

app = modal.App("dist-allreduce-benchmark")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "numpy", "einops", "einx", "jaxtyping", "matplotlib")
    .add_local_dir("cs336-basics/cs336_basics", remote_path="/root/cs336_basics")
    .add_local_dir("cs336_systems", remote_path="/root/cs336_systems")
)

# These are only importable inside the container image, not locally.
with image.imports():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.distributed as dist
    from torch.multiprocessing.spawn import spawn

    from cs336_systems.benchmarking import BenchOp, benchmark

WORLD_SIZES = [2, 4, 6]
MAX_GPUS = max(WORLD_SIZES)

# float32 elements per tensor size
DATA_SIZE = {
    "1MB": 2**20 // 4,
    "10MB": 10 * 2**20 // 4,
    "100MB": 100 * 2**20 // 4,
    "1GB": 2**30 // 4,
}


def all_reduce_worker(rank: int, world_size: int, result_path: str):
    os.environ["MASTER_ADDR"] = "localhost"
    # Unique port per world size so successive spawns don't collide.
    os.environ["MASTER_PORT"] = str(29500 + world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Defined inside the worker because BenchOp is only importable remotely.
    class AllReduceOp(BenchOp):
        def __init__(self, num_elements: int):
            self.num_elements = num_elements

        def setup(self) -> None:
            self.data = torch.randn(
                self.num_elements, dtype=torch.float32, device="cuda"
            )

        def prepare_run(self) -> None:
            # Align ranks before each timed trial so we measure the
            # collective itself, not stragglers.
            dist.barrier()

        def run(self) -> None:
            dist.all_reduce(self.data, async_op=False)

    results: dict[str, dict[str, float]] = {}
    for label, num_elements in DATA_SIZE.items():
        if rank == 0:
            print(f"--- world_size={world_size}, size={label} ---")
        mean_ms, std_ms = benchmark(
            AllReduceOp(num_elements), num_warmups=5, num_trials=10
        )

        # Average the per-rank timings so the reported number reflects all ranks.
        stats = torch.tensor([mean_ms, std_ms], device="cuda")
        dist.all_reduce(stats)
        stats /= world_size
        if rank == 0:
            results[label] = {"mean_ms": stats[0].item(), "std_ms": stats[1].item()}

    if rank == 0:
        with open(result_path, "w") as f:
            json.dump(results, f)

    dist.destroy_process_group()


def plot_results(all_results: dict[str, dict[str, dict[str, float]]]) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = list(DATA_SIZE)
    for ws, res in all_results.items():
        means = [res[s]["mean_ms"] for s in sizes]
        stds = [res[s]["std_ms"] for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=f"{ws} GPUs")
    ax.set_yscale("log")
    ax.set_xlabel("all-reduce data size (float32)")
    ax.set_ylabel("mean time (ms, log scale)")
    ax.set_title("Single-node NCCL all-reduce on L4 GPUs")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    return buf.getvalue()


@app.function(image=image, gpu=f"L4:{MAX_GPUS}", timeout=1800)
def run_benchmark() -> tuple[dict, bytes]:
    print(f"CUDA devices available: {torch.cuda.device_count()}")

    all_results: dict[str, dict] = {}
    for world_size in WORLD_SIZES:
        result_path = f"/tmp/results_ws{world_size}.json"
        spawn(
            fn=all_reduce_worker,
            args=(world_size, result_path),
            nprocs=world_size,
            join=True,
        )
        with open(result_path) as f:
            all_results[str(world_size)] = json.load(f)

    return all_results, plot_results(all_results)


@app.local_entrypoint()
def main():
    results, png = run_benchmark.remote()

    os.makedirs("charts", exist_ok=True)
    with open("charts/allreduce_single_node.png", "wb") as f:
        f.write(png)
    with open("charts/allreduce_single_node.json", "w") as f:
        json.dump(results, f, indent=2)

    sizes = list(DATA_SIZE)
    world_sizes = list(results)
    header = f"{'size':>8}" + "".join(f"{ws + ' GPUs':>18}" for ws in world_sizes)
    print("\n" + header)
    for size in sizes:
        row = f"{size:>8}"
        for ws in world_sizes:
            r = results[ws][size]
            row += f"{r['mean_ms']:>11.3f} ms ±{r['std_ms']:>4.2f}"
        print(row)
    print("\nSaved charts/allreduce_single_node.png and .json")
