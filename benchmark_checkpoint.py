"""Modal entrypoint for benchmarking gradient-checkpointing strategies.

Compares two activation-checkpointing schemes on a stack of ``num_layers``
identical Transformer blocks:

  * recursive (binary nested) checkpointing  -> O(log N) peak, O(N log N) compute
  * linear (flat) checkpointing, swept over a range of group sizes
        -> peak ~ O(N/group + group), minimized near group ~ sqrt(N)

Each strategy/group-size runs in its own Modal container so the whole sweep
runs concurrently. For every run we report mean step time (via ``benchmark``)
and peak CUDA memory.

Run with:

    modal run benchmark_checkpoint.py
    modal run benchmark_checkpoint.py --num-trials 20 --num-warmups 5
"""

from cs336_systems.modal import GPU, app, image

# Config chosen to push one block's residual `b` as close to one boundary `a` as
# possible, so the optimum group size moves toward sqrt(N). Levers (b/a is
# batch-invariant, so batch size does NOT matter here):
#   - small context_length  -> kills the O(S^2) attention term
#   - d_ff == d_model        -> minimizes the FFN residual term
#   - large d_model          -> makes the boundary a big fraction of the block
# A hard floor of ~4-5 remains (Q/K/V/attn-out are each one boundary in size),
# so we also raise num_layers to keep g* = sqrt(N*a/b) sizable.
MODEL = {
    "vocab_size": 10000,  # unused by the checkpoint ops, required by LMConfig
    "context_length": 128,
    "d_model": 2048,
    "d_ff": 2048,
    "num_layers": 300,
    "num_heads": 16,
}

# Flat-checkpoint group sizes to sweep over num_layers=300. Dense near the
# expected interior optimum (g* ~ 6) with coarse coverage out to g=N.
GROUP_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 100, 300]


@app.function(image=image, gpu=GPU, timeout=60 * 30)
def run_checkpoint_benchmark(
    strategy: str,
    group_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    num_warmups: int,
    num_trials: int,
    autocast_bfloat16: bool = False,
) -> dict[str, object]:
    import torch

    from cs336_systems.benchmarking import (
        LMConfig,
        LinearCheckpointOp,
        RecursiveCheckpointOp,
        benchmark,
    )

    assert (
        torch.cuda.is_available()
    ), "expected a CUDA device inside the Modal GPU container"

    config = LMConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    if strategy == "recursive":
        op = RecursiveCheckpointOp(config)
        label = "recursive"
    elif strategy == "linear":
        op = LinearCheckpointOp(config, group_size=group_size)
        label = f"linear(g={group_size})"
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    print(f"[{label}] running on {torch.cuda.get_device_name(0)}")

    try:
        mean, std = benchmark(
            op,
            num_warmups=num_warmups,
            num_trials=num_trials,
            autocast_bfloat16=autocast_bfloat16,
        )
        peak_bytes = torch.cuda.max_memory_allocated()
        return {
            "strategy": strategy,
            "group_size": group_size,
            "label": label,
            "mean": mean,
            "std": std,
            "peak_bytes": peak_bytes,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"[{label}] OOM")
        torch.cuda.empty_cache()
        return {
            "strategy": strategy,
            "group_size": group_size,
            "label": label,
            "mean": None,
            "std": None,
            "peak_bytes": None,
        }


def _format_table(
    results: list[dict[str, object]], num_trials: int, autocast_bfloat16: bool = False
) -> str:
    headers = ["strategy", "time (ms)", "peak (MiB)"]
    rows = [headers]

    def peak_mib(r: dict[str, object]) -> float | None:
        return None if r["peak_bytes"] is None else r["peak_bytes"] / (1024**2)

    # Best (lowest) peak among non-OOM linear runs, to flag the optimum.
    linear_peaks = [
        peak_mib(r)
        for r in results
        if r["strategy"] == "linear" and r["peak_bytes"] is not None
    ]
    best_peak = min(linear_peaks) if linear_peaks else None

    for r in results:
        if r["mean"] is None:
            time_cell = "OOM"
        else:
            time_cell = f"{r['mean']:.2f} ± {r['std']:.2f}"
        p = peak_mib(r)
        if p is None:
            peak_cell = "—"
        else:
            marker = (
                "  <-- min"
                if (best_peak is not None and abs(p - best_peak) < 1e-6)
                else ""
            )
            peak_cell = f"{p:.1f}{marker}"
        rows.append([str(r["label"]), time_cell, peak_cell])

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = "  "

    def fmt_row(r: list[str]) -> str:
        return sep.join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    line = "-" * (sum(widths) + len(sep) * (len(headers) - 1))
    precision = "bf16 autocast" if autocast_bfloat16 else "fp32"
    out = [
        f"=== Gradient Checkpointing [{precision}] (ms/step over {num_trials} trials; "
        f"{MODEL['num_layers']} layers, d_model={MODEL['d_model']}, "
        f"ctx={MODEL['context_length']}) ===",
        fmt_row(rows[0]),
        line,
        *[fmt_row(r) for r in rows[1:]],
    ]
    return "\n".join(out)


@app.local_entrypoint()
def main(num_warmups: int = 3, num_trials: int = 10, autocast_bfloat16: bool = True):
    jobs: list[tuple[str, int]] = [("recursive", 0)]
    jobs += [("linear", g) for g in GROUP_SIZES]

    # Spawn every strategy/group-size at once so the whole sweep runs concurrently.
    handles = []
    for strategy, group_size in jobs:
        handle = run_checkpoint_benchmark.spawn(
            strategy=strategy,
            group_size=group_size,
            num_warmups=num_warmups,
            num_trials=num_trials,
            autocast_bfloat16=autocast_bfloat16,
            **MODEL,
        )
        handles.append(handle)

    results = [handle.get() for handle in handles]

    print()
    print(_format_table(results, num_trials, autocast_bfloat16))


"""
=== Gradient Checkpointing [bf16 autocast] (ms/step over 10 trials; 300 layers,
d_model=2048, d_ff=2048, ctx=128, num_heads=16, batch=4) ===   [NVIDIA H200]
strategy       time (ms)        peak (MiB)
-----------------------------------------------
recursive      2212.53 ± 78.40  560.1          <- lowest peak (O(log N) activations)
linear(g=1)    676.76 ± 68.32   1724.1
linear(g=2)    632.95 ± 78.34   1208.2
linear(g=3)    637.14 ± 74.40   1092.4
linear(g=4)    602.69 ± 67.24   1076.5         <- min peak among flat (= sweet spot)
linear(g=5)    565.03 ± 66.40   1100.6
linear(g=6)    591.50 ± 97.52   1144.7
linear(g=7)    592.71 ± 80.84   1200.8
linear(g=8)    613.13 ± 101.12  1264.9
linear(g=10)   640.20 ± 131.89  1401.1
linear(g=12)   549.39 ± 68.63   1549.4
linear(g=15)   574.66 ± 60.46   1781.7
linear(g=20)   560.18 ± 74.91   2182.3
linear(g=30)   558.49 ± 87.82   3003.4
linear(g=50)   588.75 ± 102.02  4669.7
linear(g=100)  560.34 ± 78.13   8863.3
linear(g=300)  572.42 ± 84.84   25597.9

Analysis
--------
Flat (linear) peak follows  peak(g) ~= (N/g)*a + g*b + C, with the empirical fit:
    a (one boundary [4,128,2048] fp32)  ~= 4 MiB
    b (one block's recompute residual)  ~= 84 MiB   ->  b/a ~= 21
    C (params + grads + bf16 copies)    ~= 290 MiB
  => optimum g* = sqrt(N*a/b) = sqrt(300*4/84) ~= 3.8  (matches measured min at g=4;
     neighbors g=3 -> 1092.4 and g=5 -> 1100.6 are both higher, confirming the min).

  Note this is < sqrt(N)=17.3: the textbook sqrt(N) assumes a ~= b, but a real block
  saves ~b/a ~= 20 boundaries' worth of residuals (Q/K/V/attn-out/FFN), so g* is pulled
  down. The hard floor is ~4-5 (Q/K/V/attn-out are each one boundary in size).

Recursive (nested) checkpointing holds only O(log N) live boundaries + one block, so
its 560 MiB peak is almost entirely fixed model state (params+grads+bf16 ~290 MiB) plus
a small activation tail (~84 MiB block + ~log2(300)~8 boundaries). It wins on memory by
~2x vs the best flat size but pays ~3.7x in time (O(N log N) recompute).

Earlier configs for reference (same harness):
  ctx=2048, N=50, d_model=1024, d_ff=4096: monotonic, min at g=1 (b/a ~= 41 -> g* ~= 1)
  ctx=512,  N=200, d_model=1024, d_ff=4096: U-shape, min at g=3 (b/a ~= 23 -> g* ~= 3)
"""
