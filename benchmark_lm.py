"""Root-level Modal entrypoint for the benchmarking harness.

All timing logic lives in ``cs336_systems/benchmarking.py``; this file is just
the Modal function + CLI that ships it to cloud GPUs and runs the full suite.

Each model size from the assignment's Table 1 is dispatched to its own
container, so all sizes run concurrently. The local entrypoint gathers the
results and prints a formatted table.

Run with:

    modal run benchmark.py
    modal run benchmark.py --num-trials 50 --vocab-size 50257
"""

from cs336_systems.modal import GPU, app, image

# Table 1: GPT-2-style model specs. (context length defaults to 512 per handout.)
MODEL_SIZES: list[dict] = [
    {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {
        "name": "medium",
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"name": "xl", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    {"name": "10B", "d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
]

# Order of ops reported per model size.
OP_NAMES = ["forward", "backward", "forward_backward", "forward_backward_optimizer"]


@app.function(image=image, gpu=GPU, timeout=60 * 30)
def run_benchmark(
    name: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    num_warmups: int,
    num_trials: int,
    autocast_bfloat16: bool = False,
    profile_memory: bool = False,
    compile_model: bool = False,
) -> dict[str, object]:
    import tempfile
    from pathlib import Path

    import torch

    from cs336_systems.benchmarking import (
        BackwardOp,
        ForwardBackwardOp,
        ForwardOp,
        LMConfig,
        benchmark,
    )

    assert (
        torch.cuda.is_available()
    ), "expected a CUDA device inside the Modal GPU container"
    print(f"[{name}] running on {torch.cuda.get_device_name(0)}")

    config = LMConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    ops = {
        "forward": lambda: ForwardOp(config, compile_model=compile_model),
        "backward": lambda: BackwardOp(config, compile_model=compile_model),
        "forward_backward": lambda: ForwardBackwardOp(
            config, with_optimizer=False, compile_model=compile_model
        ),
        "forward_backward_optimizer": lambda: ForwardBackwardOp(
            config, with_optimizer=True, compile_model=compile_model
        ),
    }

    results: dict[str, dict[str, float] | None] = {}
    # Pickle bytes per op; shipped back to the local entrypoint to write under .profile/.
    snapshots: dict[str, bytes] = {}
    for op_name in OP_NAMES:
        print(f"[{name}] --- {op_name} ---")
        snapshot_path = None
        if profile_memory:
            # Container-local (ephemeral) path; we read the bytes back below.
            snapshot_path = str(
                Path(tempfile.gettempdir()) / f"{name}_{op_name}.pickle"
            )
        try:
            raw = benchmark(
                ops[op_name](),
                num_warmups=num_warmups,
                num_trials=num_trials,
                autocast_bfloat16=autocast_bfloat16,
                memory_snapshot_path=snapshot_path,
            )

            # Accept either `mean` or `(mean, std)` so this works whether or not
            # benchmark() has been updated to also return the standard deviation.
            if isinstance(raw, tuple):
                mean, std = raw
            else:
                mean, std = raw, None
            results[op_name] = {"mean": mean, "std": std}

            if snapshot_path is not None and Path(snapshot_path).exists():
                snapshots[op_name] = Path(snapshot_path).read_bytes()
        except torch.cuda.OutOfMemoryError:
            print(f"[{name}] {op_name}: OOM")
            results[op_name] = None
            torch.cuda.empty_cache()
    return {"results": results, "snapshots": snapshots}


def _format_cell(op_result: dict[str, float] | None) -> str:
    if op_result is None:
        return "OOM"
    mean = op_result["mean"]
    std = op_result.get("std")
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _format_table(
    results: dict[str, dict[str, dict[str, float] | None]], num_trials: int
) -> str:
    headers = ["Size", "forward", "backward", "fwd+bwd", "fwd+bwd+opt"]
    rows = [headers]
    for size in MODEL_SIZES:
        name = size["name"]
        res = results.get(name, {})
        row = [name]
        for op_name in OP_NAMES:
            row.append(_format_cell(res.get(op_name)))
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = "  "

    def fmt_row(r: list[str]) -> str:
        return sep.join(cell.rjust(widths[i]) for i, cell in enumerate(r))

    line = "-" * (sum(widths) + len(sep) * (len(headers) - 1))
    out = [
        f"=== Benchmark Results (ms/step, mean ± std over {num_trials} trials) ===",
        fmt_row(rows[0]),
        line,
        *[fmt_row(r) for r in rows[1:]],
    ]
    return "\n".join(out)


def _spawn_sweep(
    autocast_bfloat16: bool,
    vocab_size: int,
    context_length: int,
    num_warmups: int,
    num_trials: int,
    profile_memory: bool = False,
    compile_model: bool = False,
) -> list[tuple[str, object]]:
    # Spawn one container per model size (non-blocking) so they run concurrently.
    handles = []
    for size in MODEL_SIZES:
        handle = run_benchmark.spawn(
            name=size["name"],
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=size["d_model"],
            d_ff=size["d_ff"],
            num_layers=size["num_layers"],
            num_heads=size["num_heads"],
            num_warmups=num_warmups,
            num_trials=num_trials,
            autocast_bfloat16=autocast_bfloat16,
            profile_memory=profile_memory,
            compile_model=compile_model,
        )
        handles.append((size["name"], handle))
    return handles


def _gather(handles: list[tuple[str, object]]) -> dict[str, dict[str, object]]:
    return {name: handle.get() for name, handle in handles}


def _results_only(
    gathered: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, float] | None]]:
    return {name: payload["results"] for name, payload in gathered.items()}


def _write_snapshots(
    gathered: dict[str, dict[str, object]], precision: str, out_dir: str = ".profile"
) -> int:
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, payload in gathered.items():
        for op_name, data in payload.get("snapshots", {}).items():
            dest = out / f"{precision}_{name}_{op_name}.pickle"
            dest.write_bytes(data)
            written += 1
            print(f"wrote {dest} ({len(data)} bytes)")
    return written


@app.local_entrypoint()
def main(
    vocab_size: int = 10000,
    context_length: int = 512,
    num_warmups: int = 15,
    num_trials: int = 50,
    profile_memory: bool = False,
    compare_compile: bool = False,
):
    # torch_compile (b): vanilla vs torch.compile'd full Transformer (fp32).
    if compare_compile:
        vanilla_handles = _spawn_sweep(
            False,
            vocab_size,
            context_length,
            num_warmups,
            num_trials,
            compile_model=False,
        )
        compiled_handles = _spawn_sweep(
            False,
            vocab_size,
            context_length,
            num_warmups,
            num_trials,
            compile_model=True,
        )
        vanilla = _gather(vanilla_handles)
        compiled = _gather(compiled_handles)
        print("\n### VANILLA (eager, fp32) ###")
        print(_format_table(_results_only(vanilla), num_trials))
        print("\n### torch.compile (fp32) ###")
        print(_format_table(_results_only(compiled), num_trials))
        return

    # Spawn BOTH sweeps first (all 10 containers launch), then gather — so fp32
    # and bf16 run concurrently rather than one sweep after the other.
    fp32_handles = _spawn_sweep(
        False, vocab_size, context_length, num_warmups, num_trials, profile_memory
    )
    bf16_handles = _spawn_sweep(
        True, vocab_size, context_length, num_warmups, num_trials, profile_memory
    )

    fp32 = _gather(fp32_handles)
    bf16 = _gather(bf16_handles)

    print("\n### FULL PRECISION (fp32) ###")
    print(_format_table(_results_only(fp32), num_trials))
    print("\n### MIXED PRECISION (bf16 autocast) ###")
    print(_format_table(_results_only(bf16), num_trials))

    if profile_memory:
        print(
            "\n### Memory snapshots -> .profile/ (load at https://pytorch.org/memory_viz) ###"
        )
        n = _write_snapshots(fp32, "fp32") + _write_snapshots(bf16, "bf16")
        print(f"wrote {n} snapshot file(s)")


"""
### FULL PRECISION (fp32) ###
=== Benchmark Results (ms/step, mean ± std over 50 trials) ===
  Size          forward         backward           fwd+bwd      fwd+bwd+opt
---------------------------------------------------------------------------
 small   13.427 ± 0.139   21.084 ± 0.238    34.702 ± 0.690   48.244 ± 0.799
medium   24.436 ± 0.497   41.691 ± 0.866    68.020 ± 0.998   91.040 ± 1.327
 large   38.324 ± 0.071   83.421 ± 0.316   120.795 ± 0.134  161.573 ± 0.185
    xl   90.165 ± 0.078  217.067 ± 0.164   307.351 ± 0.165  428.932 ± 0.314
   10B  295.671 ± 0.866  708.629 ± 0.072  1001.089 ± 0.591              OOM

### MIXED PRECISION (bf16 autocast) ###
=== Benchmark Results (ms/step, mean ± std over 50 trials) ===
  Size         forward         backward          fwd+bwd      fwd+bwd+opt
-------------------------------------------------------------------------
 small  13.511 ± 0.034   22.968 ± 0.210   35.368 ± 0.083   46.899 ± 0.331
medium  31.255 ± 0.222   53.798 ± 0.265   81.156 ± 0.672  111.184 ± 0.370
 large  41.052 ± 0.250   69.693 ± 3.447  105.408 ± 1.038  144.338 ± 1.207
    xl  36.511 ± 0.485   85.749 ± 0.063  107.263 ± 0.868  230.229 ± 1.539
   10B  62.628 ± 0.050  255.622 ± 0.105  282.467 ± 0.093              OOM
"""


"""
torch_compile (b): vanilla vs torch.compile'd full Transformer
Run with:  modal run benchmark_lm.py --compare-compile
(ms/step, mean +/- std over 50 trials, fp32, context_length=512, NVIDIA H200)

### VANILLA (eager, fp32) ###
  Size          forward         backward           fwd+bwd      fwd+bwd+opt
---------------------------------------------------------------------------
 small   12.684 ± 0.071   20.921 ± 0.069    33.677 ± 0.067   47.032 ± 0.470
medium   27.959 ± 0.402   42.194 ± 0.570    70.468 ± 0.238   97.307 ± 1.321
 large   38.199 ± 0.045   82.278 ± 0.055   120.786 ± 0.102  161.528 ± 0.157
    xl   90.754 ± 0.130  217.195 ± 0.067   308.282 ± 0.146  429.477 ± 0.243
   10B  298.374 ± 0.807  710.221 ± 0.676  1003.602 ± 0.444              OOM

### torch.compile (fp32) ###
  Size          forward         backward          fwd+bwd      fwd+bwd+opt
--------------------------------------------------------------------------
 small    5.112 ± 0.031   10.696 ± 0.013   16.049 ± 0.050   24.430 ± 0.552
medium   14.102 ± 0.021   28.297 ± 0.026   42.645 ± 0.041   62.349 ± 0.069
 large   31.444 ± 0.062   68.642 ± 0.075  100.595 ± 0.180  143.475 ± 6.217
    xl   81.682 ± 0.079  199.354 ± 0.063  281.926 ± 0.158  403.644 ± 0.196
   10B  281.757 ± 1.088  669.071 ± 0.194  944.826 ± 0.319              OOM

Response: torch.compile speeds up every configuration. The forward pass benefits
most (small 12.7->5.1 ms ~2.5x, medium ~2x, large/xl ~1.1-1.2x), since fusing the
many small pointwise/normalization kernels removes Python+launch overhead that
dominates the cheaper forward pass. The combined forward+backward and
forward+backward+optimizer steps also improve consistently (e.g. xl fwd+bwd
308->282 ms, fwd+bwd+opt 429->404 ms; small fwd+bwd+opt 47->24 ms ~1.9x), though the
relative gain shrinks for the larger models because they are already matmul-bound
(the GEMMs, which compile can't make much faster, are the bottleneck). The optimizer
step itself is elementwise and benefits from fusion too. Speedup is largest where
overhead is the bottleneck (small models / forward only) and smallest where the work
is already big dense matmuls (xl/10B). One-time compilation cost is paid during
warm-up and excluded from the timed trials.
"""
