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
) -> dict[str, dict[str, float] | None]:
    import torch

    from cs336_systems.benchmarking import (
        BackwardOp,
        ForwardBackwardOp,
        ForwardOp,
        LMConfig,
        benchmark,
    )

    assert torch.cuda.is_available(), "expected a CUDA device inside the Modal GPU container"
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
        "forward": lambda: ForwardOp(config),
        "backward": lambda: BackwardOp(config),
        "forward_backward": lambda: ForwardBackwardOp(config, with_optimizer=False),
        "forward_backward_optimizer": lambda: ForwardBackwardOp(config, with_optimizer=True),
    }

    results: dict[str, dict[str, float] | None] = {}
    for op_name in OP_NAMES:
        print(f"[{name}] --- {op_name} ---")
        try:
            raw = benchmark(ops[op_name](), num_warmups=num_warmups, num_trials=num_trials)
            # Accept either `mean` or `(mean, std)` so this works whether or not
            # benchmark() has been updated to also return the standard deviation.
            if isinstance(raw, tuple):
                mean, std = raw
            else:
                mean, std = raw, None
            results[op_name] = {"mean": mean, "std": std}
        except torch.cuda.OutOfMemoryError:
            print(f"[{name}] {op_name}: OOM")
            results[op_name] = None
            torch.cuda.empty_cache()
    return results


def _format_cell(op_result: dict[str, float] | None) -> str:
    if op_result is None:
        return "OOM"
    mean = op_result["mean"]
    std = op_result.get("std")
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def _format_table(results: dict[str, dict[str, dict[str, float] | None]], num_trials: int) -> str:
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


@app.local_entrypoint()
def main(
    vocab_size: int = 10000,
    context_length: int = 512,
    num_warmups: int = 15,
    num_trials: int = 50,
):
    # Spawn one container per model size so all sizes run concurrently.
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
        )
        handles.append((size["name"], handle))

    results: dict[str, dict[str, dict[str, float] | None]] = {}
    for name, handle in handles:
        results[name] = handle.get()

    print("\n" + _format_table(results, num_trials))


"""
=== Benchmark Results (ms/step, mean ± std over 50 trials) ===
  Size          forward         backward          fwd+bwd      fwd+bwd+opt
--------------------------------------------------------------------------
 small   14.458 ± 0.459   22.932 ± 0.850   37.338 ± 0.844   51.603 ± 1.420
medium   42.941 ± 0.939   60.613 ± 3.344  104.344 ± 4.600  139.607 ± 4.857
 large   38.605 ± 0.036   85.545 ± 0.038  124.651 ± 0.106  176.330 ± 1.513
    xl   92.384 ± 0.325  225.470 ± 0.118  317.775 ± 0.225  485.795 ± 0.398
   10B  301.374 ± 0.890              OOM              OOM              OOM
"""
