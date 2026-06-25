"""Benchmark the Triton weighted-sum kernel vs torch raw `(x*weight).sum(-1)`.

Reports mean forward and forward+backward time for both, and the % improvement
of the Triton kernel over the torch reference (positive => Triton faster).

    modal run benchmark_weighted_sum.py
"""

from cs336_systems.modal import GPU, app, image

# (rows, D) — rows is the flattened leading dimension.
SIZES = [
    (8192, 1024),
    (16384, 2048),
    (32768, 1024),
    (16384, 8192),
]


@app.function(image=image, gpu=GPU, timeout=60 * 20)
def run(num_warmups: int, num_trials: int) -> list[dict]:
    import torch

    from cs336_systems.benchmarking import WeightedSumBenchOp, benchmark
    from cs336_systems.kernels.weighted_sum import WeightedSum

    assert (
        torch.cuda.is_available()
    ), "expected a CUDA device inside the Modal GPU container"
    print(f"running on {torch.cuda.get_device_name(0)}")

    def torch_raw(x, w):
        return (x * w).sum(dim=-1)

    impls = {"triton": WeightedSum.apply, "torch": torch_raw}
    results = []

    for rows, d in SIZES:
        row: dict[str, float] = {"rows": rows, "d": d}
        for name, fn in impls.items():
            for mode, backward in (("fwd", False), ("fwdbwd", True)):
                mean, _std = benchmark(
                    WeightedSumBenchOp(fn, rows, d, backward=backward),
                    num_warmups=num_warmups,
                    num_trials=num_trials,
                )
                row[f"{name}_{mode}"] = mean
        results.append(row)
        print(
            f"[{rows}x{d}] fwd: triton={row['triton_fwd']:.4f} torch={row['torch_fwd']:.4f} | "
            f"fwd+bwd: triton={row['triton_fwdbwd']:.4f} torch={row['torch_fwdbwd']:.4f} (ms)"
        )

    return results


def _improvement(torch_t: float, triton_t: float) -> float:
    # positive => Triton is faster than torch by this %
    return (torch_t - triton_t) / torch_t * 100.0


def _format(results: list[dict]) -> str:
    headers = [
        "rows",
        "D",
        "fwd torch",
        "fwd triton",
        "fwd impr%",
        "fb torch",
        "fb triton",
        "fb impr%",
    ]
    rows = [headers]
    for r in results:
        rows.append(
            [
                str(r["rows"]),
                str(r["d"]),
                f"{r['torch_fwd']:.4f}",
                f"{r['triton_fwd']:.4f}",
                f"{_improvement(r['torch_fwd'], r['triton_fwd']):+.1f}",
                f"{r['torch_fwdbwd']:.4f}",
                f"{r['triton_fwdbwd']:.4f}",
                f"{_improvement(r['torch_fwdbwd'], r['triton_fwdbwd']):+.1f}",
            ]
        )

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = "  "

    def fmt(r):
        return sep.join(c.rjust(widths[i]) for i, c in enumerate(r))

    line = "-" * (sum(widths) + len(sep) * (len(headers) - 1))
    return "\n".join(
        [
            "=== weighted_sum: Triton vs torch (ms; impr% = how much faster Triton is) ===",
            fmt(rows[0]),
            line,
            *[fmt(r) for r in rows[1:]],
        ]
    )


def _save_charts(results: list[dict]):
    from cs336_systems.plots import grouped_bar_chart

    rs = sorted(results, key=lambda r: r["rows"] * r["d"])
    cats = [f"{r['rows']}x{r['d']}" for r in rs]
    groups = {
        "triton fwd": [r["triton_fwd"] for r in rs],
        "torch fwd": [r["torch_fwd"] for r in rs],
        "triton f+b": [r["triton_fwdbwd"] for r in rs],
        "torch f+b": [r["torch_fwdbwd"] for r in rs],
    }
    grouped_bar_chart(
        "weighted_sum.png",
        "weighted_sum: Triton vs torch",
        "config (rows x D)",
        "latency (ms)",
        cats,
        groups,
    )


@app.local_entrypoint()
def main(num_warmups: int = 10, num_trials: int = 100):
    results = run.remote(num_warmups, num_trials)
    print()
    print(_format(results))
    _save_charts(results)


"""
=== weighted_sum: Triton vs torch (ms; impr% = how much faster Triton is) ===
 rows     D  fwd torch  fwd triton  fwd impr%  fb torch  fb triton  fb impr%
----------------------------------------------------------------------------
 8192  1024     0.0436      0.0535      -22.7    0.1612     0.2378     -47.5
16384  2048     0.1444      0.0707      +51.0    0.3874     0.2727     +29.6
32768  1024     0.1466      0.0702      +52.1    0.4028     0.2765     +31.4
16384  8192     0.5420      0.1890      +65.1    1.3777     0.5374     +61.0
Mean time for: 1.3777305638790132 ms ± 0.003724295501267915 ms
[16384x8192] fwd: triton=0.1890 torch=0.5420 | fwd+bwd: triton=0.5374 torch=1.3777 (ms)
"""
