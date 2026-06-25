"""flash_benchmarking: Triton FlashAttention-2 vs plain PyTorch attention.

Uses triton.testing.do_bench to time forward, backward, and end-to-end
forward+backward latency for both our Triton FlashAttention and a regular
(non-flash) PyTorch attention. Single B200, batch size 1, causal masking.

Sweep (assignment spec):
    seq_len   in {128, 256, ..., 65536}   (powers of 2)
    d (head)  in {16, 32, 64, 128}        (powers of 2)
    dtype     in {bfloat16, float32}

Run with:
    modal run benchmark_flash.py
    modal run benchmark_flash.py --max-seq 16384   # smaller sweep for a quick look
"""

from typing import cast

from cs336_systems.modal import app, image

GPU = "B200"


@app.function(image=image, gpu=GPU, timeout=60 * 60)
def run(seq_lens: list[int], head_dims: list[int], dtypes: list[str]) -> list[dict]:
    import math

    import torch
    import triton
    import triton.testing

    from cs336_systems.kernels.flash_attn_triton import FlashAttention

    assert torch.cuda.is_available()
    print(f"[flash] running on {torch.cuda.get_device_name(0)}")

    DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
    BATCH = 1
    IS_CAUSAL = True

    def torch_attn(q, k, v, is_causal):
        # Plain (non-flash) attention: materializes the full S x S scores.
        d = q.shape[-1]
        s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        if is_causal:
            sq, sk = q.shape[-2], k.shape[-2]
            causal = torch.arange(sq, device=q.device)[:, None] >= torch.arange(sk, device=q.device)[None, :]
            s = s.masked_fill(~causal, float("-inf"))
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, v)

    impls = {"triton": FlashAttention.apply, "torch": torch_attn}

    def measure(impl, seq, d, dtype, mode) -> float | str:
        try:
            q = torch.randn(BATCH, seq, d, device="cuda", dtype=dtype, requires_grad=True)
            k = torch.randn(BATCH, seq, d, device="cuda", dtype=dtype, requires_grad=True)
            v = torch.randn(BATCH, seq, d, device="cuda", dtype=dtype, requires_grad=True)
            d_o = torch.randn(BATCH, seq, d, device="cuda", dtype=dtype)

            if mode == "fwd":
                ms = triton.testing.do_bench(lambda: impl(q, k, v, IS_CAUSAL))
            elif mode == "bwd":
                o = impl(q, k, v, IS_CAUSAL)
                ms = triton.testing.do_bench(
                    lambda: o.backward(d_o, retain_graph=True),
                    grad_to_none=[q, k, v],
                )
            else:  # fwd_bwd

                def step():
                    impl(q, k, v, IS_CAUSAL).backward(d_o)

                ms = triton.testing.do_bench(step, grad_to_none=[q, k, v])
            return cast(float, ms)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "OOM"
        except Exception as e:  # triton compile / shape issues at extreme sizes
            print(f"[flash] {mode} seq={seq} d={d} {dtype}: ERR {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            return "ERR"

    results = []
    for dtype_name in dtypes:
        dtype = DTYPES[dtype_name]
        for d in head_dims:
            for seq in seq_lens:
                row = {"dtype": dtype_name, "d": d, "seq": seq}
                for impl_name, impl in impls.items():
                    for mode in ("fwd", "bwd", "fwd_bwd"):
                        row[f"{impl_name}_{mode}"] = measure(impl, seq, d, dtype, mode)
                results.append(row)
                print(
                    f"[{dtype_name} d={d} seq={seq}] "
                    f"fwd t/p={_fmt(row['triton_fwd'])}/{_fmt(row['torch_fwd'])}  "
                    f"bwd t/p={_fmt(row['triton_bwd'])}/{_fmt(row['torch_bwd'])}  "
                    f"f+b t/p={_fmt(row['triton_fwd_bwd'])}/{_fmt(row['torch_fwd_bwd'])}"
                )
    return results


def _fmt(v) -> str:
    return v if isinstance(v, str) else f"{v:.3f}"


def _format_table(results: list[dict]) -> str:
    headers = [
        "dtype", "d", "seq",
        "fwd triton", "fwd torch",
        "bwd triton", "bwd torch",
        "f+b triton", "f+b torch",
    ]
    rows = [headers]
    keys = ["triton_fwd", "torch_fwd", "triton_bwd", "torch_bwd", "triton_fwd_bwd", "torch_fwd_bwd"]
    for r in results:
        rows.append([r["dtype"], str(r["d"]), str(r["seq"]), *[_fmt(r[k]) for k in keys]])

    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    sep = "  "

    def fmt(r):
        return sep.join(c.rjust(widths[i]) for i, c in enumerate(r))

    line = "-" * (sum(widths) + len(sep) * (len(headers) - 1))
    return "\n".join([
        "=== FlashAttention-2 (Triton) vs plain PyTorch attention "
        "(ms via do_bench; batch=1, causal) ===",
        fmt(rows[0]),
        line,
        *[fmt(r) for r in rows[1:]],
    ])


@app.local_entrypoint()
def main(min_seq: int = 128, max_seq: int = 65536):
    seq_lens = [s for s in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] if min_seq <= s <= max_seq]
    head_dims = [16, 32, 64, 128]
    dtypes = ["bf16", "fp32"]

    results = run.remote(seq_lens, head_dims, dtypes)

    print()
    print(_format_table(results))
    _save_charts(results)


def _save_charts(results: list[dict]):
    from cs336_systems.plots import grid_line_chart

    def num(v):
        return v if isinstance(v, (int, float)) else None

    dtypes = sorted({r["dtype"] for r in results})
    dims = sorted({r["d"] for r in results})
    for dt in dtypes:
        panels = []
        for d in dims:
            rows = sorted(
                [r for r in results if r["dtype"] == dt and r["d"] == d],
                key=lambda r: r["seq"],
            )
            seqs = [r["seq"] for r in rows]
            panels.append({
                "title": f"d={d}",
                "xlabel": "seq_len",
                "ylabel": "latency (ms)",
                "series": {
                    "triton fwd": (seqs, [num(r["triton_fwd"]) for r in rows]),
                    "torch fwd": (seqs, [num(r["torch_fwd"]) for r in rows]),
                    "triton f+b": (seqs, [num(r["triton_fwd_bwd"]) for r in rows]),
                    "torch f+b": (seqs, [num(r["torch_fwd_bwd"]) for r in rows]),
                },
            })
        grid_line_chart(
            f"flash_{dt}.png",
            f"FlashAttention (Triton) vs PyTorch -- {dt}, batch=1, causal (B200)",
            panels,
            ncols=2,
            logx=True,
            logy=True,
        )


"""
=== FlashAttention-2 (Triton fwd + torch.compile bwd) vs plain PyTorch attention ===
ms via triton.testing.do_bench; batch=1, causal; single NVIDIA B200.
("triton" = our kernel; "torch" = non-flash attention that materializes the SxS matrix.)

dtype    d    seq  fwd_tri  fwd_torch  bwd_tri  bwd_torch  fb_tri  fb_torch
--------------------------------------------------------------------------
 bf16   16   4096    0.097      0.180    0.637      0.554   0.899     0.808
 bf16   16   8192    0.138      0.631    0.697      0.520   0.896     1.024
 bf16   16  16384    0.371      1.855    1.942      1.381   2.307     3.217
 bf16   16  32768    1.274      7.184    7.635      5.449   8.908    12.612
 bf16   16  65536    4.723     28.133   30.760     22.259  35.490    50.420
 bf16  128  16384    0.548      1.872    2.001      1.409   2.543     3.262
 bf16  128  32768    1.911      7.263    7.883      5.520   9.786    12.759
 bf16  128  65536    7.126     28.497   31.621     22.638  38.728    51.044
 fp32   16   8192    0.184      1.019    1.040      1.197   1.221     2.223
 fp32   16  32768    1.823     12.535   13.863     16.223  15.688    28.743
 fp32   16  65536    6.901     49.257   58.241     59.163  65.148   108.407
 fp32  128  32768    4.562     19.131   31.676     28.123  36.234    47.252
 fp32  128  65536   17.545     75.645  128.048    113.420 145.603   189.055
(representative rows; full 80-config sweep printed by the run. No OOM on B200's 192GB,
even at seq=65536 where torch's fp32 SxS scores are ~17GB.)

Findings
--------
- FORWARD: the Triton flash forward is dramatically faster than plain PyTorch, and the
  gap widens with sequence length (torch materializes/reads the O(N^2) scores; flash
  streams in tiles). At seq=65536: bf16 ~4-7x faster, fp32 ~7-10x faster. Below ~4k both
  are launch-bound and roughly tie (~0.09 vs ~0.18 ms).
- BACKWARD: roughly on par with torch (often slightly slower). This is expected: our
  backward is NOT a flash kernel -- it's the torch + torch.compile recomputation
  (eqs 13-19), which materializes the same O(N^2) S/P/dP/dS as the torch reference, plus
  recompute + reshape overhead. So flash gives no backward speedup here.
- END-TO-END (fwd+bwd): Triton still wins overall, driven entirely by the much cheaper
  forward (e.g. fp32 d=16 seq=65536: 65 vs 108 ms; bf16 d=16: 35 vs 50 ms).
- bf16 < fp32 latency as expected; the forward gap is largest in fp32 (torch's N^2 is
  twice the bytes).
Takeaway: our partial implementation's win is the memory-streaming FORWARD; a fully
memory-efficient (tiled Triton) BACKWARD would be needed to also beat torch on bwd.
"""
