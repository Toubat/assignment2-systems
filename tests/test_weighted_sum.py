"""Tests for the Triton weighted-sum kernel.

The kernel computes, per row, ``output[...] = sum_d x[..., d] * weight[d]``
(i.e. a weighted reduction over the last dim). We compare against the obvious
PyTorch reference and require the max difference to be within a small epsilon.

Triton kernels require a GPU, so these are skipped unless CUDA is available
(run them on a GPU via ``modal run run_kernel_tests.py``).
"""

import pytest
import torch

from cs336_systems.kernels.weighted_sum import WeightedSum


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # output[...] = sum_d x[..., d] * weight[d]
    return (x * weight).sum(dim=-1)


# (*leading_dims, D). Includes: rows not divisible by ROWS_TILE_SIZE (=16),
# 3-D leading dims, and D values that are / aren't powers of two. D must be
# >= 16 since the kernel uses D_TILE_SIZE = next_power_of_2(D) // 16.
SHAPES = [
    (32, 16),
    (128, 64),
    (61, 128),  # rows % 16 != 0  -> exercises the boundary check
    (4, 50, 200),  # 3-D leading dims, D not a power of two
    (8, 17, 768),
]

RTOL = 1e-3
ATOL = 1e-3


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_weighted_sum_matches_torch(shape):
    torch.manual_seed(0)
    *lead, d = shape
    x = torch.randn(*lead, d, device="cuda", dtype=torch.float32)
    weight = torch.randn(d, device="cuda", dtype=torch.float32)

    out = WeightedSum.apply(x, weight)
    ref = _reference(x, weight)

    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    # max |diff| < epsilon (assert_close checks |out-ref| <= atol + rtol*|ref|)
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
def test_weighted_sum_max_abs_diff_under_epsilon():
    """Explicit max-abs-diff check, in addition to assert_close."""
    torch.manual_seed(1)
    x = torch.randn(256, 512, device="cuda", dtype=torch.float32)
    weight = torch.randn(512, device="cuda", dtype=torch.float32)

    out = WeightedSum.apply(x, weight)
    ref = _reference(x, weight)

    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-2, f"max abs diff too large: {max_abs_diff}"


def _grads(apply_fn, x: torch.Tensor, weight: torch.Tensor, grad_out: torch.Tensor):
    """Run apply_fn's forward + backward on fresh leaves, return (grad_x, grad_weight)."""
    x = x.detach().clone().requires_grad_(True)
    weight = weight.detach().clone().requires_grad_(True)
    out = apply_fn(x, weight)
    out.backward(grad_out)
    return x.grad, weight.grad


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU must be available to run Triton kernels",
)
@pytest.mark.parametrize("shape", SHAPES)
def test_weighted_sum_backward_matches_torch(shape):
    # NOTE: requires WeightedSum.backward to be implemented.
    # Reference grads (out = sum_d x[...,d] * weight[d]):
    #   grad_x[...,d]   = grad_out[...] * weight[d]
    #   grad_weight[d]  = sum over all rows of grad_out[row] * x[row, d]
    torch.manual_seed(0)
    *lead, d = shape
    x = torch.randn(*lead, d, device="cuda", dtype=torch.float32)
    weight = torch.randn(d, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(*lead, device="cuda", dtype=torch.float32)

    ref_grad_x, ref_grad_w = _grads(_reference, x, weight, grad_out)
    grad_x, grad_w = _grads(WeightedSum.apply, x, weight, grad_out)

    torch.testing.assert_close(grad_x, ref_grad_x, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(grad_w, ref_grad_w, rtol=RTOL, atol=ATOL)
