from typing import cast
import torch
from einops import rearrange
from torch.autograd.function import FunctionCtx
import triton
import triton.language as tl


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # input pointers
    output_ptr,  # output pointer
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row_block = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight_block = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)

        output += tl.sum(row_block * weight_block[None, :], axis=1)  # (ROWS_TILE_SIZE,)

        # Move the pointers to the next tile
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide NUM_ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_bwd(
    x_ptr,
    weight_ptr,
    grad_output_ptr,
    grad_x_ptr,
    partial_grad_weight_ptr,
    stride_x_row,
    stride_x_dim,
    stride_w_dim,
    stride_grad_row,
    stride_grad_x_row,
    stride_grad_x_dim,
    stride_grad_w_batch,
    stride_grad_w_dim,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    num_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_grad_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )  # (ROWS_TILE_SIZE,)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_x_row, stride_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )  # (ROWS_TILE_SIZE, D_TILE_SIZE)

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_w_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )  # (D_TILE_SIZE,)

    # Outputs
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_grad_x_row, stride_grad_x_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )  # (ROWS_TILE_SIZE, D_TILE_SIZE)

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(num_row_tiles, D),
        strides=(stride_grad_w_batch, stride_grad_w_dim),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )  # (1, D_TILE_SIZE)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output_block = tl.load(
            grad_output_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (ROWS_TILE_SIZE,)
        x_block = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight_block = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)

        # Outer product for grad_x
        grad_x_block = (
            grad_output_block[:, None] * weight_block[None, :]
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        tl.store(grad_x_block_ptr, grad_x_block, boundary_check=(0, 1))

        # Partial grad wrt. weight
        partial_grad_weight_block = tl.sum(
            grad_output_block[:, None] * x_block, axis=0, keep_dims=True
        )  # (1, D_TILE_SIZE)
        tl.store(
            partial_grad_weight_block_ptr,
            partial_grad_weight_block,
            boundary_check=(1,),
        )

        # Move the pointers to the next tile
        x_block_ptr = tl.advance(x_block_ptr, (0, D_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (D_TILE_SIZE,))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = tl.advance(
            partial_grad_weight_block_ptr, (0, D_TILE_SIZE)
        )


class WeightedSum(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx, x: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradient wrt. x and weight
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape

        # Reshape input tensor into 2D
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)

        assert (
            len(weight.shape) == 1 and weight.shape[0] == D
        ), f"Weight must be a 1D tensor of shape ({D},)"
        assert x.is_cuda and weight.is_cuda, "x and weight must be on GPU"
        assert x.is_contiguous(), "x must be contiguous"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16  # 16 loops the embedding dim
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)
        y_flat = y.view(-1)

        # Launch our kernel with n instances in our 1D grid
        n_rows = y_flat.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y_flat,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y_flat.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,  # type: ignore
            D_TILE_SIZE=ctx.D_TILE_SIZE,  # type: ignore
        )

        return y.view(output_dims)

    @staticmethod
    def backward(
        ctx: FunctionCtx, *grad_outputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (grad_out,) = grad_outputs
        x, weight = cast(tuple[torch.Tensor, torch.Tensor], ctx.saved_tensors)

        ROW_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROW_TILE_SIZE), D), device=x.device, dtype=x.dtype
        )
        grad_x = torch.empty_like(x)

        # The kernel reads grad_out as a flat 1-D array of rows, so pass a
        # contiguous flat view (the caller's grad may be multi-dim / non-contiguous).
        grad_out_flat = grad_out.contiguous().view(-1)

        weighted_sum_bwd[(triton.cdiv(n_rows, ROW_TILE_SIZE),)](
            x,
            weight,
            grad_out_flat,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out_flat.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROW_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(dim=0)

        # grad of an input must match that input's original shape; x was flattened
        # to 2-D in forward, so restore the caller's leading dims.
        return grad_x.view(ctx.input_shape), grad_weight
