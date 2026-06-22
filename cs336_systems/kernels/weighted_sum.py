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
