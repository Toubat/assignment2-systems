from einops import rearrange
import torch
from torch.autograd.function import FunctionCtx

import triton.language as tl
import triton


@triton.jit
def flash_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    logsumexp_ptr,
    stride_q_batch,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_seq,
    stride_v_dim,
    stride_o_batch,
    stride_o_seq,
    stride_o_dim,
    stride_logsumexp_batch,
    stride_logsumexp_seq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    # Inputs
    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch_idx * stride_q_batch,
        shape=(N_QUERIES, D),
        strides=(stride_q_seq, stride_q_dim),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch_idx * stride_k_batch,
        shape=(N_KEYS, D),
        strides=(stride_k_seq, stride_k_dim),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch_idx * stride_v_batch,
        shape=(N_KEYS, D),
        strides=(stride_v_seq, stride_v_dim),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # Outputs
    o_block_ptr = tl.make_block_ptr(
        o_ptr + batch_idx * stride_o_batch,
        shape=(N_QUERIES, D),
        strides=(stride_o_seq, stride_o_dim),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    logsumexp_block_ptr = tl.make_block_ptr(
        logsumexp_ptr + batch_idx * stride_logsumexp_batch,
        shape=(N_QUERIES,),
        strides=(stride_logsumexp_seq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Buffers
    q_i = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    logsumexp_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    for k_tile_idx in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_j = tl.load(
            k_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (K_TILE_SIZE, D)
        v_j = tl.load(
            v_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (K_TILE_SIZE, D)

        s_ij = tl.dot(q_i, tl.trans(k_j)) * scale  # (Q_TILE_SIZE, K_TILE_SIZE)
        s_ij_row_max = tl.max(s_ij, axis=-1)  # (Q_TILE_SIZE,)

        m_i_prev, m_i = m_i, tl.maximum(m_i, s_ij_row_max)  # (Q_TILE_SIZE,)
        exp_m_diff = tl.exp(m_i_prev - m_i)  # (Q_TILE_SIZE,)

        p_ij = tl.exp(s_ij - m_i[:, None])  # (Q_TILE_SIZE, K_TILE_SIZE)
        p_ij_row_sum = tl.sum(p_ij, axis=-1)  # (Q_TILE_SIZE,)
        l_i = exp_m_diff * l_i + p_ij_row_sum

        # Accumulate the result of the dot product into o_i
        o_i = exp_m_diff[:, None] * o_i
        o_i = tl.dot(p_ij.to(v_j.dtype), v_j, acc=o_i)

        # Advance the pointers to the next tile
        k_block_ptr = tl.advance(k_block_ptr, (K_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (K_TILE_SIZE, 0))

    o_i = o_i / l_i[:, None]
    logsumexp_i = m_i + tl.log(l_i)

    tl.store(o_block_ptr, o_i, boundary_check=(0, 1))
    tl.store(logsumexp_block_ptr, logsumexp_i, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ):
        *input_dims, q_seq_len, D_Q = q.shape
        *_, k_seq_len, D_K = k.shape
        *_, v_seq_len, D_V = v.shape

        assert q.is_cuda and k.is_cuda and v.is_cuda, "q, k, and v must be on GPU"
        assert k_seq_len == v_seq_len, "q, k, and v must have the same sequence length"
        assert D_Q == D_K == D_V, "q, k, and v must have the same dimension"

        q = rearrange(q, "... q d -> (...) q d")
        k = rearrange(k, "... k d -> (...) k d")
        v = rearrange(v, "... v d -> (...) v d")

        ctx.D = D_Q
        ctx.N_QUERIES = q_seq_len
        ctx.N_KEYS = k_seq_len
        ctx.Q_TILE_SIZE = 16

        # tl.dot requires all matmul dims >= 16, so the key tile must be >= 16.
        ctx.K_TILE_SIZE = max(16, triton.next_power_of_2(k_seq_len) // 16)
        ctx.q_shape = q.shape
        ctx.k_shape = k.shape

        n_tiles, n_rows = triton.cdiv(ctx.N_QUERIES, ctx.Q_TILE_SIZE), q.shape[0]

        o = torch.empty((n_rows, q_seq_len, ctx.D), device=q.device, dtype=q.dtype)
        logsumexp = torch.empty((n_rows, q_seq_len), device=q.device, dtype=q.dtype)

        flash_fwd_kernel[(n_tiles, n_rows)](
            q,
            k,
            v,
            o,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            N_QUERIES=ctx.N_QUERIES,
            N_KEYS=ctx.N_KEYS,
            scale=1 / (ctx.D**0.5),
            D=ctx.D,  # type: ignore
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,  # type: ignore
            K_TILE_SIZE=ctx.K_TILE_SIZE,  # type: ignore
        )

        ctx.save_for_backward(logsumexp, q, k, v, o)
        return o.view(*input_dims, q_seq_len, ctx.D)

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        raise NotImplementedError
