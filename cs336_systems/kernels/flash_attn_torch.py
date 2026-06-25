from math import sqrt
from einops import einsum
import torch
from torch.autograd.function import FunctionCtx

import triton


class FlashAttentionTorch(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,  # (..., seq_len, d)
        k: torch.Tensor,  # (..., seq_len, d)
        v: torch.Tensor,  # (..., seq_len, d)
        is_causal: bool,
    ):
        *input_dims, _, D_Q = q.shape
        *_, k_seq_len, D_K = k.shape
        *_, v_seq_len, D_V = v.shape

        assert k_seq_len == v_seq_len, "q, k, and v must have the same sequence length"
        assert D_Q == D_K == D_V, "q, k, and v must have the same dimension"

        ctx.D = D_Q
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = max(16, triton.next_power_of_2(k_seq_len) // 16)

        q_tiles = torch.split(q, ctx.Q_TILE_SIZE, dim=-2)  # (..., T_Q, D)
        k_tiles = torch.split(k, ctx.K_TILE_SIZE, dim=-2)  # (..., T_K, D)
        v_tiles = torch.split(v, ctx.K_TILE_SIZE, dim=-2)  # (..., T_V, D)

        o_tiles = []  # (..., T_Q, D) -> (..., seq_len_q, D)
        logsumexp_tiles = []  # (..., T_Q) -> (..., seq_len_q)
        for i in range(len(q_tiles)):
            q_i = q_tiles[i]  # (..., T_Q, D)
            o_i = torch.zeros_like(q_i)  # (..., T_Q, D); zeros so 0*garbage can't NaN
            l_i = torch.zeros(
                (*input_dims, ctx.Q_TILE_SIZE), device=q.device
            )  # (..., T_Q)
            m_i = torch.full_like(l_i, float("-inf"))  # (..., T_Q)

            for j in range(len(k_tiles)):
                k_j, v_j = k_tiles[j], v_tiles[j]  # (..., T_K, D), (..., T_K, D)

                s_ij = einsum(q_i, k_j, "... q d, ... k d -> ... q k") / sqrt(
                    ctx.D
                )  # (..., T_Q, T_K)
                s_ij_row_max = s_ij.max(dim=-1).values  # (..., T_Q)

                m_i_prev, m_i = m_i, torch.maximum(m_i, s_ij_row_max)  # (..., T_Q)
                exp_m_diff = torch.exp(m_i_prev - m_i)  # (..., T_Q)

                p_ij = torch.exp(s_ij - m_i.unsqueeze(-1))  # (..., T_Q, T_K)
                p_ij_row_sum = p_ij.sum(dim=-1)  # (..., T_Q)

                l_i = exp_m_diff * l_i + p_ij_row_sum  # (..., T_Q)
                o_i = exp_m_diff.unsqueeze(-1) * o_i + p_ij @ v_j  # (..., T_Q, D)

            o_i = o_i / l_i.unsqueeze(-1)
            logsum_exp_i = m_i + torch.log(l_i)

            o_tiles.append(o_i)
            logsumexp_tiles.append(logsum_exp_i)

        o = torch.cat(o_tiles, dim=-2)
        logsumexp = torch.cat(logsumexp_tiles, dim=-1)

        ctx.save_for_backward(logsumexp, q, k, v, o)

        return o

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        d_o = grad_outputs[0]  # (..., q, d)
        logsumexp, q, k, v, o = ctx.saved_tensors

        d_q, d_k, d_v = _flash_attn_backward_torch(
            q, k, v, o, d_o, logsumexp, 1 / sqrt(ctx.D)
        )
        return d_q, d_k, d_v, None


@torch.compile
def _flash_attn_backward_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    is_causal: bool = False,
):
    # Works on any leading dims; q/k/v/o are (..., seq, d), logsumexp is (..., q).
    d = torch.sum(o * d_o, dim=-1)  # (..., q)
    s = einsum(q, k, "... q d, ... k d -> ... q k") * scale  # (..., q, k)

    if is_causal:
        # Mask future keys to -inf so they recompute to P=0 (matches the causal
        # forward, whose L already excludes them).
        q_idx = torch.arange(s.shape[-2], device=s.device)
        k_idx = torch.arange(s.shape[-1], device=s.device)
        s = s.masked_fill(q_idx[:, None] < k_idx[None, :], float("-inf"))

    p = torch.exp(s - torch.unsqueeze(logsumexp, -1))  # (..., q, k)
    # logsumexp is fp32, so p promoted to fp32; cast back so the einsums below
    # stay dtype-consistent with q/k/v/d_o (e.g. bf16).
    p = p.to(q.dtype)

    d_v = einsum(p, d_o, "... q k, ... q d -> ... k d")  # (..., k, d)
    d_p = einsum(d_o, v, "... q d, ... k d -> ... q k")  # (..., q, k)
    d_s = p * (d_p - torch.unsqueeze(d, -1))  # (..., q, k)

    d_q = einsum(d_s, k, "... q k, ... k d -> ... q d") * scale  # (..., q, d)
    d_k = einsum(d_s, q, "... q k, ... q d -> ... k d") * scale  # (..., k, d)

    return d_q, d_k, d_v
