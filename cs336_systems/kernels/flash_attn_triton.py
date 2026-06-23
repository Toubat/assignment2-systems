import torch
from torch.autograd.function import FunctionCtx


class FlashAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool,
    ):
        pass

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        raise NotImplementedError
