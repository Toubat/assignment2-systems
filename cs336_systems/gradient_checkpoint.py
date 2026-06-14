import torch
from torch.utils.checkpoint import checkpoint
from typing import Callable


def linear_checkpoint(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_layers: int,
    group_size: int,
):

    def make_group_fn(count: int):
        def group_fn(x: torch.Tensor):
            for _ in range(count):
                x = fn(x)
            return x

        return group_fn

    remaining = num_layers
    while remaining > 0:
        count = min(group_size, remaining)
        x = checkpoint(make_group_fn(count), x, use_reentrant=False)
        remaining -= count
    return x


def recursive_checkpoint(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, num_layers: int
):

    def recurse(x: torch.Tensor, start: int, end: int):
        if start == end:
            return fn(x)

        mid = (start + end) // 2

        def run_left(x: torch.Tensor):
            return recurse(x, start, mid)

        def run_right(x: torch.Tensor):
            return recurse(x, mid + 1, end)

        x = checkpoint(run_left, x, use_reentrant=False)
        x = checkpoint(run_right, x, use_reentrant=False)

        return x

    def run_all(x: torch.Tensor):
        return recurse(x, 0, num_layers - 1)

    return checkpoint(run_all, x, use_reentrant=False)
