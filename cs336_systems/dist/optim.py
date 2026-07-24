import torch
import torch.distributed as dist

from torch import optim
from torch.optim.optimizer import ParamsT
from typing import Any, Callable, Type


class ShardedOptimizer(optim.Optimizer):

    def __init__(
        self,
        params: ParamsT,
        optimizer_cls: Type[optim.Optimizer],
        **kwargs: Any,
    ):
        # get rank using world size and rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.global_idx = 0
        self.global_param_to_rank: list[tuple[torch.Tensor, int]] = []
        self.local_param_groups: list[dict[str, Any]] = []

        super().__init__(params, {})
        self.optimizer = optimizer_cls(params=self.local_param_groups, **kwargs)

    def step(  # type: ignore
        self,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ):
        result = self.optimizer.step(closure, **kwargs)

        handles: list[dist.Work] = []

        with torch.no_grad():
            for param, rank in self.global_param_to_rank:
                handle = dist.broadcast(param, src=rank, async_op=True)
                if handle is not None:
                    handles.append(handle)

        for handle in handles:
            handle.wait()

        return result

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)

        params = param_group["params"]
        group_config = {k: v for k, v in param_group.items() if k != "params"}

        local_params = []
        for param in params:
            param_rank = self.global_idx % self.world_size
            self.global_param_to_rank.append((param, param_rank))
            if param_rank == self.rank:
                local_params.append(param)
            self.global_idx += 1

        local_param_group = {**group_config, "params": local_params}
        self.local_param_groups.append(local_param_group)

        if hasattr(self, "optimizer") and isinstance(self.optimizer, optim.Optimizer):
            self.optimizer.add_param_group(local_param_group)
