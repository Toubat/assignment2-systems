from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_basics.model import Linear, RMSNorm
from cs336_systems.dist.ddp import FSDP

from .common import _cleanup_process_group, _setup_process_group


class _TwoLinearLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Linear(4, 4)
        self.second = Linear(4, 4)
        self.between_layers: Callable[[], None] | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Deliberately execute in the opposite order from module registration.
        hidden = self.second(inputs)
        if self.between_layers is not None:
            self.between_layers()
        return self.first(hidden)


def test_fsdp_mixed_precision_produces_fp32_local_gradient_shards():
    mp.spawn(
        _test_fsdp_mixed_precision_produces_fp32_local_gradient_shards,
        args=(2,),
        nprocs=2,
        join=True,
    )


def _test_fsdp_mixed_precision_produces_fp32_local_gradient_shards(
    rank: int, world_size: int
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        torch.manual_seed(42)
        model = FSDP(Linear(4, 4).to(device), compute_dtype=torch.float16)
        inputs = (
            torch.arange(8, device=device, dtype=torch.float16)
            .view(2, 4)
            .requires_grad_(True)
        )

        model(inputs).float().sum().backward()
        model.finish_gradient_synchronization()

        parameter = next(model.parameters())
        assert parameter.grad is not None
        assert parameter.grad.dtype == parameter.dtype == torch.float32
        assert parameter.grad.shape == parameter.shape
    finally:
        _cleanup_process_group()


def test_fsdp_synchronizes_replicated_parameter_gradients():
    mp.spawn(
        _test_fsdp_synchronizes_replicated_parameter_gradients,
        args=(2,),
        nprocs=2,
        join=True,
    )


def _test_fsdp_synchronizes_replicated_parameter_gradients(rank: int, world_size: int):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        model = FSDP(RMSNorm(4).to(device))
        inputs = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0]] if rank == 0 else [[4.0, 1.0, 2.0, 3.0]],
            device=device,
        )

        model(inputs).sum().backward()
        model.finish_gradient_synchronization()

        parameter = next(model.parameters())
        assert parameter.grad is not None
        gathered = [torch.empty_like(parameter.grad) for _ in range(world_size)]
        dist.all_gather(gathered, parameter.grad)
        assert torch.allclose(gathered[0], gathered[1])
    finally:
        _cleanup_process_group()


def test_fsdp_prefetches_the_next_weight_during_current_layer_compute():
    mp.spawn(
        _test_fsdp_prefetches_the_next_weight_during_current_layer_compute,
        args=(2,),
        nprocs=2,
        join=True,
    )


def _test_fsdp_prefetches_the_next_weight_during_current_layer_compute(
    rank: int, world_size: int
):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        torch.manual_seed(42)
        wrapped = _TwoLinearLayers().to(device)
        model = FSDP(wrapped)
        inputs = (
            torch.arange(8, device=device, dtype=torch.float32)
            .view(2, 4)
            .requires_grad_(True)
        )

        # The first iteration records the actual execution order.
        model(inputs).sum().backward()
        model.finish_gradient_synchronization()
        assert model.fwd_params_ordering == [wrapped.second, wrapped.first]
        assert all(
            metadata is None or metadata.work is None
            for metadata in model.sharded_params_metadata.values()
        )

        def assert_next_weight_is_in_flight():
            metadata = model.sharded_params_metadata[wrapped.first.weight]
            assert metadata is not None
            assert metadata.work is not None
            assert metadata.gathered_weight is not None

        wrapped.between_layers = assert_next_weight_is_in_flight
        model(inputs)
        assert model.fwd_params_ordering == [wrapped.second, wrapped.first]
    finally:
        _cleanup_process_group()


def test_fsdp_gathers_fp32_master_parameters():
    mp.spawn(
        _test_fsdp_gathers_fp32_master_parameters,
        args=(2,),
        nprocs=2,
        join=True,
    )


def _test_fsdp_gathers_fp32_master_parameters(rank: int, world_size: int):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    try:
        torch.manual_seed(42)
        wrapped = _TwoLinearLayers().to(device)
        expected = {
            name: parameter.detach().clone()
            for name, parameter in wrapped.named_parameters()
        }
        model = FSDP(wrapped, compute_dtype=torch.float16)

        gathered = model.gather_full_params()

        assert gathered.keys() == expected.keys()
        for name, parameter in gathered.items():
            assert parameter.dtype == torch.float32
            assert torch.equal(parameter, expected[name])
    finally:
        _cleanup_process_group()
