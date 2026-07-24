from dataclasses import dataclass
from itertools import chain
from typing import cast
from cs336_basics.model import Embedding, Linear
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


class BaseDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_parameters()

    def _broadcast_parameters(self):
        # Broadcast the parameters and buffers from rank 0 to all other ranks.
        with torch.no_grad():
            for weight in chain(
                self.module.parameters(),
                self.module.buffers(),
            ):
                dist.broadcast(weight, src=0, async_op=False)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        pass


class NaiveDDP(BaseDDP):
    def __init__(self, module: nn.Module, batch_all_reduce: bool = False):
        super().__init__(module)
        self.batch_all_reduce = batch_all_reduce

    def _synchronize_naive(self):
        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            assert param.grad is not None
            param.grad /= dist.get_world_size()
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=False)

    def _synchronize_batch(self):
        params: list[nn.Parameter] = []

        for param in self.module.parameters():
            if not param.requires_grad:
                continue

            params.append(param)

        grads = [p.grad for p in params]
        flattened_grads = _flatten_dense_tensors(grads)
        flattened_grads /= dist.get_world_size()
        dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=False)

        for i, grad in enumerate(_unflatten_dense_tensors(flattened_grads, grads)):
            params[i].grad = grad

    def finish_gradient_synchronization(self):
        if self.batch_all_reduce:
            self._synchronize_batch()
        else:
            self._synchronize_naive()


class DDP(BaseDDP):

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self.work_handles: list[dist.Work] = []
        self._init_ddp()

    def _init_ddp(self):
        def _all_reduce_grad_async(param: torch.Tensor):
            assert (
                param.requires_grad and param.grad is not None
            ), "Parameter must require grad and have a non-None grad to be all-reduced in DDP."

            param.grad /= dist.get_world_size()
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.work_handles.append(handle)  # type: ignore

        # Attach a hook to all the parameters to all-reduce the gradient asynchronously.
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            param.register_post_accumulate_grad_hook(_all_reduce_grad_async)

        def _broadcast_buffer(module: nn.Module, _):
            with torch.no_grad():
                for buf in module.buffers():
                    dist.broadcast(buf, src=0, async_op=False)

        self.module.register_forward_pre_hook(_broadcast_buffer)

    def finish_gradient_synchronization(self):
        for handle in self.work_handles:
            handle.wait()
        self.work_handles.clear()


@dataclass
class ShardMetadata:
    original_shape: tuple[int, ...]
    original_numel: int
    local_master_shard: torch.Tensor

    work: dist.Work | None = None
    gathered_weight: torch.Tensor | None = None


class FSDP(torch.nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()

        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.sharded_params_metadata: dict[torch.Tensor, ShardMetadata | None] = {}
        self.fwd_params_ordering: list[nn.Module] = []
        self.bwd_params_ordering: list[nn.Module] = []
        self.record_forward_order = True
        self.record_backward_order = True
        self.forward_index = 0
        self.backward_index = 0
        self.reduce_scatter_work_handles: list[
            tuple[nn.Parameter, torch.Tensor, dist.Work]
        ] = []
        self.all_reduce_work_handles: list[dist.Work] = []
        self._broadcast_parameters()

    def _schedule_all_gather_weight(self, metadata: ShardMetadata):
        # The same layer can be reached by multiple scheduling paths. Do not
        # overwrite an in-flight handle and lose its communication buffers.
        if metadata.work is not None:
            return

        weight_shard = metadata.local_master_shard
        dtype = self.compute_dtype or weight_shard.dtype
        communication_shard = weight_shard.to(dtype)

        metadata.gathered_weight = torch.empty(
            communication_shard.numel() * self.world_size,
            dtype=dtype,
            device=communication_shard.device,
            requires_grad=False,
        )

        metadata.work = dist.all_gather_into_tensor(
            metadata.gathered_weight,
            communication_shard,
            async_op=True,
        )

    def _wait_all_gather_weight(self, metadata: ShardMetadata) -> torch.Tensor:
        # During order-recording iterations there is no known successor to
        # prefetch, so schedule the current weight on demand.
        if metadata.work is None:
            self._schedule_all_gather_weight(metadata)

        assert metadata.work is not None, "Work not scheduled for the given metadata."
        metadata.work.wait()  # type: ignore
        assert (
            metadata.gathered_weight is not None
        ), "Gathered weight not allocated for the given metadata."

        full_weight = metadata.gathered_weight[: metadata.original_numel].view(
            metadata.original_shape
        )

        metadata.work = None
        metadata.gathered_weight = None
        return full_weight

    def _get_metadata(self, obj: nn.Module | nn.Parameter) -> ShardMetadata:
        if isinstance(obj, nn.Module):
            assert isinstance(
                obj, (Linear, Embedding)
            ), "Only Linear and Embedding modules can be sharded."
            metadata = self.sharded_params_metadata[obj.weight]
        else:
            metadata = self.sharded_params_metadata[obj]

        assert (
            metadata is not None
        ), f"Metadata not found for the given {'module' if isinstance(obj, nn.Module) else 'parameter'}."
        return metadata

    def _broadcast_parameters(self):
        def _forward_pre_hook(module: nn.Module, _):
            # Observe runtime execution order rather than assuming module
            # registration order matches an arbitrary forward graph.
            if self.record_forward_order:
                self.fwd_params_ordering.append(module)
            else:
                assert self.forward_index < len(self.fwd_params_ordering)
                assert (
                    self.fwd_params_ordering[self.forward_index] is module
                ), "Sharded module execution order changed between iterations."

            metadata = self._get_metadata(module)
            full_weight = self._wait_all_gather_weight(metadata)

            # Launch the successor only after the current weight is ready, so
            # its communication overlaps with this module's forward compute.
            self.forward_index += 1
            if not self.record_forward_order and self.forward_index < len(
                self.fwd_params_ordering
            ):
                next_module = self.fwd_params_ordering[self.forward_index]
                self._schedule_all_gather_weight(self._get_metadata(next_module))

            with torch.no_grad():
                module.weight.data = full_weight

        def _backward_pre_hook(module: nn.Module, _):
            # Record backward independently: branches, shared modules, and
            # recomputation need not follow exact reverse-forward order.
            if self.record_backward_order:
                self.bwd_params_ordering.append(module)
            else:
                assert self.backward_index < len(self.bwd_params_ordering)
                assert (
                    self.bwd_params_ordering[self.backward_index] is module
                ), "Sharded module backward order changed between iterations."

            metadata = self._get_metadata(module)
            full_weight = self._wait_all_gather_weight(metadata)

            # Prefetch the next observed backward module while this module
            # computes its input and parameter gradients.
            self.backward_index += 1
            if not self.record_backward_order and self.backward_index < len(
                self.bwd_params_ordering
            ):
                next_module = self.bwd_params_ordering[self.backward_index]
                self._schedule_all_gather_weight(self._get_metadata(next_module))

            with torch.no_grad():
                module.weight.data = full_weight

        def _forward_post_hook(module: nn.Module, _, __):
            metadata = self._get_metadata(module)
            with torch.no_grad():
                # Restore this exact tensor so Parameter and metadata continue
                # to alias the FP32 shard that AdamW updates in place.
                module.weight.data = metadata.local_master_shard

        def _reduce_scatter_grad_hook(param: nn.Parameter):
            metadata = self._get_metadata(param)

            assert param.grad is not None
            flattened_grad = param.grad.detach().flatten()
            padded_grad = pad_to_shard_size(flattened_grad, self.world_size)
            padded_grad /= self.world_size

            dtype = self.compute_dtype or metadata.local_master_shard.dtype
            padded_grad = padded_grad.to(dtype)

            local_grad = torch.empty_like(metadata.local_master_shard, dtype=dtype)
            # Every rank contributes a full local-batch gradient but receives
            # only the averaged chunk matching its master-weight shard.
            work = dist.reduce_scatter_tensor(
                output=local_grad,
                input=padded_grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            self.reduce_scatter_work_handles.append((param, local_grad, work))  # type: ignore

            with torch.no_grad():
                # AdamW must see the local FP32 parameter. Its matching local
                # gradient is attached after the collective completes.
                param.grad = None
                param.data = metadata.local_master_shard

        def _all_reduce_grad_hook(param: nn.Parameter):
            assert (
                param.requires_grad and param.grad is not None
            ), "Parameter must require grad and have a non-None grad to be all-reduced in DDP."

            param.grad /= dist.get_world_size()
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.all_reduce_work_handles.append(handle)  # type: ignore

        for submodule in self.module.modules():
            if isinstance(submodule, (Linear, Embedding)):
                self.sharded_params_metadata[submodule.weight] = None
                submodule.register_forward_pre_hook(_forward_pre_hook)
                submodule.register_forward_hook(_forward_post_hook)
                submodule.register_full_backward_pre_hook(_backward_pre_hook)
                submodule.weight.register_post_accumulate_grad_hook(
                    _reduce_scatter_grad_hook
                )

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0, async_op=False)
                if param in self.sharded_params_metadata:
                    self._shard_param(param)
                elif param.requires_grad:
                    # Small parameters such as norms remain replicated, so
                    # synchronize their full gradients with all-reduce.
                    param.register_post_accumulate_grad_hook(_all_reduce_grad_hook)

            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0, async_op=False)

    def _shard_param(self, param: torch.Tensor):
        flattened = param.detach().flatten()

        padded = pad_to_shard_size(flattened, self.world_size)
        local_shard = padded.chunk(self.world_size)[self.rank].clone()

        self.sharded_params_metadata[param] = ShardMetadata(
            original_shape=param.shape,
            original_numel=param.numel(),
            local_master_shard=local_shard,
        )

        with torch.no_grad():
            param.data = local_shard

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """Reconstruct FP32 master parameters without changing local shards."""
        full_params: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for name, param in self.module.named_parameters():
                metadata = self.sharded_params_metadata.get(param)
                if metadata is None:
                    full_params[name] = param.detach().clone()
                    continue

                local_shard = metadata.local_master_shard
                gathered = torch.empty(
                    local_shard.numel() * self.world_size,
                    dtype=local_shard.dtype,
                    device=local_shard.device,
                )
                dist.all_gather_into_tensor(gathered, local_shard)
                full_params[name] = (
                    gathered[: metadata.original_numel]
                    .view(metadata.original_shape)
                    .clone()
                )

        return full_params

    def forward(self, *args, **kwargs):
        # Hooks advance these cursors in observed execution order. Reset both
        # once per training iteration before entering the wrapped model.
        self.forward_index = 0
        self.backward_index = 0

        if not self.record_forward_order and self.fwd_params_ordering:
            # Seed the pipeline so the first pre-hook waits only for residual
            # communication instead of launching its gather from scratch.
            first_module = self.fwd_params_ordering[0]
            self._schedule_all_gather_weight(self._get_metadata(first_module))

        output = self.module(*args, **kwargs)
        self.record_forward_order = False

        if not self.record_backward_order and self.bwd_params_ordering:
            # Once backward order is known, gather its first weight while the
            # caller computes the loss between forward and loss.backward().
            first_module = self.bwd_params_ordering[0]
            self._schedule_all_gather_weight(self._get_metadata(first_module))

        return output

    def finish_gradient_synchronization(self):
        with torch.no_grad():
            for param, local_grad, work in self.reduce_scatter_work_handles:
                work.wait()
                param.grad = local_grad.to(param.dtype)

            for handle in self.all_reduce_work_handles:
                handle.wait()

        self.reduce_scatter_work_handles.clear()
        self.all_reduce_work_handles.clear()
        self.record_backward_order = False


def pad_to_shard_size(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Pad a tensor to the next multiple of the world size."""
    shard_numel = (tensor.numel() + world_size - 1) // world_size
    padded_numel = shard_numel * world_size
    padding = padded_numel - tensor.numel()
    return F.pad(tensor, (0, padding), value=0.0)
