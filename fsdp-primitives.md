# FSDP Primitives — API Reference

This document describes the PyTorch APIs needed for the FSDP assignment,
their contracts, and the lifecycle of a sharded `Linear` or `Embedding`
weight.

The central invariant is:

> Between layer computations, each rank stores only its FP32 master-weight
> shard. Immediately before a layer runs, its full weight is temporarily
> reconstructed with an all-gather.

---

## 1. Public wrapper interface

```python
class FSDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None: ...

    def forward(self, *inputs, **kwargs): ...

    def finish_gradient_synchronization(self) -> None: ...
```

| Method                            | Contract                                                                                                                                               |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `__init__`                        | Synchronize initial state, select shardable layers, shard their weights, and register communication hooks. Must run before constructing the optimizer. |
| `forward`                         | Delegate to the wrapped module. Registered layer hooks load and free full weights around each layer call.                                              |
| `finish_gradient_synchronization` | Wait for asynchronous gradient collectives and leave every parameter with an FP32 gradient matching its current local parameter shape.                 |

Example:

```python
model = TransformerLM(...).cuda()
model = FSDP(model, compute_dtype=torch.float16)
optimizer = AdamW(model.parameters(), lr=1e-3)

logits = model(inputs)
loss = cross_entropy(logits, targets)
loss.backward()

model.finish_gradient_synchronization()
optimizer.step()
```

The optimizer must be created **after** `FSDP(module)`. At that point,
sharded parameters contain their local FP32 shards, so AdamW creates state
only for those shards.

---

## 2. Should `__init__` shard the module?

Yes. The recommended construction order is:

1. Read `rank` and `world_size`.
2. Broadcast initial parameters and buffers from rank 0.
3. Iterate through all submodules.
4. Shard `Linear.weight` and `Embedding.weight`.
5. Keep small parameters such as normalization weights replicated.
6. Register forward, backward, and gradient hooks.
7. Return the wrapper.
8. Construct the optimizer from `fsdp_model.parameters()`.

Suggested skeleton:

```python
class FSDP(torch.nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        self.work_handles = []
        self.shard_metadata = {}

        self._synchronize_initial_state()
        self._shard_module()
        self._register_hooks()
```

Do not wait until the first optimizer step to shard. Otherwise the optimizer
will allocate state for complete parameters on every rank.

---

## 3. Finding shardable layers

```python
for name, submodule in module.named_modules():
    if isinstance(submodule, (Linear, Embedding)):
        weight = submodule.weight
```

| API                         | Return                                                  |
| --------------------------- | ------------------------------------------------------- |
| `module.modules()`          | Recursive iterator over the module and all submodules.  |
| `module.named_modules()`    | Recursive iterator of `(qualified_name, submodule)`.    |
| `module.parameters()`       | Recursive iterator over parameters without their names. |
| `module.named_parameters()` | Recursive iterator of `(qualified_name, parameter)`.    |

For this assignment:

- Shard `Linear.weight`.
- Shard `Embedding.weight`.
- Keep small normalization parameters replicated.
- Keep buffers replicated unless you explicitly design buffer sharding.

---

## 4. Metadata required for each sharded weight

The local shard no longer contains enough information to reconstruct the
original parameter shape. Store metadata separately.

```python
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass
class ShardMetadata:
    parameter: nn.Parameter
    original_shape: torch.Size
    original_numel: int
    padded_numel: int
    shard_numel: int
    local_master_shard: torch.Tensor
    full_buffer: torch.Tensor | None = None
    gather_work: dist.Work | None = None
```

Important distinction:

- `parameter` remains the registered `nn.Parameter` object known to the
  module and optimizer.
- `local_master_shard` is its persistent FP32 storage.
- `full_buffer` is temporary storage used only during forward or backward.

Preserving the same `nn.Parameter` object avoids breaking module
registration, hooks, and optimizer references.

---

## 5. Sharding one parameter

### Suggested signature

```python
def _shard_parameter(
    self,
    parameter: torch.nn.Parameter,
) -> ShardMetadata: ...
```

### Required operations

```python
flat = parameter.detach().flatten()
original_numel = flat.numel()

shard_numel = math.ceil(original_numel / world_size)
padded_numel = shard_numel * world_size
padding = padded_numel - original_numel

if padding:
    flat = torch.nn.functional.pad(flat, (0, padding))

start = rank * shard_numel
local_shard = flat.narrow(0, start, shard_numel).clone()
```

Then install the local shard as the parameter's resting storage:

```python
with torch.no_grad():
    parameter.data = local_shard
```

For a full weight with shape `(8, d)` and two ranks:

```text
Original parameter:             (8, d)
Flattened parameter:            (8 * d,)
Rank 0 persistent shard:        first 4 * d elements
Rank 1 persistent shard:        final 4 * d elements
```

The local shard is commonly kept flat. Before layer computation, all-gather
reconstructs the flat full parameter and `.view(original_shape)` restores its
logical shape.

### Why padding is needed

Collective APIs require equal-size input shards. If:

```text
original_numel % world_size != 0
```

pad the flattened parameter to:

```text
ceil(original_numel / world_size) * world_size
```

After all-gather, remove padding before reshaping.

Do not use unequal `torch.tensor_split` outputs directly with
`all_gather_into_tensor`.

---

## 6. Initial state synchronization

```python
dist.broadcast(tensor, src=0)
```

| Property  | Contract                                                  |
| --------- | --------------------------------------------------------- |
| Input     | A tensor with the same shape and dtype on every rank.     |
| Effect    | Rank 0's value overwrites the tensor on every other rank. |
| Return    | `None`, or `dist.Work` with `async_op=True`.              |
| Call rule | Every rank must call collectives in the same order.       |

Synchronize full parameters **before** slicing them into rank-local shards:

```python
with torch.no_grad():
    for parameter in module.parameters():
        dist.broadcast(parameter, src=0)

    for buffer in module.buffers():
        dist.broadcast(buffer, src=0)
```

This guarantees that concatenating all local shards reconstructs one
consistent model.

---

## 7. All-gather: reconstructing a full weight

### API

```python
work = dist.all_gather_into_tensor(
    output_tensor,
    input_tensor,
    async_op=True,
)
```

| Argument         | Contract                                                               |
| ---------------- | ---------------------------------------------------------------------- |
| `input_tensor`   | This rank's equal-sized shard.                                         |
| `output_tensor`  | Preallocated tensor with `world_size * input_tensor.numel()` elements. |
| `async_op=False` | Blocks and returns `None`.                                             |
| `async_op=True`  | Returns a `dist.Work`; output is unsafe until `work.wait()` completes. |

For mixed-precision FSDP, cast the local master shard before communicating:

```python
communication_dtype = compute_dtype or local_master_shard.dtype
communication_shard = local_master_shard.to(communication_dtype)

full_buffer = torch.empty(
    padded_numel,
    dtype=communication_dtype,
    device=communication_shard.device,
)

work = dist.all_gather_into_tensor(
    full_buffer,
    communication_shard,
    async_op=True,
)
```

After waiting:

```python
work.wait()
full_weight = full_buffer[:original_numel].view(original_shape)
```

All-gather does not add a world-size dimension. It concatenates shards and
reconstructs the original tensor.

---

## 8. Loading a weight before forward

### Hook API

```python
handle = submodule.register_forward_pre_hook(hook)
```

Default hook signature:

```python
def hook(
    module: torch.nn.Module,
    args: tuple[object, ...],
) -> None: ...
```

Role:

1. Start or retrieve the layer's all-gather.
2. Wait until the full buffer is ready.
3. Remove padding and restore the original shape.
4. Temporarily install the full weight.

Suggested internal interface:

```python
def _start_weight_all_gather(self, module: torch.nn.Module) -> None: ...

def _install_gathered_weight(self, module: torch.nn.Module) -> None: ...
```

Example lifecycle:

```python
def forward_pre_hook(module, args):
    self._start_weight_all_gather(module)
    self._install_gathered_weight(module)
```

At the end of this hook:

```text
module.weight.data.shape == original_shape
module.weight.data.dtype == compute_dtype or FP32
```

The FP32 `local_master_shard` must remain referenced in metadata so it can be
restored later.

---

## 9. Freeing a weight after forward

### Hook API

```python
handle = submodule.register_forward_hook(hook)
```

Default hook signature:

```python
def hook(
    module: torch.nn.Module,
    args: tuple[object, ...],
    output: object,
) -> None: ...
```

Role:

1. Restore `module.weight.data` to the FP32 local master shard.
2. Remove references to the gathered full weight.
3. Clear completed communication state.

Suggested helper:

```python
def _free_full_weight(self, module: torch.nn.Module) -> None:
    metadata = self.shard_metadata[module.weight]

    with torch.no_grad():
        module.weight.data = metadata.local_master_shard

    metadata.full_buffer = None
    metadata.gather_work = None
```

The temporary buffer must remain alive until:

- The all-gather `Work` has completed.
- The layer has finished using the full weight.

“Free” means releasing Python and tensor references. CUDA's caching allocator
may keep the underlying storage reserved for reuse.

---

## 10. Loading a weight before backward

### Hook API

```python
handle = submodule.register_full_backward_pre_hook(hook)
```

Hook signature:

```python
def hook(
    module: torch.nn.Module,
    grad_output: tuple[torch.Tensor | None, ...],
) -> None: ...
```

`Linear` backward needs the full weight to compute the gradient with respect
to its input. Since the forward copy was freed, the backward pre-hook performs
a second all-gather:

```python
def backward_pre_hook(module, grad_output):
    self._start_weight_all_gather(module)
    self._install_gathered_weight(module)
```

This is the source of the two parameter all-gathers per training step:

1. One before forward.
2. One before backward.

---

## 11. Gradient-ready hook

### API

```python
handle = parameter.register_post_accumulate_grad_hook(hook)
```

Hook signature:

```python
def hook(parameter: torch.Tensor) -> None: ...
```

| Property         | Contract                                                      |
| ---------------- | ------------------------------------------------------------- |
| Argument         | The parameter itself, not its gradient.                       |
| Gradient access  | Read `parameter.grad`.                                        |
| Timing           | Runs after autograd has accumulated the parameter's gradient. |
| Return           | Must return `None`.                                           |
| Valid parameters | Leaf tensors with `requires_grad=True`.                       |

For a sharded layer, the hook observes the gradient of the temporary full
weight. It should:

1. Save the full gradient.
2. Clear `parameter.grad`.
3. Restore `parameter.data` to the local master shard.
4. Launch reduce-scatter into an FP32 local gradient buffer.
5. Store the returned `Work` handle.

---

## 12. Reduce-scatter: synchronize and shard gradients

### API

```python
work = dist.reduce_scatter_tensor(
    output_tensor,
    input_tensor,
    op=dist.ReduceOp.SUM,
    async_op=True,
)
```

| Argument        | Contract                                                                      |
| --------------- | ----------------------------------------------------------------------------- |
| `input_tensor`  | A full flattened gradient with `world_size * output_tensor.numel()` elements. |
| `output_tensor` | This rank's reduced gradient shard.                                           |
| Reduction       | Elementwise sum across ranks.                                                 |
| Scatter         | Rank `r` receives chunk `r` of the reduced tensor.                            |

Suppose two ranks compute full gradients `G0` and `G1`:

```text
Reduction:       G = G0 + G1
Rank 0 output:   G[0:shard_numel]
Rank 1 output:   G[shard_numel:2*shard_numel]
```

FSDP requires averaged gradients, so divide exactly once:

```python
full_grad = full_grad.float()
full_grad /= world_size

local_grad = torch.empty_like(local_master_shard)
work = dist.reduce_scatter_tensor(
    local_grad,
    full_grad,
    op=dist.ReduceOp.SUM,
    async_op=True,
)
```

If the full gradient was created from an unpadded parameter, pad it to
`padded_numel` before reduce-scatter.

The local gradient output must have:

```python
local_grad.shape == parameter.data.shape
local_grad.dtype == parameter.data.dtype == torch.float32
```

---

## 13. Asynchronous `dist.Work`

Collectives called with `async_op=True` return a `dist.Work`.

```python
work.wait()
done = work.is_completed()
```

Until `wait()` completes:

- Do not read communication output.
- Do not modify input or output buffers.
- Do not release the last reference to those buffers.

Store both the `Work` and any associated output tensor:

```python
self.work_handles.append((parameter, local_grad, work))
```

Then finalize after `loss.backward()`:

```python
def finish_gradient_synchronization(self) -> None:
    for parameter, local_grad, work in self.work_handles:
        work.wait()
        parameter.grad = local_grad

    self.work_handles.clear()
```

Only call `optimizer.step()` after this method returns.

---

## 14. Replicated parameters

Small parameters such as normalization weights remain complete on every rank.
Their gradients need all-reduce rather than reduce-scatter:

```python
parameter.grad /= dist.get_world_size()
work = dist.all_reduce(
    parameter.grad,
    op=dist.ReduceOp.SUM,
    async_op=True,
)
```

After synchronization:

```text
Every rank has the same full parameter.
Every rank has the same averaged full gradient.
Every rank performs the same optimizer update.
```

Do not reduce-scatter a parameter that remains replicated; its optimizer
expects a full gradient matching the full parameter.

---

## 15. Prefetching the next weight

A synchronous forward pre-hook is sufficient for correctness but blocks
compute while waiting for communication.

To overlap communication:

1. Build an ordered list of sharded layers.
2. Start layer `i`'s all-gather from an earlier layer's forward hook.
3. Store the full buffer and `Work` in layer `i`'s metadata.
4. In layer `i`'s pre-hook, wait only if communication is not finished.

Suggested interface:

```python
def _prefetch_forward_weight(self, layer_index: int) -> None: ...

def _wait_and_install_forward_weight(self, layer_index: int) -> None: ...
```

The assignment's scheduling constraint says not to begin gathering a layer
until the layer two positions before it has completed. This limits the number
of simultaneously materialized full-weight buffers.

Prefetching changes when all-gather runs, not its result or communication
volume.

---

## 16. Gathering complete parameters for validation

The test adapter needs to reconstruct a normal state dictionary.

Suggested signature:

```python
def gather_full_params(
    fsdp_model: FSDP,
) -> dict[str, torch.Tensor]: ...
```

For each parameter:

- Sharded: all-gather FP32 master shards, trim padding, and reshape.
- Replicated: clone the existing full parameter.

Example:

```python
full_parameters = fsdp_gather_full_params(fsdp_model)
full_weight = full_parameters["linear1.weight"]
```

This helper is for inspection and testing. It should not permanently replace
the model's local shards.

---

## 17. Complete layer state machine

```text
FP32 local master shard
    │
    ├── cast to compute_dtype
    ├── all-gather
    ▼
full compute weight
    │
    ├── local forward
    ├── restore and free
    ▼
FP32 local master shard
    │
    ├── cast and all-gather again
    ├── local backward
    ▼
full gradient from this rank's local batch
    │
    ├── cast/pad to FP32
    ├── divide by world_size
    ├── reduce-scatter
    ▼
FP32 local gradient shard
    │
    ├── optimizer.step()
    ▼
updated FP32 local master shard
```

Communication per sharded parameter per training step:

```text
2 × all-gather(parameter shard)
1 × reduce-scatter(full gradient)
```

---

## 18. Common correctness failures

1. **Constructing the optimizer before sharding**
   Allocates optimizer state for complete parameters.

2. **Replacing the `nn.Parameter` object repeatedly**
   Breaks optimizer references and registered hooks. Preserve the parameter
   object and switch only its temporary storage.

3. **Unequal shard sizes**
   Pad before collectives and trim after all-gather.

4. **Freeing an asynchronous output too early**
   Keep buffers alive until `Work.wait()` returns.

5. **Calling collectives in different orders across ranks**
   Causes hangs or mismatched communication.

6. **Forgetting gradient averaging**
   `ReduceOp.SUM` produces a sum, not the DDP mean.

7. **Leaving a full gradient attached to a local shard**
   `parameter.grad.shape` must match `parameter.data.shape` before the
   optimizer step.

8. **Applying reduce-scatter to replicated norms**
   Replicated parameters require all-reduce.

9. **Losing the FP32 master shard while installing a compute weight**
   Keep an explicit metadata reference to persistent FP32 storage.

10. **Gathering every layer simultaneously**
    Correct but defeats FSDP's memory savings. Materialize only the weights
    needed by the current prefetch window.
