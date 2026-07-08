# DDP Primitives — API Reference

The four primitives needed to build a DDP wrapper (`get_ddp` / `ddp_on_after_backward`),
plus the buffer-sync hooks. Contracts only, no implementation.

---

## P1. Per-parameter gradient-ready hook

```python
handle = param.register_post_accumulate_grad_hook(hook)
```

|                |                                                                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Lives on       | `torch.Tensor` (not `nn.Module`)                                                                                                   |
| Hook signature | `hook(param: torch.Tensor) -> None`                                                                                                |
| Fires          | During `loss.backward()`, immediately **after** the autograd engine finishes writing this parameter's `.grad` for the current pass |
| Hook argument  | The **parameter itself**, not the gradient — read the gradient via `param.grad`                                                    |
| Hook return    | Must be `None`; cannot replace the gradient by returning a value (mutate `param.grad` in place instead)                            |
| Valid on       | Leaf tensors with `requires_grad=True` only (i.e. trainable parameters); raises on non-leaf tensors                                |
| Returns        | `torch.utils.hooks.RemovableHandle` — call `.remove()` to detach                                                                   |

**Role in DDP:** the trigger. Backward becomes a stream of per-parameter
"gradient is final" events, letting you launch communication mid-backward
while earlier layers are still computing. Gradients become ready in
output→input order (last layers first).

**Tied weights:** a parameter used in multiple places has a single
`AccumulateGrad` node; the engine's dependency counting runs it once, after
all usage paths are summed into `.grad`. The hook therefore fires **once per
backward**, with the complete gradient.

**Contrast —** `param.register_hook(hook)`**:**
`hook(grad: Tensor) -> Tensor | None`. Fires with the _incoming_ gradient
**before** accumulation (may fire per usage path — partial gradients with
tied weights) and may return a replacement gradient. Wrong primitive for DDP.

---

## P2. One-time weight sync

```python
dist.broadcast(tensor, src=0)
```

|                  |                                                                                |
| ---------------- | ------------------------------------------------------------------------------ |
| Semantics        | Collective: rank `src`'s data overwrites `tensor` in-place on every other rank |
| Call requirement | Every rank in the group must call it (sender and receivers alike)              |
| Returns          | `None` (blocking) or a `Work` handle with `async_op=True`                      |
| Applied to       | **All** parameters _and_ buffers, once, at wrapper construction                |

**Role in DDP:** establishes the invariant that all ranks start from
identical state. Gradient averaging preserves weight _differences_ — identical
updates applied to different starting points never converge, and the averaged
gradient of inconsistent models is not a valid gradient for any of them.

**Iterating targets:**

```python
module.parameters()   /  module.named_parameters()   # Parameters
module.buffers()      /  module.named_buffers()      # registered buffers
```

Buffers (e.g. BatchNorm running stats) are model state without gradients —
broadcast yes, all-reduce never.

---

## P3. Async collective + Work handle

```python
work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
```

|                       |                                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| Semantics             | In-place: `tensor` on every rank is replaced by the elementwise reduction across ranks        |
| `async_op=False`      | Blocks; returns `None`                                                                        |
| `async_op=True`       | Returns immediately with a `Work` handle; communication proceeds in background                |
| `Work.wait()`         | Blocks until the collective completes                                                         |
| `Work.is_completed()` | Non-blocking status check                                                                     |
| Hazard                | Until `wait()` returns, the tensor's contents are in transition — must not be read or written |

**Role in DDP:** the overlap mechanism. Launched inside the P1 hook; handles
are collected and all waited on in `finish_gradient_synchronization()`
**before** `optimizer.step()` — otherwise the optimizer reads partially
reduced gradients and ranks silently drift.

**Averaging:** all-reduce gives SUM; DDP semantics require the mean.
Dividing by `world_size` is your responsibility (before launch or after wait).

---

## P4. The wrapper is a plain `nn.Module`

No base-class overrides. The wrapper subclasses `nn.Module` and:

| Surface                                | Contract                                                                                                                                                        |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `__init__`                             | Stores the wrapped module as a child; performs P2 broadcast; attaches P1 hooks (trainable params only)                                                          |
| `forward(*args, **kwargs)`             | Delegates to the wrapped module                                                                                                                                 |
| `parameters()` / `state_dict()` / etc. | Free — module registration recurses into children automatically                                                                                                 |
| `finish_gradient_synchronization()`    | The one invented method: waits on all stored P3 handles, clears the list. Called by `ddp_on_after_backward`, after `loss.backward()`, before `optimizer.step()` |

**Hook/broadcast scope:** hooks → only `requires_grad=True` params
(others never produce gradients); broadcast → all params and buffers.

---

## Extra: module forward/backward hooks (buffer sync per step)

All live on `nn.Module`, all return `RemovableHandle`, all fire on
`module(...)` (`__call__`) — bypassed if `forward()` is called directly.

| Method                               | Fires                    | Hook signature                                          |
| ------------------------------------ | ------------------------ | ------------------------------------------------------- |
| `register_forward_pre_hook(h)`       | Before `forward`         | `h(module, args)` — return `None` or new args tuple     |
| `register_forward_hook(h)`           | After `forward`          | `h(module, args, output)` — return `None` or new output |
| `register_full_backward_pre_hook(h)` | Before module's backward | `h(module, grad_output)`                                |
| `register_full_backward_hook(h)`     | After module's backward  | `h(module, grad_input, grad_output)`                    |

- Pre-hook may return a tuple to rewrite the module's inputs; `None` leaves them untouched.
- `with_kwargs=True` at registration adds kwargs to the hook signature.
- No per-tensor forward hook exists — forward isn't a per-tensor autograd event; only module granularity.
- Global variant: `torch.nn.modules.module.register_module_forward_pre_hook(h)` fires for every module in the process (instrumentation-grade).

**Why per-forward buffer sync exists:** buffers like BatchNorm running stats
mutate during forward, divergently per rank (different data shards), and
gradient all-reduce never touches them. Real `DistributedDataParallel`
re-broadcasts buffers each forward (`broadcast_buffers=True` default).
Construction-time sync suffices only when no buffer mutates.

---

## Choreography summary (one training step)

1. `forward` → (optional) buffer broadcast, then delegate.
2. `loss.backward()` → P1 hooks fire per parameter, output→input order;
   each launches a P3 async all-reduce and stores the `Work` handle.
3. `finish_gradient_synchronization()` → wait on all handles; grads now
   complete and averaged on every rank.
4. `optimizer.step()` → identical update everywhere; P2's invariant is preserved.
