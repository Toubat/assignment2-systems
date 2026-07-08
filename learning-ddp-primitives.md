# Understanding: DDP primitives (naive_ddp / overlapped DDP)

## 1. The problem
- [ ] What DDP must guarantee (all ranks step with identical averaged grads)
- [ ] Why "all-reduce after backward finishes" wastes time (no overlap)
- [ ] Why we need a per-parameter trigger *during* backward

## 2. The solution — the four primitives
- [ ] P1: autograd hooks — `Tensor.register_post_accumulate_grad_hook`
      (when exactly it fires, vs. `register_hook`)
- [ ] P2: `dist.broadcast` at construction (why rank 0 → all)
- [ ] P3: `dist.all_reduce(..., async_op=True)` → `Work` handle
- [ ] P4: the wrapper is itself an `nn.Module` (forward delegation,
      `parameters()` passthrough, a `finish_gradient_synchronization()` method)

## 3. Edge cases the test checks
- [ ] Tied weights: same `Parameter` object registered twice — hook fires once?
- [ ] `requires_grad=False` params: broadcast yes, reduce never
- [ ] Averaging: all-reduce gives SUM, you need mean

## 4. Broader context
- [ ] Why this is exactly what `torch.nn.parallel.DistributedDataParallel` does
      (bucketing aside)
- [ ] Why `ddp_on_after_backward` must exist at all (what breaks without it)
