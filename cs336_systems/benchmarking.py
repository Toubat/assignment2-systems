import random
import statistics
import timeit
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from cs336_basics.data import get_random_batch
from cs336_basics.model import BasicsTransformerLM, RotaryEmbedding, TransformerBlock
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from torch import nn
from torch.optim import Optimizer

from cs336_systems.gradient_checkpoint import linear_checkpoint, recursive_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LMConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


class BenchOp(ABC):
    @abstractmethod
    def setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError


class ForwardOp(BenchOp):
    def __init__(self, config: LMConfig, compile_model: bool = False):
        self.config = config
        self.compile_model = compile_model

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)
        if self.compile_model:
            self.lm = torch.compile(self.lm)

    def prepare_run(self) -> None:
        self.input, _ = sample_one(self.config)

    def run(self) -> None:
        self.lm(self.input)


class BackwardOp(BenchOp):
    def __init__(self, config: LMConfig, compile_model: bool = False):
        self.config = config
        self.compile_model = compile_model

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)
        if self.compile_model:
            self.lm = torch.compile(self.lm)
        self.optimizer = AdamW(self.lm.parameters(), lr=1e-3)  # type: ignore

    def prepare_run(self) -> None:
        self.optimizer.zero_grad()

        inputs, targets = sample_one(self.config)
        logits = self.lm(inputs)
        self.loss = cross_entropy(logits, targets)

    def run(self) -> None:
        self.loss.backward()


class ForwardBackwardOp(BenchOp):
    def __init__(
        self, config: LMConfig, with_optimizer: bool = True, compile_model: bool = False
    ):
        self.config = config
        self.with_optimizer = with_optimizer
        self.compile_model = compile_model

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)
        if self.compile_model:
            self.lm = torch.compile(self.lm)
        self.optimizer = AdamW(self.lm.parameters(), lr=1e-3)  # type: ignore

    def prepare_run(self) -> None:
        self.optimizer.zero_grad()
        self.inputs, self.targets = sample_one(self.config)

    def run(self) -> None:
        logits = self.lm(self.inputs)
        loss = cross_entropy(logits, self.targets)
        loss.backward()

        if self.with_optimizer:
            self.optimizer.step()


class _CheckpointBenchOp(BenchOp):
    """Shared setup for gradient-checkpointing benchmarks.

    Builds a single (compiled) ``TransformerBlock`` on the GPU and applies it
    ``num_layers`` times via a checkpointing strategy supplied by subclasses.
    """

    def __init__(self, config: LMConfig):
        self.config = config

    def setup(self) -> None:
        block = TransformerBlock(
            self.config.d_model,
            self.config.num_heads,
            self.config.d_ff,
            RotaryEmbedding(
                self.config.context_length,
                self.config.d_model // self.config.num_heads,
            ),
        ).to(DEVICE)
        self.block = torch.compile(block, fullgraph=True)

    def prepare_run(self) -> None:
        self.x = torch.randn(
            (4, self.config.context_length, self.config.d_model),
            requires_grad=True,
            device=DEVICE,
        )

    def apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def run(self) -> None:
        x = self.apply_blocks(self.x)
        x.sum().backward()


class RecursiveCheckpointOp(_CheckpointBenchOp):
    def apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        return recursive_checkpoint(self.block, x, num_layers=self.config.num_layers)


class LinearCheckpointOp(_CheckpointBenchOp):
    def __init__(self, config: LMConfig, group_size: int):
        super().__init__(config)
        self.group_size = group_size

    def apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        return linear_checkpoint(
            self.block,
            x,
            num_layers=self.config.num_layers,
            group_size=self.group_size,
        )


class WeightedSumBenchOp(BenchOp):
    """Benchmarks a weighted-sum implementation ``fn(x, weight) -> out``.

    The op is initialized with an arbitrary kernel function, so it works for
    both the Triton kernel (``WeightedSum.apply``) and a torch reference such as
    ``lambda x, w: (x * w).sum(-1)``. Set ``backward=True`` to time fwd+bwd.
    """

    def __init__(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        rows: int,
        d: int,
        backward: bool = False,
    ):
        self.fn = fn
        self.rows = rows
        self.d = d
        self.backward = backward

    def setup(self) -> None:
        self.x = torch.randn(
            self.rows, self.d, device=DEVICE, requires_grad=self.backward
        )
        self.weight = torch.randn(self.d, device=DEVICE, requires_grad=self.backward)

    def prepare_run(self) -> None:
        if self.backward:
            self.x.grad = None
            self.weight.grad = None

    def run(self) -> None:
        out = self.fn(self.x, self.weight)
        if self.backward:
            out.sum().backward()


def benchmark(
    op: BenchOp,
    num_warmups: int = 10,
    num_trials: int = 20,
    autocast_bfloat16: bool = False,
    memory_snapshot_path: str | None = None,
) -> tuple[float, float]:
    """Benchmark the given function by running it for a number of warmups and trials.

    When ``memory_snapshot_path`` is set, a CUDA memory snapshot of the timed
    trials is dumped to that path (load it at https://pytorch.org/memory_viz).
    Recording starts only after warm-up so allocator warm-up noise is excluded.
    """
    assert torch.cuda.is_available(), "benchmark() requires CUDA"

    op.setup()

    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if autocast_bfloat16
        else nullcontext()
    )

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    for _ in range(num_warmups):
        with ctx:
            op.prepare_run()
            op.run()

    torch.cuda.synchronize()

    # Reset after warm-up so peak-memory readings reflect only the timed trials
    # (excludes torch.compile / allocator warm-up transients).
    torch.cuda.reset_peak_memory_stats()

    if memory_snapshot_path is not None:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    times: list[float] = []
    for _ in range(num_trials):
        op.prepare_run()

        with ctx:
            start_event.record()
            op.run()
            stop_event.record()

        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(stop_event))

    if memory_snapshot_path is not None:
        torch.cuda.memory._dump_snapshot(memory_snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    mean_time = sum(times) / len(times)
    std_time = statistics.stdev(times)

    print(f"Mean time for: {mean_time} ms ± {std_time} ms")
    return mean_time, std_time


def get_transformer_lm(config: LMConfig) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        config.vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
    ).to(DEVICE)


@lru_cache
def batch_input(
    vocab_size: int, context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    print(
        f"Getting batch input for vocab size {vocab_size} and context length {context_length}"
    )
    X, y = get_random_batch(
        dataset_size=10000,
        vocab_size=vocab_size,
        batch_size=100,
        context_length=context_length,
        device=DEVICE,
    )

    return X.to(DEVICE), y.to(DEVICE)


def sample_one(config: LMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = batch_input(config.vocab_size, config.context_length)

    # randomly sample a single input from the batch
    idx = random.randint(0, X.shape[0] - 1)
    return X[idx].unsqueeze(0), y[idx].unsqueeze(0)


def dist_train_step(
    model: nn.Module,
    optimizer: Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Run one distributed training step and return ``(step_ms, comm_ms)``.

    ``comm_ms`` is the wall time spent in ``finish_gradient_synchronization()``,
    i.e. the gradient-communication cost that is NOT overlapped with the
    backward pass:

    - naive/flat DDP: all all-reduces happen there, so it is the full comm time.
    - overlapped DDP: only the residual wait on in-flight all-reduces.

    Before timing the comm window we synchronize the *compute* stream only
    (``torch.cuda.current_stream()``), so async NCCL work launched during
    backward keeps running and is correctly attributed to the comm window.
    Works with any wrapper exposing ``finish_gradient_synchronization``
    (DDP today, FSDP later); plain modules report 0 comm time.
    """
    timer = timeit.default_timer

    step_start = timer()

    optimizer.zero_grad()
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    loss.backward()

    # Wait for backward *compute* only; async NCCL streams keep running.
    torch.cuda.current_stream().synchronize()
    comm_start = timer()

    finish = getattr(model, "finish_gradient_synchronization", None)
    if finish is not None:
        finish()
    # Wait for all streams (incl. NCCL) so grads are final.
    torch.cuda.synchronize()
    comm_stop = timer()

    optimizer.step()
    torch.cuda.synchronize()
    step_stop = timer()

    return (step_stop - step_start) * 1e3, (comm_stop - comm_start) * 1e3


def benchmark_parallel_training(
    config: LMConfig,
    wrap_model: Callable[[nn.Module], nn.Module],
    make_optimizer: Callable[[Iterable[nn.Parameter]], Optimizer] | None = None,
    global_batch_size: int = 4,
    num_warmups: int = 5,
    num_trials: int = 20,
) -> dict[str, float]:
    """Benchmark one rank of data-parallel training. Call from every rank.

    Requires an initialized process group (the caller spawns workers and calls
    ``dist.init_process_group``). Builds the LM, wraps it with ``wrap_model``
    (DDP variant / FSDP / identity), builds the optimizer via ``make_optimizer``
    (pluggable so a sharded optimizer can be benchmarked later), and times
    ``num_trials`` training steps on a ``global_batch_size / world_size`` local
    batch. Returns stats (ms) averaged across ranks.
    """
    assert dist.is_initialized(), "requires an initialized process group"
    world_size = dist.get_world_size()
    assert (
        global_batch_size % world_size == 0
    ), f"world_size={world_size} must divide global_batch_size={global_batch_size}"
    local_batch_size = global_batch_size // world_size

    model = wrap_model(get_transformer_lm(config))
    if make_optimizer is None:
        make_optimizer = lambda params: AdamW(params, lr=1e-3)  # noqa: E731
    optimizer = make_optimizer(model.parameters())

    def sample_local_batch() -> tuple[torch.Tensor, torch.Tensor]:
        return get_random_batch(
            dataset_size=10000,
            vocab_size=config.vocab_size,
            batch_size=local_batch_size,
            context_length=config.context_length,
            device=DEVICE,
        )

    step_times: list[float] = []
    comm_times: list[float] = []

    for i in range(num_warmups + num_trials):
        inputs, targets = sample_local_batch()

        # Align ranks so we measure the step, not stragglers.
        dist.barrier()
        torch.cuda.synchronize()

        step_ms, comm_ms = dist_train_step(model, optimizer, inputs, targets)
        if i >= num_warmups:
            step_times.append(step_ms)
            comm_times.append(comm_ms)

    stats = torch.tensor(
        [
            statistics.mean(step_times),
            statistics.stdev(step_times),
            statistics.mean(comm_times),
            statistics.stdev(comm_times),
        ],
        device=DEVICE,
    )
    # Average the per-rank stats so the result reflects all ranks.
    dist.all_reduce(stats)
    stats /= world_size

    step_mean, step_std, comm_mean, comm_std = stats.tolist()
    return {
        "step_ms_mean": step_mean,
        "step_ms_std": step_std,
        "comm_ms_mean": comm_mean,
        "comm_ms_std": comm_std,
        "comm_fraction": comm_mean / step_mean if step_mean else 0.0,
    }
