from contextlib import nullcontext
import random
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import torch
from cs336_basics.data import get_random_batch
from cs336_basics.model import BasicsTransformerLM, RotaryEmbedding, TransformerBlock
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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
    def __init__(self, config: LMConfig):
        self.config = config

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)

    def prepare_run(self) -> None:
        self.input, _ = sample_one(self.config)

    def run(self) -> None:
        self.lm.forward(self.input)


class BackwardOp(BenchOp):
    def __init__(self, config: LMConfig):
        self.config = config

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)
        self.optimizer = AdamW(self.lm.parameters(), lr=1e-3)

    def prepare_run(self) -> None:
        self.optimizer.zero_grad()

        inputs, targets = sample_one(self.config)
        logits = self.lm.forward(inputs)
        self.loss = cross_entropy(logits, targets)

    def run(self) -> None:
        self.loss.backward()


class ForwardBackwardOp(BenchOp):
    def __init__(self, config: LMConfig, with_optimizer: bool = True):
        self.config = config
        self.with_optimizer = with_optimizer

    def setup(self) -> None:
        self.lm = get_transformer_lm(self.config)
        self.optimizer = AdamW(self.lm.parameters(), lr=1e-3)

    def prepare_run(self) -> None:
        self.optimizer.zero_grad()
        self.inputs, self.targets = sample_one(self.config)

    def run(self) -> None:
        logits = self.lm.forward(self.inputs)
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
