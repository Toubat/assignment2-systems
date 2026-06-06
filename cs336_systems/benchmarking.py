from contextlib import nullcontext
import random
import statistics
import timeit
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import torch
from cs336_basics.data import get_random_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

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


class Timer(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def elapsed(self) -> float:
        raise NotImplementedError

    def __enter__(self) -> None:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> float:
        self.stop()
        return self.elapsed()


class PythonTimer(Timer):
    """timeit.default_timer() module wrapper"""

    def __init__(self):
        self.start_time = None
        self.stop_time = None

    def start(self) -> None:
        self.start_time = timeit.default_timer()

    def stop(self) -> None:
        self.stop_time = timeit.default_timer()

    def elapsed(self) -> float:
        return (self.stop_time - self.start_time) * 1000


class CUDATimer(Timer):
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self.start_event.record()

    def stop(self) -> None:
        self.stop_event.record()
        torch.cuda.synchronize()

    def elapsed(self) -> float:
        return self.start_event.elapsed_time(self.stop_event)


def benchmark(op: BenchOp, num_warmups: int = 10, num_trials: int = 20, autocast_bfloat16: bool = False) -> tuple[float, float]:
    """Benchmark the given function by running it for a number of warmups and trials."""
    timer = CUDATimer() if DEVICE == "cuda" else PythonTimer()

    op.setup()

    ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast_bfloat16 else nullcontext()

    for _ in range(num_warmups):
        with ctx:
            op.prepare_run()
            op.run()

    with timer:
        pass

    times: list[float] = []
    for _ in range(num_trials):
        op.prepare_run()

        with ctx:
            with timer:
                op.run()

        times.append(timer.elapsed())

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
def batch_input(vocab_size: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"Getting batch input for vocab size {vocab_size} and context length {context_length}")
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
