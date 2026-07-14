import torch

from cs336_systems import benchmarking


class _OptimizerWrapper:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer


class _TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(5, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embedding(inputs)


def test_training_memory_components_counts_nested_optimizer_state():
    assert hasattr(benchmarking, "_training_memory_components")

    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())

    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()

    components = benchmarking._training_memory_components(
        model, _OptimizerWrapper(optimizer)
    )

    expected_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    expected_gradient_bytes = sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    expected_optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )

    assert components == {
        "parameter_bytes": expected_parameter_bytes,
        "gradient_bytes": expected_gradient_bytes,
        "optimizer_state_bytes": expected_optimizer_state_bytes,
    }


def test_parallel_training_reports_memory_checkpoints(monkeypatch):
    monkeypatch.setattr(benchmarking, "DEVICE", "cpu")
    monkeypatch.setattr(benchmarking.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(benchmarking.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(benchmarking.dist, "barrier", lambda: None)
    monkeypatch.setattr(
        benchmarking.dist, "all_reduce", lambda tensor, *args, **kwargs: tensor
    )
    monkeypatch.setattr(benchmarking.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        benchmarking.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(benchmarking.torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(benchmarking.torch.cuda, "max_memory_allocated", lambda: 150)
    monkeypatch.setattr(benchmarking, "get_transformer_lm", lambda config: _TinyLM())
    monkeypatch.setattr(
        benchmarking,
        "get_random_batch",
        lambda **kwargs: (
            torch.zeros((kwargs["batch_size"], 2), dtype=torch.long),
            torch.zeros((kwargs["batch_size"], 2), dtype=torch.long),
        ),
    )
    profiled_steps = 0

    def fake_dist_train_step(*args, memory_samples=None, **kwargs):
        nonlocal profiled_steps
        assert memory_samples is not None
        profiled_steps += 1
        memory_samples["memory_before_optimizer_step_bytes"].append(100)
        memory_samples["peak_forward_backward_bytes"].append(150)
        memory_samples["memory_after_optimizer_step_bytes"].append(120)
        memory_samples["peak_optimizer_step_bytes"].append(160)
        return 1.0, 0.25

    monkeypatch.setattr(benchmarking, "dist_train_step", fake_dist_train_step)

    results = benchmarking.benchmark_parallel_training(
        config=benchmarking.LMConfig(5, 2, 1, 1, 1, 1),
        wrap_model=lambda model: model,
        make_optimizer=lambda params: torch.optim.AdamW(params),
        global_batch_size=1,
        num_warmups=1,
        num_trials=2,
    )

    assert profiled_steps == 3
    for checkpoint in (
        "memory_after_model_bytes",
        "memory_before_optimizer_step_bytes",
        "memory_after_optimizer_step_bytes",
        "peak_model_init_bytes",
        "peak_forward_backward_bytes",
        "peak_optimizer_step_bytes",
        "parameter_bytes",
        "gradient_bytes",
        "optimizer_state_bytes",
    ):
        assert results[f"{checkpoint}_mean"] >= 0
        assert results[f"{checkpoint}_max"] >= results[f"{checkpoint}_mean"]
