"""Cross-seed process scheduling tests for the Fermi CLI."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from experiments.run_fermi_v3 import run_seed_batch


class ImmediateExecutor:
    """ProcessPool-compatible test double that records submitted seed jobs."""

    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.submitted = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def submit(self, _fn, _args, seed, _label, _out_root):
        self.submitted.append(seed)
        future = Future()
        future.set_result({"seed": seed, "completed": True})
        return future


def test_multiple_seeds_use_one_process_slot_per_seed_by_default():
    ImmediateExecutor.instances.clear()
    callbacks = []
    args = SimpleNamespace(seed_workers=None)

    results = run_seed_batch(
        args,
        [2, 0, 1],
        "test",
        Path("."),
        on_result=lambda partial: callbacks.append(list(partial)),
        executor_cls=ImmediateExecutor,
    )

    executor = ImmediateExecutor.instances[-1]
    assert executor.max_workers == 3
    assert executor.submitted == [2, 0, 1]
    assert [row["seed"] for row in results] == [2, 0, 1]
    assert [row["seed"] for row in callbacks[-1]] == [2, 0, 1]


def test_seed_worker_limit_caps_process_count():
    ImmediateExecutor.instances.clear()
    args = SimpleNamespace(seed_workers=2)

    run_seed_batch(
        args,
        [0, 1, 2],
        "test",
        Path("."),
        executor_cls=ImmediateExecutor,
    )

    assert ImmediateExecutor.instances[-1].max_workers == 2
