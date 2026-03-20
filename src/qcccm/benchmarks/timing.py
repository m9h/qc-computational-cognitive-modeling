"""Core timing utilities for benchmarks."""

from __future__ import annotations

import time
from typing import Any, Callable, NamedTuple

import numpy as np


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark."""

    name: str
    sizes: list[int]
    times_mean: list[float]
    times_std: list[float]
    jit_compile_time: float | None = None


def time_fn(
    fn: Callable,
    *args: Any,
    n_warmup: int = 1,
    n_repeats: int = 5,
) -> tuple[float, float, float]:
    """Time a JAX function, separating JIT compilation from execution.

    Returns (jit_time, mean_time, std_time) in seconds.
    """
    # First call includes JIT compilation
    t0 = time.perf_counter()
    result = fn(*args)
    if hasattr(result, "block_until_ready"):
        result.block_until_ready()
    jit_time = time.perf_counter() - t0

    # Warm-up calls (ensure JIT cache is hot)
    for _ in range(n_warmup):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed calls
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        times.append(time.perf_counter() - t0)

    return jit_time, float(np.mean(times)), float(np.std(times))


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    for r in results:
        print(f"\n{'=' * 60}")
        print(f"  {r.name}")
        print(f"{'=' * 60}")
        if r.jit_compile_time is not None:
            print(f"  JIT compile: {r.jit_compile_time * 1000:.1f} ms")
        print(f"  {'Size':>8s}  {'Mean (ms)':>10s}  {'Std (ms)':>10s}")
        print(f"  {'-' * 32}")
        for s, m, sd in zip(r.sizes, r.times_mean, r.times_std):
            print(f"  {s:>8d}  {m * 1000:>10.2f}  {sd * 1000:>10.2f}")
