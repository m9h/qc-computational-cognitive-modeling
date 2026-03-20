"""JAX-level benchmarks: JIT compilation, walk scaling, density matrix ops."""

from __future__ import annotations

from qcccm.benchmarks.timing import BenchmarkResult, time_fn
from qcccm.core.density_matrix import (
    fidelity,
    maximally_mixed,
    pure_state_density_matrix,
    purity,
    von_neumann_entropy,
)
from qcccm.core.quantum_walk import QuantumWalkParams, quantum_walk_evolution
from qcccm.core.states import plus_state


def bench_jit_compilation() -> BenchmarkResult:
    """Time JIT compilation vs execution for core functions."""
    rho = pure_state_density_matrix(plus_state())
    sigma = maximally_mixed(2)

    fns = {
        "von_neumann_entropy": (von_neumann_entropy, rho),
        "purity": (purity, rho),
        "fidelity": (fidelity, rho, sigma),
    }

    sizes, means, stds = [], [], []
    first_jit = None

    for i, (name, args) in enumerate(fns.items()):
        fn = args[0]
        fn_args = args[1:]
        jit_t, mean_t, std_t = time_fn(fn, *fn_args)
        if i == 0:
            first_jit = jit_t
        sizes.append(i)
        means.append(mean_t)
        stds.append(std_t)

    return BenchmarkResult(
        name="JIT Compilation (core functions)",
        sizes=sizes,
        times_mean=means,
        times_std=stds,
        jit_compile_time=first_jit,
    )


def bench_walk_scaling(
    site_sizes: list[int] | None = None,
) -> BenchmarkResult:
    """Time quantum_walk_evolution for increasing lattice sizes."""
    if site_sizes is None:
        site_sizes = [51, 101, 201, 501]

    means, stds = [], []
    jit_t0 = None

    for n in site_sizes:
        params = QuantumWalkParams(n_sites=n, n_steps=50, start_pos=n // 2)
        jit_t, mean_t, std_t = time_fn(quantum_walk_evolution, params, n_repeats=3)
        if jit_t0 is None:
            jit_t0 = jit_t
        means.append(mean_t)
        stds.append(std_t)

    return BenchmarkResult(
        name="Quantum Walk Scaling (n_sites, n_steps=50)",
        sizes=site_sizes,
        times_mean=means,
        times_std=stds,
        jit_compile_time=jit_t0,
    )


def bench_density_matrix_ops(
    dims: list[int] | None = None,
) -> BenchmarkResult:
    """Time density matrix operations at increasing Hilbert space dimensions."""
    if dims is None:
        dims = [2, 4, 8, 16]

    means, stds = [], []
    jit_t0 = None

    for d in dims:
        rho = maximally_mixed(d)
        jit_t, mean_t, std_t = time_fn(von_neumann_entropy, rho)
        if jit_t0 is None:
            jit_t0 = jit_t
        means.append(mean_t)
        stds.append(std_t)

    return BenchmarkResult(
        name="von_neumann_entropy Scaling (Hilbert space dim)",
        sizes=dims,
        times_mean=means,
        times_std=stds,
        jit_compile_time=jit_t0,
    )
