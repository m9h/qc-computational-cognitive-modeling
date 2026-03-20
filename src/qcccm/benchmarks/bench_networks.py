"""Network evolution scaling benchmarks."""

from __future__ import annotations

import jax.numpy as jnp

from qcccm.benchmarks.timing import BenchmarkResult, time_fn
from qcccm.networks.multi_agent import (
    NetworkEvolutionParams,
    init_network_state,
    network_evolution,
)
from qcccm.networks.observables import network_entropy
from qcccm.networks.topology import complete_graph


def bench_network_evolution(
    agent_counts: list[int] | None = None,
    n_outcomes: int = 3,
) -> BenchmarkResult:
    """Time network_evolution for increasing agent counts."""
    if agent_counts is None:
        agent_counts = [3, 5, 10, 20]

    means, stds = [], []
    jit_t0 = None

    for n in agent_counts:
        topo = complete_graph(n)
        beliefs = jnp.ones((n, n_outcomes)) / n_outcomes
        state = init_network_state(topo, beliefs)
        params = NetworkEvolutionParams(n_steps=10, coupling_strength=0.3)

        jit_t, mean_t, std_t = time_fn(network_evolution, state, params, n_repeats=3)
        if jit_t0 is None:
            jit_t0 = jit_t
        means.append(mean_t)
        stds.append(std_t)

    return BenchmarkResult(
        name="Network Evolution Scaling (n_agents, n_steps=10)",
        sizes=agent_counts,
        times_mean=means,
        times_std=stds,
        jit_compile_time=jit_t0,
    )


def bench_network_observables(
    agent_counts: list[int] | None = None,
) -> BenchmarkResult:
    """Time network observables at increasing agent counts."""
    if agent_counts is None:
        agent_counts = [3, 5, 10]

    means, stds = [], []

    for n in agent_counts:
        topo = complete_graph(n)
        beliefs = jnp.ones((n, 3)) / 3
        state = init_network_state(topo, beliefs)

        _, mean_t, std_t = time_fn(network_entropy, state, n_repeats=3)
        means.append(mean_t)
        stds.append(std_t)

    return BenchmarkResult(
        name="network_entropy Scaling (n_agents)",
        sizes=agent_counts,
        times_mean=means,
        times_std=stds,
    )
