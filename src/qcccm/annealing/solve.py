"""Solver interface: D-Wave, simulated annealing, brute-force fallback.

The brute-force fallback works without dwave-ocean-sdk installed,
enabling tests to run on any machine.
"""

from __future__ import annotations

from itertools import product
from typing import Any, NamedTuple

import numpy as np
from jax import Array

from qcccm.annealing.qubo import (
    decode_qubo_solution,
    efe_to_qubo,
)


class PolicySolution(NamedTuple):
    """Result of solving a policy assignment problem."""

    assignments: np.ndarray  # (n_agents,) policy indices
    energy: float
    method: str
    raw_response: Any = None


# ---------------------------------------------------------------------------
# QUBO energy evaluation
# ---------------------------------------------------------------------------


def _evaluate_qubo(Q: dict[tuple[int, int], float], sample: dict[int, int]) -> float:
    """Evaluate QUBO energy for a given sample."""
    energy = 0.0
    for (i, j), coeff in Q.items():
        xi = sample.get(i, 0)
        xj = sample.get(j, 0)
        if i == j:
            energy += coeff * xi
        else:
            energy += coeff * xi * xj
    return energy


# ---------------------------------------------------------------------------
# Brute-force solver (no dwave needed)
# ---------------------------------------------------------------------------


def _solve_brute_force(
    Q: dict[tuple[int, int], float],
    n_agents: int,
    n_policies: int,
) -> PolicySolution:
    """Brute-force: enumerate all valid one-hot assignments."""
    best_energy = float("inf")
    best_sample: dict[int, int] = {}

    # Enumerate all policy assignments (K^N combinations)
    for combo in product(range(n_policies), repeat=n_agents):
        sample = {}
        for i in range(n_agents):
            for k in range(n_policies):
                v = i * n_policies + k
                sample[v] = 1 if combo[i] == k else 0

        energy = _evaluate_qubo(Q, sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample.copy()

    assignments = decode_qubo_solution(best_sample, n_agents, n_policies)
    return PolicySolution(
        assignments=assignments,
        energy=best_energy,
        method="brute_force",
    )


# ---------------------------------------------------------------------------
# Simulated annealing (uses dimod if available, else random search)
# ---------------------------------------------------------------------------


def _solve_simulated(
    Q: dict[tuple[int, int], float],
    n_agents: int,
    n_policies: int,
    num_reads: int = 100,
) -> PolicySolution:
    """Simulated annealing via dimod, or random search fallback."""
    try:
        import dimod

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        sampler = dimod.SimulatedAnnealingSampler()
        response = sampler.sample(bqm, num_reads=num_reads)
        best = response.first
        sample = dict(best.sample)
        return PolicySolution(
            assignments=decode_qubo_solution(sample, n_agents, n_policies),
            energy=best.energy,
            method="simulated_annealing",
            raw_response=response,
        )
    except ImportError:
        # Fallback: random search with one-hot constraint
        n_vars = n_agents * n_policies
        best_energy = float("inf")
        best_sample: dict[int, int] = {}
        rng = np.random.RandomState(42)

        for _ in range(num_reads):
            sample: dict[int, int] = {v: 0 for v in range(n_vars)}
            for i in range(n_agents):
                k = rng.randint(0, n_policies)
                sample[i * n_policies + k] = 1

            energy = _evaluate_qubo(Q, sample)
            if energy < best_energy:
                best_energy = energy
                best_sample = sample.copy()

        return PolicySolution(
            assignments=decode_qubo_solution(best_sample, n_agents, n_policies),
            energy=best_energy,
            method="random_search",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve_policy_qubo(
    Q: dict[tuple[int, int], float],
    n_agents: int,
    n_policies: int,
    method: str = "simulated",
    num_reads: int = 100,
) -> PolicySolution:
    """Solve a policy assignment QUBO.

    Args:
        Q: QUBO dictionary from build_qubo.
        n_agents: number of agents.
        n_policies: number of policies.
        method: "brute_force", "simulated", or "dwave".
        num_reads: number of annealing reads.

    Returns:
        PolicySolution with optimal assignments.
    """
    if method == "brute_force":
        return _solve_brute_force(Q, n_agents, n_policies)

    if method == "simulated":
        return _solve_simulated(Q, n_agents, n_policies, num_reads)

    if method == "dwave":
        try:
            import dimod
            from dwave.system import DWaveSampler, EmbeddingComposite
        except ImportError:
            raise ImportError(
                "D-Wave SDK required. Install with: uv pip install 'qcccm[annealing]'"
            ) from None

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=num_reads)
        best = response.first
        sample = dict(best.sample)
        return PolicySolution(
            assignments=decode_qubo_solution(sample, n_agents, n_policies),
            energy=best.energy,
            method="dwave",
            raw_response=response,
        )

    raise ValueError(f"Unknown method '{method}'. Use 'brute_force', 'simulated', or 'dwave'.")


def solve_policy_assignment(
    efe_matrix: Array | np.ndarray,
    adjacency: Array | np.ndarray,
    method: str = "simulated",
    interaction_strength: float = 1.0,
    penalty_strength: float = 10.0,
    num_reads: int = 100,
) -> PolicySolution:
    """End-to-end: build QUBO from EFE matrix and solve.

    Args:
        efe_matrix: (n_agents, n_policies) EFE values.
        adjacency: (n_agents, n_agents) network adjacency.
        method: solver method.
        interaction_strength: coupling between neighbours.
        penalty_strength: one-hot constraint penalty.
        num_reads: number of annealing reads.

    Returns:
        PolicySolution with optimal assignments.
    """
    efe = np.asarray(efe_matrix)
    Q = efe_to_qubo(efe, adjacency, interaction_strength, penalty_strength)
    return solve_policy_qubo(
        Q, n_agents=efe.shape[0], n_policies=efe.shape[1],
        method=method, num_reads=num_reads,
    )
