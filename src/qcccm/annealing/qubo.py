"""QUBO formulation for multi-agent policy optimisation.

Maps the problem of assigning optimal policies to networked agents into
a Quadratic Unconstrained Binary Optimisation (QUBO) form suitable for
quantum annealing (D-Wave) or classical solvers.

Binary variables: x[i*K + k] = 1 if agent i selects policy k.
Constraint: exactly one policy per agent (one-hot, enforced via penalty).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from jax import Array


class PolicyAssignment(NamedTuple):
    """Problem specification for multi-agent policy optimisation."""

    n_agents: int
    n_policies: int
    efe_matrix: np.ndarray  # (n_agents, n_policies) EFE values
    adjacency: np.ndarray  # (n_agents, n_agents) network adjacency
    interaction_strength: float = 1.0  # coupling between neighbours
    penalty_strength: float = 10.0  # one-hot constraint penalty


# ---------------------------------------------------------------------------
# QUBO construction
# ---------------------------------------------------------------------------


def build_qubo(assignment: PolicyAssignment) -> dict[tuple[int, int], float]:
    """Build QUBO dictionary from a policy assignment problem.

    Variable indexing: v = agent_idx * n_policies + policy_idx.

    The QUBO encodes:
    1. Linear terms: G[i, k] for each (agent, policy) pair
    2. One-hot constraint: P * (Σ_k x[i,k] - 1)² per agent
    3. Interaction: J * |G[i,k] - G[j,l]| for neighbouring agents

    Args:
        assignment: problem specification.

    Returns:
        QUBO dict mapping (var_a, var_b) → coefficient.
    """
    N = assignment.n_agents
    K = assignment.n_policies
    G = np.asarray(assignment.efe_matrix)
    adj = np.asarray(assignment.adjacency)
    P = assignment.penalty_strength
    J = assignment.interaction_strength

    Q: dict[tuple[int, int], float] = {}

    def _add(i: int, j: int, val: float) -> None:
        key = (min(i, j), max(i, j))
        Q[key] = Q.get(key, 0.0) + val

    for i in range(N):
        for k in range(K):
            v = i * K + k

            # 1. Linear: EFE cost
            _add(v, v, G[i, k])

            # 2. One-hot constraint: P * (Σ_k x[i,k] - 1)²
            # Expanded: P * (Σ_k x_k² + 2 Σ_{k<l} x_k x_l - 2 Σ_k x_k + 1)
            # Since x² = x for binary: diagonal += P * (1 - 2) = -P
            _add(v, v, -P)

            # Cross terms within agent's variables
            for m in range(k + 1, K):
                w = i * K + m
                _add(v, w, 2.0 * P)

    # 3. Interaction terms between neighbouring agents
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] > 0:
                for k in range(K):
                    for m in range(K):
                        v = i * K + k
                        w = j * K + m
                        cost = J * adj[i, j] * abs(G[i, k] - G[j, m])
                        if cost > 1e-10:
                            _add(v, w, cost)

    return Q


def decode_qubo_solution(
    sample: dict[int, int],
    n_agents: int,
    n_policies: int,
) -> np.ndarray:
    """Convert a QUBO sample to a policy assignment.

    Args:
        sample: {variable_index: 0 or 1} from the solver.
        n_agents: number of agents.
        n_policies: number of policies.

    Returns:
        (n_agents,) array of selected policy indices.
    """
    assignments = np.zeros(n_agents, dtype=int)
    for i in range(n_agents):
        best_k = 0
        for k in range(n_policies):
            v = i * n_policies + k
            if sample.get(v, 0) == 1:
                best_k = k
                break
        assignments[i] = best_k
    return assignments


def efe_to_qubo(
    efe_matrix: Array | np.ndarray,
    adjacency: Array | np.ndarray,
    interaction_strength: float = 1.0,
    penalty_strength: float = 10.0,
) -> dict[tuple[int, int], float]:
    """Convenience: build QUBO from EFE matrix and adjacency.

    Args:
        efe_matrix: (n_agents, n_policies) EFE values.
        adjacency: (n_agents, n_agents) network adjacency.
        interaction_strength: coupling between neighbours.
        penalty_strength: one-hot constraint penalty.

    Returns:
        QUBO dictionary.
    """
    efe = np.asarray(efe_matrix)
    adj = np.asarray(adjacency)
    assignment = PolicyAssignment(
        n_agents=efe.shape[0],
        n_policies=efe.shape[1],
        efe_matrix=efe,
        adjacency=adj,
        interaction_strength=interaction_strength,
        penalty_strength=penalty_strength,
    )
    return build_qubo(assignment)
