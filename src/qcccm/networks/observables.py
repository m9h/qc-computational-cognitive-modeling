"""Network-level measurements and order parameters."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from qcccm.core.density_matrix import fidelity, von_neumann_entropy
from qcccm.networks.multi_agent import (
    MultiAgentState,
    NetworkEvolutionParams,
    init_network_state,
    network_evolution,
)
from qcccm.networks.topology import NetworkTopology, adjacency_to_stochastic


# ---------------------------------------------------------------------------
# Per-snapshot observables
# ---------------------------------------------------------------------------


def network_entropy(state: MultiAgentState) -> Array:
    """Mean von Neumann entropy across all agents.

    Args:
        state: current network state.

    Returns:
        Scalar mean entropy.
    """
    entropies = jnp.array([von_neumann_entropy(state.beliefs[i]) for i in range(state.topology.n_agents)])
    return jnp.mean(entropies)


def belief_polarization(state: MultiAgentState) -> Array:
    """Variance of agent beliefs in the computational basis.

    Higher values → more polarised. Computes Var_i[diag(ρ_i)] averaged
    over outcomes.

    Args:
        state: current network state.

    Returns:
        Scalar polarisation measure.
    """
    # Extract diagonals: (n_agents, d)
    diags = jnp.diagonal(state.beliefs, axis1=1, axis2=2).real
    # Variance across agents, averaged over outcomes
    return jnp.mean(jnp.var(diags, axis=0))


def mean_pairwise_fidelity(state: MultiAgentState) -> Array:
    """Mean fidelity F(ρ_i, ρ_j) across all agent pairs.

    F = 1 when all agents have identical beliefs (consensus).

    Args:
        state: current network state.

    Returns:
        Scalar mean fidelity ∈ [0, 1].
    """
    n = state.topology.n_agents
    if n < 2:
        return jnp.array(1.0)

    total = jnp.array(0.0)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total = total + fidelity(state.beliefs[i], state.beliefs[j])
            count += 1
    return total / count


def network_coherence(state: MultiAgentState) -> Array:
    """Mean off-diagonal magnitude across all agents.

    C = (1/N) Σ_i (Σ_{j≠k} |ρ_i,jk|) / (d² − d)

    Zero for classical (diagonal) beliefs, positive for quantum states.

    Args:
        state: current network state.

    Returns:
        Scalar coherence ∈ [0, 1].
    """
    d = state.n_outcomes
    off_diag_mask = 1.0 - jnp.eye(d)
    off_diag_sum = jnp.sum(jnp.abs(state.beliefs) * off_diag_mask[None, :, :], axis=(1, 2))
    normalised = off_diag_sum / max(d * d - d, 1)
    return jnp.mean(normalised)


# ---------------------------------------------------------------------------
# Quantum vs classical comparison
# ---------------------------------------------------------------------------


def quantum_vs_classical_consensus(
    topology: NetworkTopology,
    initial_beliefs: Array,
    params: NetworkEvolutionParams,
) -> tuple[Array, Array, Array]:
    """Compare consensus dynamics: quantum network vs classical DeGroot model.

    Classical DeGroot: p_i(t+1) = Σ_j W_ij p_j(t)
    Quantum: full density matrix evolution with coherences.

    Args:
        topology: network topology.
        initial_beliefs: (n_agents, d) initial belief vectors.
        params: evolution parameters.

    Returns:
        times: (n_steps + 1,)
        classical_disagreement: (n_steps + 1,) mean pairwise L2 distance.
        quantum_fidelity: (n_steps + 1,) mean pairwise fidelity.
    """
    n_agents, d = initial_beliefs.shape
    W = adjacency_to_stochastic(topology.adjacency)

    # --- Classical DeGroot ---
    classical_trajectory = [initial_beliefs]
    p = initial_beliefs
    for _ in range(params.n_steps):
        p = (W @ p.T).T  # (n_agents, d)  — note: W acts on columns
        # Renormalise
        p = p / jnp.sum(p, axis=1, keepdims=True).clip(1e-12)
        classical_trajectory.append(p)
    classical_trajectory = jnp.stack(classical_trajectory)  # (T+1, n, d)

    # Classical disagreement: mean pairwise L2
    def _classical_disagreement(beliefs_t):
        total = jnp.array(0.0)
        count = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                total = total + jnp.sum((beliefs_t[i] - beliefs_t[j]) ** 2)
                count += 1
        return total / max(count, 1)

    classical_disagree = jnp.array(
        [_classical_disagreement(classical_trajectory[t]) for t in range(params.n_steps + 1)]
    )

    # --- Quantum evolution ---
    state0 = init_network_state(topology, initial_beliefs)
    q_traj = network_evolution(state0, params)  # (T+1, n, d, d)

    def _quantum_fidelity_t(beliefs_t):
        total = jnp.array(0.0)
        count = 0
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                total = total + fidelity(beliefs_t[i], beliefs_t[j])
                count += 1
        return total / max(count, 1)

    quantum_fid = jnp.array(
        [_quantum_fidelity_t(q_traj[t]) for t in range(params.n_steps + 1)]
    )

    times = jnp.arange(params.n_steps + 1, dtype=jnp.float32)
    return times, classical_disagree, quantum_fid
