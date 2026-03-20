"""Multi-agent quantum cognitive state evolution.

Each agent maintains a density matrix (generalised beliefs). Interactions
are mediated by the network adjacency structure via a mean-field update.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.models.bridge import beliefs_to_density_matrix
from qcccm.networks.topology import NetworkTopology, adjacency_to_stochastic


class MultiAgentState(NamedTuple):
    """State of a multi-agent quantum cognitive network."""

    beliefs: Array  # (n_agents, d, d) density matrices
    topology: NetworkTopology
    n_outcomes: int  # d


class NetworkEvolutionParams(NamedTuple):
    """Parameters for multi-agent network evolution."""

    n_steps: int = 10
    coupling_strength: float = 0.5  # α: how much agents influence each other
    decoherence_rate: float = 0.0  # γ: rate of decoherence
    self_weight: float = 0.5  # included in stochastic matrix via self-loops


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_network_state(
    topology: NetworkTopology,
    initial_beliefs: Array,
) -> MultiAgentState:
    """Initialise a multi-agent network from classical belief vectors.

    Args:
        topology: network topology.
        initial_beliefs: (n_agents, d) probability vectors per agent.

    Returns:
        MultiAgentState with diagonal density matrices.
    """
    n_agents, d = initial_beliefs.shape
    rhos = jax.vmap(beliefs_to_density_matrix)(initial_beliefs)  # (n_agents, d, d)
    return MultiAgentState(beliefs=rhos, topology=topology, n_outcomes=d)


# ---------------------------------------------------------------------------
# Single-step evolution
# ---------------------------------------------------------------------------


def network_evolution_step(
    state: MultiAgentState,
    params: NetworkEvolutionParams,
) -> MultiAgentState:
    """One step of multi-agent quantum belief evolution.

    For each agent i:
    1. Compute weighted neighbour influence: ρ_neighbour_i = Σ_j W_ij ρ_j
    2. Mix: ρ_i' = α · ρ_neighbour_i + (1−α) · ρ_i
    3. Optional decoherence: zero out off-diagonals at rate γ

    Args:
        state: current network state.
        params: evolution parameters.

    Returns:
        Updated MultiAgentState.
    """
    W = adjacency_to_stochastic(state.topology.adjacency)  # (n, n)
    alpha = params.coupling_strength
    gamma = params.decoherence_rate
    d = state.n_outcomes

    # Weighted neighbour influence: for each agent i, sum W_ij * rho_j
    # W is (n, n), beliefs is (n, d, d) → einsum
    rho_influence = jnp.einsum("ij,jkl->ikl", W, state.beliefs)

    # Mix own beliefs with neighbour influence
    new_beliefs = alpha * rho_influence + (1.0 - alpha) * state.beliefs

    # Optional decoherence: blend toward diagonal
    if gamma > 0.0:
        diag_mask = jnp.eye(d, dtype=jnp.complex64)
        diagonal_part = new_beliefs * diag_mask[None, :, :]
        new_beliefs = (1.0 - gamma) * new_beliefs + gamma * diagonal_part

    # Ensure trace = 1 for each agent
    traces = jnp.trace(new_beliefs, axis1=1, axis2=2).real[:, None, None]
    new_beliefs = new_beliefs / jnp.clip(traces, 1e-12, None)

    return state._replace(beliefs=new_beliefs)


# ---------------------------------------------------------------------------
# Full evolution
# ---------------------------------------------------------------------------


def network_evolution(
    initial_state: MultiAgentState,
    params: NetworkEvolutionParams,
) -> Array:
    """Run full network evolution and return belief trajectory.

    Args:
        initial_state: starting network state.
        params: evolution parameters.

    Returns:
        (n_steps + 1, n_agents, d, d) array of density matrices.
    """
    def scan_fn(state_beliefs, _):
        state = initial_state._replace(beliefs=state_beliefs)
        new_state = network_evolution_step(state, params)
        return new_state.beliefs, new_state.beliefs

    _, trajectory = jax.lax.scan(
        scan_fn, initial_state.beliefs, None, length=params.n_steps
    )

    # Prepend initial state
    return jnp.concatenate(
        [initial_state.beliefs[None, :, :, :], trajectory], axis=0
    )
