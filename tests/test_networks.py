"""Tests for the networks module."""

import jax.numpy as jnp
import pytest

from qcccm.networks.topology import (
    adjacency_to_stochastic,
    complete_graph,
    ring_graph,
    star_graph,
)
from qcccm.networks.multi_agent import (
    NetworkEvolutionParams,
    init_network_state,
    network_evolution,
    network_evolution_step,
)
from qcccm.networks.observables import (
    belief_polarization,
    mean_pairwise_fidelity,
    network_coherence,
    network_entropy,
)


class TestTopology:
    def test_complete_graph_shape(self):
        """Complete graph adjacency should be (n, n) with zeros on diagonal."""
        topo = complete_graph(4)
        assert topo.adjacency.shape == (4, 4)
        assert jnp.all(jnp.diag(topo.adjacency) == 0)
        # Off-diagonal entries should all be 1
        off_diag = topo.adjacency.at[jnp.diag_indices(4)].set(1.0)
        assert jnp.all(off_diag == 1)

    def test_ring_graph_degree(self):
        """Each node in a ring graph should have exactly 2 neighbours."""
        topo = ring_graph(6)
        degrees = jnp.sum(topo.adjacency, axis=1)
        assert jnp.allclose(degrees, 2.0)

    def test_star_graph_center_degree(self):
        """Center node (0) should have n−1 connections."""
        n = 5
        topo = star_graph(n)
        assert float(jnp.sum(topo.adjacency[0])) == pytest.approx(n - 1)
        # Peripheral nodes have degree 1
        for i in range(1, n):
            assert float(jnp.sum(topo.adjacency[i])) == pytest.approx(1.0)

    def test_adjacency_to_stochastic(self):
        """Columns of stochastic matrix should sum to 1."""
        topo = complete_graph(4)
        W = adjacency_to_stochastic(topo.adjacency)
        col_sums = jnp.sum(W, axis=0)
        assert jnp.allclose(col_sums, 1.0, atol=1e-5)


class TestMultiAgentState:
    def test_init_from_beliefs(self):
        """init_network_state should produce valid density matrices."""
        topo = complete_graph(3)
        beliefs = jnp.array([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]])
        state = init_network_state(topo, beliefs)
        assert state.beliefs.shape == (3, 2, 2)
        # Each density matrix should have trace 1
        for i in range(3):
            trace = jnp.trace(state.beliefs[i]).real
            assert trace == pytest.approx(1.0, abs=1e-5)


class TestNetworkEvolution:
    def test_single_step_preserves_trace(self):
        """Tr(ρ_i) should remain 1 after one step."""
        topo = complete_graph(3)
        beliefs = jnp.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        state = init_network_state(topo, beliefs)
        params = NetworkEvolutionParams(coupling_strength=0.3)
        new_state = network_evolution_step(state, params)
        for i in range(3):
            trace = jnp.trace(new_state.beliefs[i]).real
            assert trace == pytest.approx(1.0, abs=1e-5)

    def test_zero_coupling_no_change(self):
        """With α=0, agents should not change."""
        topo = complete_graph(3)
        beliefs = jnp.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        state = init_network_state(topo, beliefs)
        params = NetworkEvolutionParams(coupling_strength=0.0)
        new_state = network_evolution_step(state, params)
        assert jnp.allclose(state.beliefs, new_state.beliefs, atol=1e-5)

    def test_trajectory_shape(self):
        """network_evolution should return (n_steps+1, n_agents, d, d)."""
        topo = ring_graph(4)
        beliefs = jnp.array([[0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]])
        state = init_network_state(topo, beliefs)
        params = NetworkEvolutionParams(n_steps=5, coupling_strength=0.3)
        traj = network_evolution(state, params)
        assert traj.shape == (6, 4, 2, 2)

    def test_decoherence_reduces_coherence(self):
        """Non-zero decoherence should reduce off-diagonal elements."""
        topo = complete_graph(2)
        # Start with coherent states (manually inject off-diag)
        rho = jnp.array([
            [[0.5 + 0j, 0.3 + 0j], [0.3 + 0j, 0.5 + 0j]],
            [[0.5 + 0j, 0.2 + 0j], [0.2 + 0j, 0.5 + 0j]],
        ])
        from qcccm.networks.multi_agent import MultiAgentState
        state = MultiAgentState(beliefs=rho, topology=topo, n_outcomes=2)
        params = NetworkEvolutionParams(coupling_strength=0.0, decoherence_rate=0.5)
        new_state = network_evolution_step(state, params)
        # Off-diagonals should be smaller
        old_offdiag = jnp.sum(jnp.abs(rho[:, 0, 1]))
        new_offdiag = jnp.sum(jnp.abs(new_state.beliefs[:, 0, 1]))
        assert float(new_offdiag) < float(old_offdiag)


class TestObservables:
    def test_identical_beliefs_max_fidelity(self):
        """All agents with same beliefs → mean fidelity = 1."""
        topo = complete_graph(3)
        beliefs = jnp.array([[0.6, 0.4]] * 3)
        state = init_network_state(topo, beliefs)
        f = mean_pairwise_fidelity(state)
        assert float(f) == pytest.approx(1.0, abs=1e-4)

    def test_diagonal_beliefs_zero_coherence(self):
        """Classical (diagonal) beliefs → coherence = 0."""
        topo = complete_graph(3)
        beliefs = jnp.array([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5]])
        state = init_network_state(topo, beliefs)
        c = network_coherence(state)
        assert float(c) == pytest.approx(0.0, abs=1e-5)

    def test_polarization_zero_for_consensus(self):
        """Identical beliefs → polarisation = 0."""
        topo = complete_graph(4)
        beliefs = jnp.array([[0.5, 0.5]] * 4)
        state = init_network_state(topo, beliefs)
        p = belief_polarization(state)
        assert float(p) == pytest.approx(0.0, abs=1e-5)

    def test_entropy_finite(self):
        """network_entropy should return a finite value."""
        topo = ring_graph(3)
        beliefs = jnp.array([[0.8, 0.2], [0.5, 0.5], [0.3, 0.7]])
        state = init_network_state(topo, beliefs)
        s = network_entropy(state)
        assert jnp.isfinite(s)
