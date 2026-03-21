"""Tests for multiscale neural circuit simulations."""

import jax.numpy as jnp
import pytest

from qcccm.neuroai.multiscale import (
    NeuralCircuitParams,
    coarse_grain,
    make_neural_qnode,
    multiscale_hierarchy,
    neural_state_tomography,
)


class TestNeuralQNode:
    def test_output_shape(self):
        """QNode should return 2^N probabilities."""
        params = NeuralCircuitParams(n_neurons=2, n_layers=1)
        qnode = make_neural_qnode(params)
        rates = jnp.array([0.6, 0.4])
        weights = jnp.array([[0.0, 0.5], [0.5, 0.0]])
        result = qnode(rates, weights)
        assert result.shape == (4,)

    def test_output_is_valid_distribution(self):
        """Probabilities should sum to 1."""
        params = NeuralCircuitParams(n_neurons=2, n_layers=2)
        qnode = make_neural_qnode(params)
        rates = jnp.array([0.7, 0.3])
        weights = jnp.array([[0.0, 0.3], [0.3, 0.0]])
        result = qnode(rates, weights)
        assert float(jnp.sum(result)) == pytest.approx(1.0, abs=1e-5)
        assert jnp.all(result >= -1e-8)

    def test_zero_weights_preserves_encoding(self):
        """Zero synaptic weights should not change the encoded state."""
        params = NeuralCircuitParams(n_neurons=2, n_layers=1)
        qnode = make_neural_qnode(params)
        rates = jnp.array([0.9, 0.1])
        weights = jnp.zeros((2, 2))
        result = qnode(rates, weights)
        # With zero weights, output should reflect encoded rates
        # P(|00⟩) ≈ (1-0.9)*(1-0.1) = 0.09
        assert float(result[0]) == pytest.approx(0.09, abs=0.05)


class TestNeuralStateTomography:
    def test_reconstructed_trace_one(self):
        """Reconstructed density matrix should have trace 1."""
        params = NeuralCircuitParams(n_neurons=2, n_layers=1)
        qnode = make_neural_qnode(params)
        rates = jnp.array([0.5, 0.5])
        weights = jnp.array([[0.0, 0.2], [0.2, 0.0]])
        rho = neural_state_tomography(qnode, rates, weights, n_qubits=2)
        assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-5)

    def test_reconstructed_psd(self):
        """Eigenvalues should all be ≥ 0."""
        params = NeuralCircuitParams(n_neurons=2, n_layers=1)
        qnode = make_neural_qnode(params)
        rates = jnp.array([0.6, 0.4])
        weights = jnp.array([[0.0, 0.3], [0.3, 0.0]])
        rho = neural_state_tomography(qnode, rates, weights, n_qubits=2)
        eigvals = jnp.linalg.eigvalsh(rho).real
        assert jnp.all(eigvals >= -1e-5)


class TestMultiscaleHierarchy:
    def test_hierarchy_structure(self):
        """Hierarchy should have correct params and groupings."""
        h = multiscale_hierarchy(n_micro=8, n_meso=4, n_macro=2)
        assert h.micro.n_neurons == 8
        assert h.meso.n_neurons == 4
        assert h.macro.n_neurons == 2
        assert len(h.micro_grouping) == 4  # 4 meso groups
        assert len(h.meso_grouping) == 2  # 2 macro groups

    def test_grouping_covers_all_qubits(self):
        """All micro qubits should appear in exactly one meso group."""
        h = multiscale_hierarchy(n_micro=8, n_meso=4, n_macro=2)
        all_qubits = set()
        for group in h.micro_grouping:
            all_qubits.update(group)
        assert all_qubits == set(range(8))


class TestCoarseGrain:
    def test_preserves_trace(self):
        """Coarse-grained density matrix should have trace 1."""
        # 4-qubit state → 2-qubit coarse-grained
        rho = jnp.eye(16, dtype=jnp.complex64) / 16  # maximally mixed
        grouping = ((0, 1), (2, 3))
        rho_coarse = coarse_grain(rho, n_qubits=4, grouping=grouping)
        assert jnp.trace(rho_coarse).real == pytest.approx(1.0, abs=1e-4)

    def test_reduces_dimension(self):
        """Output should be smaller than input."""
        rho = jnp.eye(8, dtype=jnp.complex64) / 8  # 3-qubit maximally mixed
        grouping = ((0,), (1, 2))  # keep qubit 0, trace out one from group (1,2)
        rho_coarse = coarse_grain(rho, n_qubits=3, grouping=grouping)
        assert rho_coarse.shape[0] < rho.shape[0]
