"""Tests for neural state density matrix representations."""

import jax.numpy as jnp
import pytest

from qcccm.neuroai.neural_states import (
    decode_neural_state,
    firing_rates_to_density_matrix,
    neural_entropy,
    neural_fidelity_trajectory,
    neural_mutual_information,
)


class TestFiringRatesToDensityMatrix:
    def test_single_neuron(self):
        """1 neuron with rate 0.7 → 2×2 diagonal (0.3, 0.7)."""
        rho = firing_rates_to_density_matrix(jnp.array([0.7]))
        assert rho.shape == (2, 2)
        assert float(rho[0, 0].real) == pytest.approx(0.3, abs=1e-4)
        assert float(rho[1, 1].real) == pytest.approx(0.7, abs=1e-4)

    def test_two_neuron_product_state(self):
        """2 independent neurons → 4×4 diagonal, product of marginals."""
        rho = firing_rates_to_density_matrix(jnp.array([0.6, 0.8]))
        assert rho.shape == (4, 4)
        # |00⟩ = (1-0.6)*(1-0.8) = 0.08
        assert float(rho[0, 0].real) == pytest.approx(0.08, abs=1e-3)
        # |11⟩ = 0.6 * 0.8 = 0.48
        assert float(rho[3, 3].real) == pytest.approx(0.48, abs=1e-3)

    def test_trace_one(self):
        """Density matrix should always have Tr(ρ) = 1."""
        for rates in [[0.5], [0.3, 0.7], [0.2, 0.5, 0.8]]:
            rho = firing_rates_to_density_matrix(jnp.array(rates))
            assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-5)

    def test_correlations_add_offdiagonal(self):
        """With correlations, off-diagonal elements should be non-zero."""
        rates = jnp.array([0.5, 0.5])
        corr = jnp.array([[0.0, 0.3], [0.3, 0.0]])
        rho = firing_rates_to_density_matrix(rates, correlations=corr)
        off_diag_sum = float(jnp.sum(jnp.abs(rho)) - jnp.sum(jnp.abs(jnp.diag(rho))))
        assert off_diag_sum > 0.01

    def test_positive_semidefinite(self):
        """Eigenvalues should all be ≥ 0."""
        rates = jnp.array([0.6, 0.4])
        corr = jnp.array([[0.0, 0.5], [0.5, 0.0]])
        rho = firing_rates_to_density_matrix(rates, correlations=corr)
        eigvals = jnp.linalg.eigvalsh(rho).real
        assert jnp.all(eigvals >= -1e-5)

    def test_correlations_preserve_trace(self):
        """Trace should remain 1 even with strong correlations."""
        rates = jnp.array([0.3, 0.7, 0.5])
        corr = jnp.array([[0, 0.4, 0.1], [0.4, 0, 0.2], [0.1, 0.2, 0]])
        rho = firing_rates_to_density_matrix(rates, correlations=corr)
        assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-5)


class TestNeuralEntropy:
    def test_definite_state_low_entropy(self):
        """Rate=1.0 (always firing) → low entropy."""
        rho = firing_rates_to_density_matrix(jnp.array([1.0 - 1e-6]))
        assert float(neural_entropy(rho)) < 0.1

    def test_uncertain_state_high_entropy(self):
        """Rate=0.5 → maximum entropy for 1 qubit (ln 2)."""
        rho = firing_rates_to_density_matrix(jnp.array([0.5]))
        assert float(neural_entropy(rho)) == pytest.approx(float(jnp.log(2.0)), abs=0.01)


class TestDecodeNeuralState:
    def test_roundtrip_no_correlations(self):
        """Encode then decode should recover original rates."""
        rates_in = jnp.array([0.3, 0.7])
        rho = firing_rates_to_density_matrix(rates_in)
        rates_out, corr_out = decode_neural_state(rho, n_neurons=2)
        assert jnp.allclose(rates_out, rates_in, atol=1e-3)
        # No correlations → correlations should be near zero
        assert jnp.allclose(corr_out, 0.0, atol=1e-3)


class TestNeuralMutualInformation:
    def test_independent_neurons_zero_mi(self):
        """Uncorrelated neurons → MI ≈ 0."""
        rates = jnp.array([0.5, 0.5])
        rho = firing_rates_to_density_matrix(rates)
        mi = neural_mutual_information(rho, n_neurons=2, partition=((0,), (1,)))
        assert float(mi) == pytest.approx(0.0, abs=0.05)


class TestNeuralFidelityTrajectory:
    def test_static_trajectory(self):
        """Constant state over time → fidelity = 1 at each step."""
        rho = firing_rates_to_density_matrix(jnp.array([0.5]))
        trajectory = jnp.stack([rho] * 5)
        fids = neural_fidelity_trajectory(trajectory)
        assert fids.shape == (4,)
        assert jnp.allclose(fids, 1.0, atol=1e-4)
