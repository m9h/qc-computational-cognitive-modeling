"""Tests for neural data → quantum state interface."""

import jax.numpy as jnp
import numpy as np
import pytest

from qcccm.neuroai.data_interface import (
    compute_correlations,
    neural_data_to_density_matrix,
    neural_state_fidelity_over_time,
    quantum_neural_analysis,
    spike_raster_to_rates,
    spike_trains_to_rates,
)


class TestSpikeTrainsToRates:
    def test_output_shape(self):
        """Should produce (n_bins, n_units) rate array."""
        spike_times = [
            np.array([0.1, 0.2, 0.5, 0.8]),
            np.array([0.3, 0.6, 0.9]),
        ]
        rates, edges = spike_trains_to_rates(spike_times, duration_s=1.0, bin_size_s=0.5)
        assert rates.shape == (2, 2)  # 2 bins, 2 units
        assert len(edges) == 3

    def test_rates_nonnegative(self):
        """Firing rates should be non-negative."""
        spike_times = [np.array([0.1, 0.5]), np.array([0.3])]
        rates, _ = spike_trains_to_rates(spike_times, duration_s=1.0)
        assert np.all(rates >= 0)


class TestSpikeRasterToRates:
    def test_output_shape(self):
        """Should bin raster into fewer time bins."""
        raster = np.random.randint(0, 2, size=(1000, 10))
        rates = spike_raster_to_rates(raster, dt_ms=0.5, bin_ms=50.0)
        assert rates.shape == (10, 10)  # 1000*0.5 = 500ms / 50ms = 10 bins

    def test_rates_bounded(self):
        """Rates should be in [0, 1]."""
        raster = np.random.randint(0, 2, size=(500, 5))
        rates = spike_raster_to_rates(raster, dt_ms=0.5, bin_ms=25.0)
        assert np.all(rates >= 0)
        assert np.all(rates <= 1)


class TestComputeCorrelations:
    def test_diagonal_is_zero(self):
        """Diagonal of correlation matrix should be 0."""
        rates = np.random.rand(100, 4)
        corr = compute_correlations(rates)
        assert np.allclose(np.diag(corr), 0.0)

    def test_symmetric(self):
        """Correlation matrix should be symmetric."""
        rates = np.random.rand(100, 4)
        corr = compute_correlations(rates)
        assert np.allclose(corr, corr.T, atol=1e-10)


class TestNeuralDataToDensityMatrix:
    def test_trace_one(self):
        """Density matrix should have trace 1."""
        rates = np.random.rand(50, 4) * 0.5 + 0.1
        rho = neural_data_to_density_matrix(rates, neuron_indices=[0, 1, 2])
        assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-4)

    def test_positive_semidefinite(self):
        """Eigenvalues should be non-negative."""
        rates = np.random.rand(50, 4) * 0.5 + 0.1
        rho = neural_data_to_density_matrix(rates, neuron_indices=[0, 1])
        eigvals = jnp.linalg.eigvalsh(rho).real
        assert jnp.all(eigvals >= -1e-5)

    def test_too_many_neurons_raises(self):
        """More than 8 neurons should raise ValueError."""
        rates = np.random.rand(50, 10) * 0.5 + 0.1
        with pytest.raises(ValueError, match="Cannot construct"):
            neural_data_to_density_matrix(rates, neuron_indices=list(range(10)))


class TestQuantumNeuralAnalysis:
    def test_returns_correct_keys(self):
        """Analysis should return dict with expected keys."""
        rates = np.random.rand(50, 4) * 0.5 + 0.1
        result = quantum_neural_analysis(rates, neuron_indices=[0, 1], time_bins=[0, 25, 49])
        assert "entropy" in result
        assert "purity" in result
        assert len(result["entropy"]) == 3

    def test_entropy_is_positive(self):
        """Entropy should be non-negative."""
        rates = np.random.rand(50, 3) * 0.5 + 0.1
        result = quantum_neural_analysis(rates, neuron_indices=[0, 1])
        assert np.all(result["entropy"] >= -1e-5)


class TestFidelityOverTime:
    def test_output_shapes(self):
        """Should return matching time and fidelity arrays."""
        rates = np.random.rand(20, 3) * 0.5 + 0.1
        times, fids = neural_state_fidelity_over_time(rates, neuron_indices=[0, 1])
        assert len(times) == len(fids)
        assert len(times) > 0

    def test_fidelity_bounded(self):
        """Fidelity should be in [0, 1]."""
        rates = np.random.rand(20, 3) * 0.5 + 0.1
        _, fids = neural_state_fidelity_over_time(rates, neuron_indices=[0, 1])
        assert np.all(fids >= -0.01)
        assert np.all(fids <= 1.01)
