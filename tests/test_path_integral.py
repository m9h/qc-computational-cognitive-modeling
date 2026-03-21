"""Tests for Feynman path integral evidence accumulation."""

import numpy as np
import pytest

from qcccm.neuroai.path_integral import (
    PathIntegralParams,
    classical_action,
    classical_vs_quantum_paths,
    evidence_accumulation_density,
    path_integral_fpt,
    path_integral_propagator,
)


class TestClassicalAction:
    def test_straight_path_at_drift(self):
        """Path following drift exactly should have near-zero action."""
        drift = 1.0
        dt = 0.01
        n_steps = 100
        path = np.arange(n_steps) * drift * dt  # straight line at drift rate
        S = classical_action(path, dt, drift=drift, diffusion=1.0)
        assert S == pytest.approx(0.0, abs=0.1)

    def test_deviation_increases_action(self):
        """Path deviating from drift should have higher action."""
        dt = 0.01
        n_steps = 50
        straight = np.arange(n_steps) * 0.5 * dt
        wobbly = straight + np.sin(np.arange(n_steps) * 0.5) * 0.5

        S_straight = classical_action(straight, dt, drift=0.5)
        S_wobbly = classical_action(wobbly, dt, drift=0.5)
        assert S_wobbly > S_straight


class TestPathIntegralPropagator:
    def test_returns_complex(self):
        """Propagator should be a complex number."""
        params = PathIntegralParams(n_paths=100, dt=0.05, T_max=0.5)
        K = path_integral_propagator(0.0, 0.5, 0.3, params)
        assert isinstance(K, complex)

    def test_finite_value(self):
        """Propagator should be finite."""
        params = PathIntegralParams(n_paths=200, dt=0.05, T_max=0.5)
        K = path_integral_propagator(0.0, 0.2, 0.2, params)
        assert np.isfinite(abs(K))


class TestEvidenceAccumulationDensity:
    def test_output_shapes(self):
        """Should return matching times, grid, and density arrays."""
        params = PathIntegralParams(n_paths=50, dt=0.1, T_max=0.5)
        times, x_grid, density = evidence_accumulation_density(params, n_grid=21)
        assert len(times) > 0
        assert len(x_grid) == 21
        assert density.shape == (len(times), 21)

    def test_density_nonnegative(self):
        """Probability density should be non-negative."""
        params = PathIntegralParams(n_paths=50, dt=0.1, T_max=0.3)
        _, _, density = evidence_accumulation_density(params, n_grid=11)
        assert np.all(density >= -1e-8)


class TestPathIntegralFPT:
    def test_fpt_nonnegative(self):
        """FPT density should be non-negative."""
        params = PathIntegralParams(n_paths=500, dt=0.05, T_max=1.0, drift=1.0)
        times, fpt = path_integral_fpt(params, boundary=0.5)
        assert np.all(fpt >= -1e-8)

    def test_fpt_sums_to_at_most_one(self):
        """Total FPT probability should be ≤ 1."""
        params = PathIntegralParams(n_paths=500, dt=0.05, T_max=2.0, drift=0.5)
        times, fpt = path_integral_fpt(params, boundary=0.5)
        assert float(np.sum(fpt)) <= 1.0 + 1e-5


class TestClassicalVsQuantum:
    def test_output_shapes(self):
        """Should return three matching arrays."""
        params = PathIntegralParams(n_paths=50, dt=0.1, T_max=0.3)
        times, classical, quantum = classical_vs_quantum_paths(params)
        assert len(times) > 0
        assert classical.shape[0] == len(times)
        assert quantum.shape[0] == len(times)
        assert classical.shape[1] == quantum.shape[1]

    def test_classical_is_peaked(self):
        """Classical density should be peaked (Gaussian)."""
        params = PathIntegralParams(n_paths=50, dt=0.1, T_max=0.3, drift=1.0)
        times, classical, _ = classical_vs_quantum_paths(params)
        # At the last time step, density should peak somewhere
        assert np.max(classical[-1]) > 0
