"""Tests for games/agreement ↔ spin_glass integration."""

import numpy as np

from qcccm.games.integration import (
    agreement_to_spin_glass,
    benchmark_agreement_solvers,
)
from qcccm.games.agreement import AgreementParams


class TestAgreementToSpinGlass:
    def test_returns_valid_params(self):
        """Conversion should produce valid SocialSpinGlassParams."""
        params = AgreementParams(n_agents=8, frustration=0.3, seed=42)
        sg = agreement_to_spin_glass(params)
        assert sg.n_agents == 8
        assert sg.adjacency.shape == (8, 8)
        assert sg.J.shape == (8, 8)
        assert sg.fields.shape == (8,)

    def test_frustration_creates_negative_couplings(self):
        """Non-zero frustration should produce some negative J_ij."""
        params = AgreementParams(n_agents=10, frustration=0.5, seed=42)
        sg = agreement_to_spin_glass(params)
        assert np.any(sg.J < 0)

    def test_zero_frustration_all_positive(self):
        """Zero frustration should produce all non-negative J_ij."""
        params = AgreementParams(n_agents=10, frustration=0.0, seed=42)
        sg = agreement_to_spin_glass(params)
        assert np.all(sg.J >= 0)


class TestBenchmarkSolvers:
    def test_returns_valid_result(self):
        """Benchmark should return a complete result."""
        result = benchmark_agreement_solvers(
            n_agents=6, frustration=0.2, n_steps=500, seed=42,
        )
        assert result.classical_spins.shape == (6,)
        assert result.quantum_spins.shape == (6,)
        assert np.isfinite(result.classical_energy)
        assert np.isfinite(result.quantum_energy)
        assert np.isfinite(result.energy_improvement)

    def test_quantum_finds_solution(self):
        """Quantum solver should find a finite-energy configuration."""
        result = benchmark_agreement_solvers(
            n_agents=6, frustration=0.3, transverse_field=1.0,
            n_steps=1000, seed=42,
        )
        assert np.isfinite(result.quantum_energy)
        assert set(np.unique(result.quantum_spins)).issubset({-1.0, 1.0})
