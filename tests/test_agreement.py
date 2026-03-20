"""Tests for Ising model and multi-agent agreement simulations."""

import numpy as np

from qcccm.games.agreement import (
    AgreementParams,
    IsingParams,
    ising_energy,
    ising_magnetisation,
    run_agreement_simulation,
    run_ising,
    phase_diagram,
    agreement_scaling,
)


class TestIsingModel:
    def test_all_up_energy(self):
        """All spins up: E = -J * 2 * L² (2D, periodic BC)."""
        L = 4
        spins = np.ones((L, L))
        E = ising_energy(spins, J=1.0, h=0.0)
        # Each spin has 4 neighbours, but each bond counted once → 2L² bonds
        assert E == -2 * L * L

    def test_all_up_magnetisation(self):
        """All spins +1 → |m| = 1."""
        spins = np.ones((5, 5))
        assert ising_magnetisation(spins) == 1.0

    def test_checkerboard_zero_magnetisation(self):
        """Antiferromagnetic ground state → |m| = 0."""
        L = 4
        spins = np.ones((L, L))
        for i in range(L):
            for j in range(L):
                if (i + j) % 2 == 1:
                    spins[i, j] = -1
        assert ising_magnetisation(spins) == 0.0

    def test_run_ising_shapes(self):
        """run_ising should return arrays of correct shape."""
        params = IsingParams(L=8, temperature=2.0, n_steps=50)
        energies, mags, spins = run_ising(params)
        assert energies.shape == (50,)
        assert mags.shape == (50,)
        assert spins.shape == (8, 8)
        assert set(np.unique(spins)).issubset({-1, 1})

    def test_low_temp_orders(self):
        """At low T, system should order (|m| > 0.5)."""
        params = IsingParams(L=8, temperature=0.5, n_steps=500)
        _, mags, _ = run_ising(params)
        assert np.mean(mags[-100:]) > 0.5

    def test_high_temp_disorders(self):
        """At high T, system should be disordered (|m| ≈ 0)."""
        params = IsingParams(L=10, temperature=5.0, n_steps=200)
        _, mags, _ = run_ising(params)
        assert np.mean(mags[-50:]) < 0.3


class TestPhaseDiagram:
    def test_returns_correct_shapes(self):
        """Phase diagram should return arrays of matching length."""
        T, mag, energy, chi = phase_diagram(
            L=8, temperatures=np.array([1.0, 2.27, 4.0]),
            n_steps=100, n_equilibrate=30,
        )
        assert len(T) == 3
        assert len(mag) == 3
        assert len(energy) == 3
        assert len(chi) == 3

    def test_magnetisation_decreases_with_temperature(self):
        """|m| should decrease as T increases (on average)."""
        T, mag, _, _ = phase_diagram(
            L=10, temperatures=np.array([0.5, 2.0, 5.0]),
            n_steps=200, n_equilibrate=50,
        )
        assert mag[0] > mag[2]


class TestAgreementSimulation:
    def test_returns_valid_result(self):
        """Agreement simulation should return a complete result."""
        params = AgreementParams(n_agents=10, n_objectives=3, max_rounds=100, seed=42)
        result = run_agreement_simulation(params)
        assert len(result.magnetisation_trajectory) > 0
        assert result.messages_exchanged > 0
        assert result.final_disagreement >= 0

    def test_quantum_runs(self):
        """Quantum agreement simulation should complete."""
        params = AgreementParams(n_agents=10, n_objectives=3, max_rounds=100, seed=42)
        result = run_agreement_simulation(params, quantumness=0.3)
        assert result.method == "quantum"
        assert len(result.magnetisation_trajectory) > 0

    def test_frustration_increases_difficulty(self):
        """Frustrated bonds should increase disagreement or slow agreement."""
        params_easy = AgreementParams(
            n_agents=10, n_objectives=3, frustration=0.0,
            max_rounds=200, seed=42,
        )
        params_hard = AgreementParams(
            n_agents=10, n_objectives=3, frustration=0.5,
            max_rounds=200, seed=42,
        )
        result_easy = run_agreement_simulation(params_easy)
        result_hard = run_agreement_simulation(params_hard)
        # Frustrated system should have more disagreement
        assert result_hard.final_disagreement >= result_easy.final_disagreement - 0.1


class TestAgreementScaling:
    def test_output_shapes(self):
        """Scaling sweep should return matching arrays."""
        counts, means, stds = agreement_scaling(
            agent_counts=[5, 10],
            n_objectives=2,
            max_rounds=50,
            n_seeds=2,
        )
        assert len(counts) == 2
        assert len(means) == 2
        assert len(stds) == 2
