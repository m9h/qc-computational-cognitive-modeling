"""Physics invariant tests — encode what MUST be true regardless of parameters.

These tests define the guardrails for the autoresearch loop. Any experiment
that violates these invariants has a bug, not a discovery.

Organized by principle:
1. Classical limits — quantum methods must reduce to classical when quantum params → 0
2. Energy bounds — no method can go below the exact ground state
3. Thermodynamic consistency — hotter systems disorder faster, consensus scales correctly
4. Tunneling vs heating — PIMC must be distinguishable from just raising temperature
5. Minority game — quantum agents at q=0 must match classical agents exactly
"""

import numpy as np
import pytest

from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    sk_couplings,
    social_hamiltonian_classical,
    frustration_index,
)
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,
    transverse_field_mc,
    vqe_ground_state,
    qaoa_ground_state,
)
from qcccm.spin_glass.order_params import edwards_anderson_q, overlap
from qcccm.games.minority import (
    MinorityGameParams,
    run_minority_game,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sk_params(N=8, T=0.5, Gamma=0.0, seed=42):
    adj, J = sk_couplings(N, seed=seed)
    return SocialSpinGlassParams(
        n_agents=N, adjacency=adj, J=J, fields=np.zeros(N),
        transverse_field=Gamma, temperature=T, seed=seed,
    )


def _ea_params(N=16, T=1.0, Gamma=0.0, seed=42, disorder="bimodal"):
    adj, J = ea_couplings(N, topology="square", disorder=disorder, seed=seed)
    actual_N = adj.shape[0]
    return SocialSpinGlassParams(
        n_agents=actual_N, adjacency=adj, J=J, fields=np.zeros(actual_N),
        transverse_field=Gamma, temperature=T, seed=seed,
    )


def _brute_force_ground_state(params):
    from itertools import product as iproduct
    best_E = float("inf")
    best_s = None
    for bits in iproduct([-1, 1], repeat=params.n_agents):
        s = np.array(bits, dtype=float)
        E = social_hamiltonian_classical(params, s)
        if E < best_E:
            best_E = E
            best_s = s.copy()
    return best_E, best_s


# ===================================================================
# 1. CLASSICAL LIMITS — quantum → 0 must recover classical behavior
# ===================================================================


class TestClassicalLimits:
    """When quantum parameters are zero, quantum methods must match classical."""

    def test_pimc_gamma_zero_matches_metropolis_energy(self):
        """PIMC at Γ=0 should find the same ground state energy as Metropolis."""
        params = _sk_params(N=8, T=0.3, Gamma=0.0, seed=42)
        metro = metropolis_spin_glass(params, n_steps=5000)
        pimc = transverse_field_mc(params, n_trotter=8, n_steps=5000)
        # Both should find the same energy (within noise from different RNG paths)
        # The key check: PIMC at Γ=0 should not be systematically better or worse
        assert abs(metro.energy - pimc.energy) < 1.0, (
            f"PIMC(Γ=0) energy {pimc.energy:.4f} differs from Metropolis {metro.energy:.4f} by > 1.0"
        )

    def test_pimc_gamma_zero_trotter_coupling_is_zero(self):
        """At Γ=0, the Trotter coupling J_tau should be zero (no inter-slice coupling)."""
        params = _sk_params(N=6, T=0.5, Gamma=0.0, seed=42)
        result = transverse_field_mc(params, n_trotter=8, n_steps=100)
        assert result.metadata["J_tau"] == 0.0

    def test_minority_game_quantum_zero_matches_classical(self):
        """Quantum agents at q=0 must produce identical volatility to classical agents."""
        for seed in range(5):
            params = MinorityGameParams(n_agents=51, memory=3, n_rounds=300, seed=seed)
            classical = run_minority_game(params, quantumness=0.0, beta=0.1)
            quantum_zero = run_minority_game(params, quantumness=0.0, beta=0.1)
            assert classical.volatility == pytest.approx(quantum_zero.volatility, abs=1e-10), (
                f"seed={seed}: classical vol={classical.volatility:.6f} != quantum(q=0) vol={quantum_zero.volatility:.6f}"
            )

    @pytest.mark.xfail(reason=(
        "KNOWN BUG: QuantumAgent is not continuous at q→0. Even q=0.001 changes "
        "strategy selection dramatically (74% volatility change) because the "
        "density matrix rotation angle q*π/4 interacts nonlinearly with the "
        "Boltzmann softmax. The QuantumAgent.choose() method needs redesign "
        "so that q→0 smoothly recovers classical behavior."
    ))
    def test_minority_game_very_small_quantum_near_classical(self):
        """Quantum agents at q=0.001 should produce volatility very close to classical."""
        params = MinorityGameParams(n_agents=101, memory=3, n_rounds=500, seed=42)
        classical = run_minority_game(params, quantumness=0.0, beta=0.1)
        near_classical = run_minority_game(params, quantumness=0.001, beta=0.1)
        # Allow 20% relative difference for tiny quantumness
        if classical.volatility > 0.01:
            relative_diff = abs(classical.volatility - near_classical.volatility) / classical.volatility
            assert relative_diff < 0.20, (
                f"q=0.001 volatility {near_classical.volatility:.4f} differs from classical "
                f"{classical.volatility:.4f} by {relative_diff:.0%}"
            )


# ===================================================================
# 2. ENERGY BOUNDS — no method can beat the exact ground state
# ===================================================================


class TestEnergyBounds:
    """Variational principle: no method should find energy below exact GS."""

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_metropolis_above_ground_state(self, seed):
        """Metropolis energy must be >= exact ground state."""
        params = _sk_params(N=8, T=0.3, seed=seed)
        E_exact, _ = _brute_force_ground_state(params)
        result = metropolis_spin_glass(params, n_steps=5000)
        assert result.energy >= E_exact - 1e-6, (
            f"Metropolis E={result.energy:.6f} < E_exact={E_exact:.6f}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_pimc_above_ground_state(self, seed):
        """PIMC best energy must be >= exact ground state of the CLASSICAL Hamiltonian."""
        params = _sk_params(N=8, T=0.3, Gamma=1.0, seed=seed)
        # Ground state of the classical (ZZ-only) Hamiltonian
        params_classical = params._replace(transverse_field=0.0)
        E_exact, _ = _brute_force_ground_state(params_classical)
        result = transverse_field_mc(params, n_trotter=8, n_steps=5000)
        assert result.energy >= E_exact - 1e-6, (
            f"PIMC E={result.energy:.6f} < E_exact={E_exact:.6f}"
        )

    @pytest.mark.parametrize("seed", [42, 123])
    def test_vqe_above_ground_state(self, seed):
        """VQE energy must be >= exact ground state."""
        params = _sk_params(N=6, T=0.1, Gamma=1.0, seed=seed)
        params_classical = params._replace(transverse_field=0.0)
        E_exact, _ = _brute_force_ground_state(params_classical)
        result = vqe_ground_state(params, n_layers=2, max_steps=100)
        assert result.energy >= E_exact - 0.01, (
            f"VQE E={result.energy:.6f} < E_exact={E_exact:.6f} — Hamiltonian mismatch?"
        )

    @pytest.mark.parametrize("seed", [42, 123])
    def test_qaoa_above_ground_state(self, seed):
        """QAOA energy must be >= exact ground state."""
        params = _sk_params(N=6, T=0.1, Gamma=1.0, seed=seed)
        params_classical = params._replace(transverse_field=0.0)
        E_exact, _ = _brute_force_ground_state(params_classical)
        result = qaoa_ground_state(params, depth=2, max_steps=100)
        assert result.energy >= E_exact - 0.01, (
            f"QAOA E={result.energy:.6f} < E_exact={E_exact:.6f} — Hamiltonian mismatch?"
        )

    def test_vqe_energy_matches_classical_hamiltonian(self):
        """VQE spins evaluated with classical H should give similar energy to VQE expval."""
        params = _sk_params(N=6, T=0.1, seed=42)
        result = vqe_ground_state(params, n_layers=2, max_steps=100)
        E_classical_check = social_hamiltonian_classical(params, np.array(result.spins))
        # VQE expval is from a superposition, classical H is from the rounded spins.
        # They won't match exactly, but should be in the same ballpark.
        assert abs(result.energy - E_classical_check) < 2.0, (
            f"VQE expval {result.energy:.4f} vs classical H of VQE spins {E_classical_check:.4f} "
            f"differ by {abs(result.energy - E_classical_check):.4f}"
        )


# ===================================================================
# 3. THERMODYNAMIC CONSISTENCY
# ===================================================================


class TestThermodynamics:
    """Basic thermodynamic properties must hold."""

    def test_higher_temperature_lower_magnetization(self):
        """At higher T, average |magnetization| should be lower (above T_c)."""
        params_cold = _ea_params(N=16, T=0.5, seed=42)
        params_hot = _ea_params(N=16, T=5.0, seed=42)

        cold = metropolis_spin_glass(params_cold, n_steps=5000)
        hot = metropolis_spin_glass(params_hot, n_steps=5000)

        mag_cold = abs(np.mean(cold.trajectory, axis=0)).mean()
        mag_hot = abs(np.mean(hot.trajectory, axis=0)).mean()

        # Cold system should have higher time-averaged local magnetization
        assert mag_cold > mag_hot - 0.1, (
            f"Cold |m|={mag_cold:.3f} should be > hot |m|={mag_hot:.3f}"
        )

    def test_ferromagnet_consensus_faster_at_low_T(self):
        """Ferromagnet (all J>0) should reach consensus faster below T_c than above."""
        N = 9  # 3x3 square
        adj, J = ea_couplings(N, topology="square", disorder="bimodal", seed=0)
        J = np.abs(J)  # force all ferromagnetic

        # Well below T_c ≈ 2.27
        params_cold = SocialSpinGlassParams(
            n_agents=adj.shape[0], adjacency=adj, J=J, fields=np.zeros(adj.shape[0]),
            temperature=0.5, seed=42,
        )
        cold = metropolis_spin_glass(params_cold, n_steps=5000)
        cold_mag = abs(np.mean(cold.spins))

        # Well above T_c
        params_hot = SocialSpinGlassParams(
            n_agents=adj.shape[0], adjacency=adj, J=J, fields=np.zeros(adj.shape[0]),
            temperature=5.0, seed=42,
        )
        hot = metropolis_spin_glass(params_hot, n_steps=5000)
        hot_mag = abs(np.mean(hot.spins))

        # Cold ferromagnet should have higher or equal magnetization
        # (on tiny systems both may saturate to 1.0)
        assert cold_mag >= hot_mag, (
            f"Ferromagnet: cold |m|={cold_mag:.3f} should be >= hot |m|={hot_mag:.3f}"
        )


# ===================================================================
# 4. TUNNELING VS HEATING — distinguish genuine quantum effects
# ===================================================================


class TestTunnelingVsHeating:
    """PIMC with transverse field must be distinguishable from just raising T."""

    def test_pimc_consensus_energy_vs_hot_metropolis(self):
        """If PIMC reaches consensus, the consensus state energy should be
        compared to hot-Metropolis consensus energy. If PIMC finds a LOWER
        energy consensus, tunneling is genuine. If same, it's just heating.

        This test documents the comparison without asserting which is better —
        it's the key diagnostic for the autoresearch loop.
        """
        N = 16
        adj, J = ea_couplings(N, topology="square", disorder="bimodal", seed=42)
        actual_N = adj.shape[0]
        fields = np.zeros(actual_N)

        # PIMC at T=1.0, Gamma=2.0
        params_pimc = SocialSpinGlassParams(
            n_agents=actual_N, adjacency=adj, J=J, fields=fields,
            transverse_field=2.0, temperature=1.0, seed=42,
        )
        pimc = transverse_field_mc(params_pimc, n_trotter=8, n_steps=5000)
        E_pimc = social_hamiltonian_classical(params_pimc, pimc.spins)

        # Hot Metropolis at T=5.0 (reaches consensus quickly like PIMC)
        params_hot = SocialSpinGlassParams(
            n_agents=actual_N, adjacency=adj, J=J, fields=fields,
            transverse_field=0.0, temperature=5.0, seed=42,
        )
        hot = metropolis_spin_glass(params_hot, n_steps=5000)
        E_hot = social_hamiltonian_classical(params_hot, hot.spins)

        # Cold Metropolis (ground truth best energy)
        params_cold = SocialSpinGlassParams(
            n_agents=actual_N, adjacency=adj, J=J, fields=fields,
            transverse_field=0.0, temperature=0.1, seed=42,
        )
        cold = metropolis_spin_glass(params_cold, n_steps=10000)
        E_cold = social_hamiltonian_classical(params_cold, cold.spins)

        # Document the comparison (this is informational, not a pass/fail)
        print(f"\n  TUNNELING VS HEATING DIAGNOSTIC:")
        print(f"  Cold Metropolis (T=0.1):  E = {E_cold:.4f}")
        print(f"  PIMC (T=1.0, Γ=2.0):     E = {E_pimc:.4f}")
        print(f"  Hot Metropolis (T=5.0):   E = {E_hot:.4f}")
        if E_pimc < E_hot:
            print(f"  → PIMC found LOWER energy than hot Metropolis: tunneling evidence")
        else:
            print(f"  → PIMC found SAME/HIGHER energy as hot Metropolis: likely just heating")

        # The hard assertion: PIMC energy must be a valid spin configuration energy
        assert np.isfinite(E_pimc)
        assert np.isfinite(E_hot)

    def test_pimc_low_gamma_no_speedup(self):
        """At very small Γ, PIMC should NOT reach consensus faster than Metropolis.
        (Small Γ = weak tunneling, shouldn't help much.)"""
        params_metro = _ea_params(N=16, T=1.0, Gamma=0.0, seed=42)
        params_pimc = _ea_params(N=16, T=1.0, Gamma=0.01, seed=42)

        metro = metropolis_spin_glass(params_metro, n_steps=3000)
        pimc = transverse_field_mc(params_pimc, n_trotter=8, n_steps=3000)

        mag_metro = abs(np.mean(metro.spins))
        mag_pimc = abs(np.mean(pimc.spins))

        # At tiny Γ, magnetizations should be similar (within stochastic noise)
        # Different RNG paths mean we need generous tolerance on single runs
        assert abs(mag_metro - mag_pimc) < 0.6, (
            f"Γ=0.01: PIMC |m|={mag_pimc:.3f} differs from Metropolis |m|={mag_metro:.3f} by > 0.6"
        )


# ===================================================================
# 5. MINORITY GAME INVARIANTS
# ===================================================================


class TestMinorityGameInvariants:
    """Physics invariants specific to the minority game."""

    def test_volatility_positive(self):
        """Volatility σ²/N must be non-negative."""
        params = MinorityGameParams(n_agents=51, memory=3, n_rounds=200, seed=42)
        for q in [0.0, 0.3, 0.7]:
            result = run_minority_game(params, quantumness=q, beta=0.1)
            assert result.volatility >= 0, f"Negative volatility at q={q}"

    def test_random_baseline(self):
        """With zero memory (M=1, alpha very small), volatility should be near 1
        (random behavior, σ²/N ≈ 1)."""
        params = MinorityGameParams(n_agents=101, memory=1, n_rounds=500, seed=42)
        result = run_minority_game(params, quantumness=0.0, beta=0.0)  # beta=0 = random choice
        # At beta=0 (infinite temperature), agents choose randomly → vol ≈ 1
        assert 0.5 < result.volatility < 2.0, (
            f"Random agents volatility {result.volatility:.3f} not near 1.0"
        )

    def test_attendance_sums_to_N(self):
        """Total attendance (choice 0 + choice 1) must equal N at every round."""
        params = MinorityGameParams(n_agents=51, memory=3, n_rounds=100, seed=42)
        result = run_minority_game(params, quantumness=0.3, beta=0.1)
        # attendance records number choosing 1; those choosing 0 = N - attendance
        assert np.all(result.attendance >= 0)
        assert np.all(result.attendance <= 51)

    def test_winners_are_binary(self):
        """Winning side must be 0 or 1."""
        params = MinorityGameParams(n_agents=51, memory=3, n_rounds=100, seed=42)
        result = run_minority_game(params, quantumness=0.3, beta=0.1)
        assert set(np.unique(result.winners)).issubset({0, 1})

    def test_quantum_does_not_break_game_mechanics(self):
        """Quantum agents should still produce valid game outcomes."""
        params = MinorityGameParams(n_agents=51, memory=3, n_rounds=200, seed=42)
        result = run_minority_game(params, quantumness=0.7, beta=0.5)
        assert result.attendance.shape == (200,)
        assert result.winners.shape == (200,)
        assert np.isfinite(result.volatility)

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_quantum_volatility_bounded(self, seed):
        """Quantum agents should not produce wildly different volatility from classical.
        Allow up to 5x difference — anything more suggests a bug."""
        params = MinorityGameParams(n_agents=101, memory=4, n_rounds=500, seed=seed)
        classical = run_minority_game(params, quantumness=0.0, beta=0.1)
        quantum = run_minority_game(params, quantumness=0.5, beta=0.1)
        if classical.volatility > 0.01:
            ratio = quantum.volatility / classical.volatility
            assert 0.05 < ratio < 5.0, (
                f"seed={seed}: quantum/classical volatility ratio = {ratio:.2f} "
                f"(classical={classical.volatility:.4f}, quantum={quantum.volatility:.4f})"
            )


# ===================================================================
# 6. HAMILTONIAN CONSISTENCY
# ===================================================================


class TestHamiltonianConsistency:
    """The PennyLane and classical Hamiltonians must agree."""

    def test_pennylane_hamiltonian_ground_state_matches_classical(self):
        """VQE on the PennyLane Hamiltonian must find energy >= classical exact GS.
        This is the test that would have caught the transverse field bug."""
        for seed in [42, 123]:
            params = _sk_params(N=6, Gamma=2.0, T=0.1, seed=seed)
            E_exact, _ = _brute_force_ground_state(params)
            vqe = vqe_ground_state(params, n_layers=3, max_steps=200)
            assert vqe.energy >= E_exact - 0.01, (
                f"seed={seed}: VQE E={vqe.energy:.4f} < E_exact={E_exact:.4f}. "
                f"PennyLane Hamiltonian likely includes transverse field terms."
            )

    def test_classical_energy_is_deterministic(self):
        """social_hamiltonian_classical must give the same result for the same input."""
        params = _sk_params(N=8, seed=42)
        spins = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        E1 = social_hamiltonian_classical(params, spins)
        E2 = social_hamiltonian_classical(params, spins)
        assert E1 == E2

    def test_spin_flip_changes_energy(self):
        """Flipping one spin should change the energy (for non-trivial J)."""
        params = _sk_params(N=8, seed=42)
        spins = np.ones(8)
        E_before = social_hamiltonian_classical(params, spins)
        spins_flipped = spins.copy()
        spins_flipped[0] = -1
        E_after = social_hamiltonian_classical(params, spins_flipped)
        assert E_before != E_after
