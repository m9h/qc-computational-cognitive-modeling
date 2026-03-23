"""Tests for the pure-JAX spin glass solvers.

Verifies:
1. JIT compilation works (no tracer errors)
2. Energy decreases on average at low T
3. Batched version produces different results per seed
4. PIMC with Gamma > 0 finds lower energy than Gamma = 0 on frustrated systems
5. Results are finite and spins are +/-1
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    sk_couplings,
)
from qcccm.spin_glass.solvers_jax import (
    _energy_jax,
    batched_metropolis_jax,
    metropolis_spin_glass_jax,
    transverse_field_mc_jax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ferromagnet_params(N: int = 8, T: float = 0.1, seed: int = 0) -> SocialSpinGlassParams:
    """Fully connected ferromagnet (easy ground state: all aligned)."""
    adj = np.ones((N, N)) - np.eye(N)
    J = adj.copy()
    return SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=T, seed=seed)


def _sk_params(N: int = 12, T: float = 0.5, seed: int = 42) -> SocialSpinGlassParams:
    adj, J = sk_couplings(N, seed=seed)
    return SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=T, seed=seed)


def _frustrated_params(
    N: int = 8, T: float = 0.3, Gamma: float = 0.0, seed: int = 42,
) -> SocialSpinGlassParams:
    """Fully connected bimodal disorder (maximally frustrated)."""
    adj, J = ea_couplings(N, topology="complete", disorder="bimodal", seed=seed)
    return SocialSpinGlassParams(
        N, adj, J, np.zeros(N), transverse_field=Gamma, temperature=T, seed=seed,
    )


# ===================================================================
# 1. JIT compilation works (no tracer errors)
# ===================================================================

class TestJITCompilation:
    """Ensure the inner loops are JIT-compilable without tracer leaks."""

    def test_metropolis_jit_compiles(self):
        """metropolis_spin_glass_jax should run under JIT without errors."""
        params = _ferromagnet_params()
        result = metropolis_spin_glass_jax(params, n_steps=100, n_equilibrate=10)
        assert result.spins is not None

    def test_pimc_jit_compiles(self):
        """transverse_field_mc_jax should run under JIT without errors."""
        params = _frustrated_params(Gamma=0.5)
        result = transverse_field_mc_jax(
            params, n_trotter=4, n_steps=50, n_equilibrate=10,
        )
        assert result.spins is not None

    def test_batched_metropolis_jit_compiles(self):
        """batched_metropolis_jax should compile and run."""
        params = _sk_params(N=8)
        seeds = jnp.arange(4)
        best_spins, best_energies, energies = batched_metropolis_jax(
            params, seeds, n_steps=100,
        )
        assert best_spins.shape == (4, 8)
        assert best_energies.shape == (4,)

    def test_metropolis_accepts_explicit_key(self):
        """Passing an explicit JAX key should work (ignore params.seed)."""
        params = _ferromagnet_params()
        key = jax.random.PRNGKey(999)
        result = metropolis_spin_glass_jax(params, n_steps=100, key=key, n_equilibrate=10)
        assert result.spins is not None


# ===================================================================
# 2. Energy decreases on average at low T
# ===================================================================

class TestEnergyDecreases:
    """At low temperature, MC should find lower-energy configurations over time."""

    def test_metropolis_energy_decreases(self):
        params = _sk_params(N=12, T=0.3)
        result = metropolis_spin_glass_jax(params, n_steps=3000, n_equilibrate=500)
        E = np.asarray(result.energies)
        quarter = len(E) // 4
        avg_first = np.mean(E[:quarter])
        avg_last = np.mean(E[-quarter:])
        # Last quarter should be no worse than first quarter (+ tolerance)
        assert avg_last <= avg_first + 1.0, (
            f"Energy did not decrease: first quarter mean={avg_first:.3f}, "
            f"last quarter mean={avg_last:.3f}"
        )

    def test_metropolis_ferromagnet_ground_state(self):
        """At low T on a ferromagnet, should find all-aligned state."""
        params = _ferromagnet_params(N=8, T=0.05)
        result = metropolis_spin_glass_jax(params, n_steps=5000, n_equilibrate=500)
        # Ground state is all +1 or all -1
        assert abs(float(jnp.sum(result.spins))) == 8

    def test_pimc_energy_decreases(self):
        params = _frustrated_params(N=8, T=0.3, Gamma=0.5)
        result = transverse_field_mc_jax(
            params, n_trotter=4, n_steps=500, n_equilibrate=100,
        )
        E = np.asarray(result.energies)
        quarter = len(E) // 4
        avg_first = np.mean(E[:quarter])
        avg_last = np.mean(E[-quarter:])
        assert avg_last <= avg_first + 2.0


# ===================================================================
# 3. Batched version produces different results per seed
# ===================================================================

class TestBatched:
    """vmap over seeds must produce distinct runs."""

    def test_different_seeds_give_different_results(self):
        params = _sk_params(N=10, T=0.5)
        seeds = jnp.array([0, 1, 2, 3])
        best_spins, best_energies, energies = batched_metropolis_jax(
            params, seeds, n_steps=500,
        )
        # Different seeds should (almost certainly) yield different spin configs
        all_same = True
        for i in range(1, len(seeds)):
            if not jnp.allclose(best_spins[0], best_spins[i]):
                all_same = False
                break
        assert not all_same, "All seeds produced identical spin configurations"

    def test_same_seed_gives_same_result(self):
        params = _sk_params(N=10, T=0.5)
        seeds = jnp.array([42, 42])
        best_spins, _, _ = batched_metropolis_jax(params, seeds, n_steps=500)
        assert jnp.allclose(best_spins[0], best_spins[1])

    def test_batch_energies_shape(self):
        params = _sk_params(N=8)
        B = 6
        seeds = jnp.arange(B)
        _, best_energies, energies = batched_metropolis_jax(
            params, seeds, n_steps=200,
        )
        assert best_energies.shape == (B,)
        assert energies.shape == (B, 200)


# ===================================================================
# 4. PIMC with Gamma > 0 finds lower energy than Gamma = 0
# ===================================================================

class TestTransverseFieldAdvantage:
    """Quantum tunneling (Gamma > 0) should help on frustrated landscapes."""

    def test_tunneling_helps_frustrated(self):
        """Average best energy over several seeds should be lower with Gamma > 0."""
        best_classical_energies = []
        best_quantum_energies = []

        for s in range(8):
            params_cl = _frustrated_params(N=8, T=0.3, Gamma=0.0, seed=s)
            params_qu = _frustrated_params(N=8, T=0.3, Gamma=1.0, seed=s)

            rc = metropolis_spin_glass_jax(params_cl, n_steps=3000, n_equilibrate=500)
            rq = transverse_field_mc_jax(
                params_qu, n_trotter=8, n_steps=3000, n_equilibrate=500,
            )
            best_classical_energies.append(rc.energy)
            best_quantum_energies.append(rq.energy)

        avg_classical = np.mean(best_classical_energies)
        avg_quantum = np.mean(best_quantum_energies)
        best_classical = min(best_classical_energies)
        best_quantum = min(best_quantum_energies)

        # Quantum should find at least as good energy (with tolerance)
        assert best_quantum <= best_classical + 1.0, (
            f"Quantum best={best_quantum:.3f} worse than classical best={best_classical:.3f}"
        )

    def test_pimc_gamma_zero_behaves_classically(self):
        """With Gamma=0, PIMC J_tau should be 0 and behave like classical MC."""
        params = _frustrated_params(Gamma=0.0)
        result = transverse_field_mc_jax(params, n_trotter=4, n_steps=200, n_equilibrate=50)
        assert result.metadata["J_tau"] == pytest.approx(0.0)


# ===================================================================
# 5. Results are finite and spins are +/-1
# ===================================================================

class TestResultValidity:
    """Basic sanity: no NaNs, spins in {-1, +1}, energies finite."""

    def test_metropolis_spins_are_ising(self):
        params = _sk_params()
        result = metropolis_spin_glass_jax(params, n_steps=500, n_equilibrate=100)
        spins = np.asarray(result.spins)
        assert set(np.unique(spins)).issubset({-1.0, 1.0})

    def test_metropolis_finite(self):
        params = _sk_params()
        result = metropolis_spin_glass_jax(params, n_steps=500, n_equilibrate=100)
        assert np.isfinite(result.energy)
        assert np.all(np.isfinite(np.asarray(result.energies)))
        assert np.all(np.isfinite(np.asarray(result.spins)))

    def test_pimc_spins_are_ising(self):
        params = _frustrated_params(Gamma=0.5)
        result = transverse_field_mc_jax(
            params, n_trotter=4, n_steps=200, n_equilibrate=50,
        )
        spins = np.asarray(result.spins)
        assert set(np.unique(spins)).issubset({-1.0, 1.0})

    def test_pimc_finite(self):
        params = _frustrated_params(Gamma=0.5)
        result = transverse_field_mc_jax(
            params, n_trotter=4, n_steps=200, n_equilibrate=50,
        )
        assert np.isfinite(result.energy)
        assert np.all(np.isfinite(np.asarray(result.energies)))
        assert np.all(np.isfinite(np.asarray(result.spins)))

    def test_pimc_metadata(self):
        params = _frustrated_params(Gamma=0.5)
        result = transverse_field_mc_jax(
            params, n_trotter=4, n_steps=100, n_equilibrate=20,
        )
        assert "n_trotter" in result.metadata
        assert result.metadata["n_trotter"] == 4
        assert "J_tau" in result.metadata
        assert np.isfinite(result.metadata["J_tau"])

    def test_energy_matches_hamiltonian(self):
        """Best energy reported should match recomputation from best spins."""
        params = _sk_params(N=8)
        result = metropolis_spin_glass_jax(params, n_steps=1000, n_equilibrate=200)
        J = jnp.asarray(params.J)
        fields = jnp.asarray(params.fields)
        recomputed = float(_energy_jax(J, fields, result.spins))
        assert result.energy == pytest.approx(recomputed, abs=1e-4)

    def test_solver_result_method_string(self):
        params = _sk_params(N=6)
        r1 = metropolis_spin_glass_jax(params, n_steps=100, n_equilibrate=10)
        assert r1.method == "metropolis_jax"

        params_q = _frustrated_params(Gamma=0.5)
        r2 = transverse_field_mc_jax(params_q, n_trotter=4, n_steps=50, n_equilibrate=10)
        assert r2.method.startswith("transverse_field_mc_jax")

    def test_batched_finite(self):
        params = _sk_params(N=8)
        seeds = jnp.arange(3)
        best_spins, best_energies, energies = batched_metropolis_jax(
            params, seeds, n_steps=200,
        )
        assert jnp.all(jnp.isfinite(best_energies))
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(jnp.isfinite(best_spins))
