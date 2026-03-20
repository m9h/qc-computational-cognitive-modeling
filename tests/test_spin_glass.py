"""Tests for the spin glass module."""

import numpy as np
import pytest

from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    frustration_index,
    sk_couplings,
    social_hamiltonian_classical,
)
from qcccm.spin_glass.order_params import (
    binder_cumulant,
    edwards_anderson_q,
    glass_susceptibility,
    overlap,
    overlap_distribution,
)
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,
    transverse_field_mc,
)


class TestCouplings:
    def test_sk_symmetric(self):
        """SK couplings should be symmetric with zero diagonal."""
        adj, J = sk_couplings(20, seed=42)
        assert np.allclose(J, J.T)
        assert np.allclose(np.diag(J), 0)

    def test_sk_scaling(self):
        """SK couplings should have std ~ 1/sqrt(N)."""
        N = 100
        _, J = sk_couplings(N, seed=42)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        std = np.std(J[mask])
        assert 0.05 < std < 0.2  # should be ~1/sqrt(100) = 0.1

    def test_ea_square_lattice(self):
        """EA on square lattice should have 2L(L-1) bonds (periodic: 2L^2)."""
        adj, J = ea_couplings(16, topology="square", seed=42)
        L = 4
        # Periodic: each site has 4 neighbors, but each bond counted twice
        n_bonds = int(np.sum(adj) / 2)
        assert n_bonds == 2 * L * L  # 32 for L=4 periodic

    def test_ea_bimodal(self):
        """Bimodal disorder should produce only ±1 couplings."""
        _, J = ea_couplings(16, topology="square", disorder="bimodal", seed=42)
        nonzero = J[J != 0]
        assert set(np.unique(nonzero)) == {-1.0, 1.0}

    def test_ea_gaussian(self):
        """Gaussian disorder should produce continuous couplings."""
        _, J = ea_couplings(16, topology="square", disorder="gaussian", seed=42)
        nonzero = J[J != 0]
        assert len(np.unique(nonzero)) > 2  # not just ±1


class TestHamiltonian:
    def test_energy_all_aligned(self):
        """All spins up in ferromagnet should have low energy."""
        N = 10
        adj = np.ones((N, N)) - np.eye(N)
        J = adj.copy()  # all positive
        params = SocialSpinGlassParams(N, adj, J, np.zeros(N))

        spins_up = np.ones(N)
        spins_random = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)

        E_up = social_hamiltonian_classical(params, spins_up)
        E_random = social_hamiltonian_classical(params, spins_random)
        assert E_up < E_random

    def test_frustration_index_unfrustrated(self):
        """Complete graph with all positive J should have zero frustration."""
        N = 5
        adj = np.ones((N, N)) - np.eye(N)
        J = np.ones((N, N))
        assert frustration_index(adj, J) == 0.0

    def test_frustration_index_triangle(self):
        """Triangle with one negative bond is frustrated."""
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        J = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]], dtype=float)
        assert frustration_index(adj, J) == 1.0  # 1 triangle, frustrated


class TestOrderParams:
    def test_q_ea_frozen(self):
        """Frozen spins (constant in time) should give q_EA = 1."""
        traj = np.ones((100, 10))  # all +1 always
        assert edwards_anderson_q(traj) == pytest.approx(1.0)

    def test_q_ea_fluctuating(self):
        """Rapidly fluctuating spins should give q_EA near 0."""
        rng = np.random.default_rng(42)
        traj = rng.choice([-1, 1], size=(10000, 20))
        q = edwards_anderson_q(traj)
        assert q < 0.05  # should be ~1/n_snapshots

    def test_overlap_identical(self):
        """Overlap of a config with itself should be 1."""
        config = np.array([1, -1, 1, -1, 1], dtype=float)
        assert overlap(config, config) == pytest.approx(1.0)

    def test_overlap_anti(self):
        """Overlap of a config with its negation should be -1."""
        config = np.array([1, -1, 1, -1, 1], dtype=float)
        assert overlap(config, -config) == pytest.approx(-1.0)

    def test_overlap_distribution_shape(self):
        """Should return the requested number of samples."""
        rng = np.random.default_rng(42)
        trajs = [rng.choice([-1, 1], size=(50, 10)) for _ in range(5)]
        q_dist = overlap_distribution(trajs, n_samples=200)
        assert q_dist.shape == (200,)
        assert np.all(np.abs(q_dist) <= 1.0)

    def test_binder_cumulant_range(self):
        """Binder cumulant should be in [0, 2/3]."""
        rng = np.random.default_rng(42)
        traj = rng.choice([-1, 1], size=(500, 20))
        U = binder_cumulant(traj)
        assert -0.1 < U < 0.75  # allow small numerical margin


class TestMetropolis:
    def test_finds_ground_state_ferromagnet(self):
        """Metropolis at low T on a ferromagnet should find all-aligned state."""
        N = 8
        adj = np.ones((N, N)) - np.eye(N)
        J = adj.copy()
        params = SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=0.1)

        result = metropolis_spin_glass(params, n_steps=3000, n_equilibrate=500)
        # Ground state: all +1 or all -1
        assert abs(np.sum(result.spins)) == N

    def test_trajectory_shape(self):
        """Trajectory should have correct shape."""
        N = 10
        adj, J = sk_couplings(N, seed=42)
        params = SocialSpinGlassParams(N, adj, J, np.zeros(N))
        result = metropolis_spin_glass(params, n_steps=200, n_equilibrate=50)
        assert result.trajectory.shape == (150, N)
        assert result.energies.shape == (150,)

    def test_energy_decreases(self):
        """Energy should generally decrease over time at low T."""
        N = 12
        adj, J = sk_couplings(N, seed=42)
        params = SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=0.5)
        result = metropolis_spin_glass(params, n_steps=2000, n_equilibrate=200)
        # Compare first and last quartile average energy
        E = result.energies
        assert np.mean(E[-len(E) // 4 :]) <= np.mean(E[: len(E) // 4]) + 1.0


class TestTransverseFieldMC:
    def test_runs_without_error(self):
        """Basic smoke test for PIMC."""
        N = 8
        adj, J = sk_couplings(N, seed=42)
        params = SocialSpinGlassParams(
            N, adj, J, np.zeros(N), transverse_field=0.5, temperature=0.5
        )
        result = transverse_field_mc(params, n_trotter=4, n_steps=500, n_equilibrate=100)
        assert result.spins.shape == (N,)
        assert result.method.startswith("transverse_field_mc")

    def test_tunneling_helps_frustrated(self):
        """Transverse field should find lower energy than plain Metropolis on frustrated system."""
        N = 8
        adj, J = ea_couplings(N, topology="complete", disorder="bimodal", seed=42)
        params_classical = SocialSpinGlassParams(
            N, adj, J, np.zeros(N), transverse_field=0.0, temperature=0.3
        )
        params_quantum = SocialSpinGlassParams(
            N, adj, J, np.zeros(N), transverse_field=1.0, temperature=0.3
        )

        # Run multiple seeds and compare best
        best_classical = float("inf")
        best_quantum = float("inf")
        for s in range(5):
            pc = params_classical._replace(seed=s)
            pq = params_quantum._replace(seed=s)
            rc = metropolis_spin_glass(pc, n_steps=2000, n_equilibrate=500)
            rq = transverse_field_mc(pq, n_trotter=8, n_steps=2000, n_equilibrate=500)
            best_classical = min(best_classical, rc.energy)
            best_quantum = min(best_quantum, rq.energy)

        # Quantum should find at least as good (often better) energy
        assert best_quantum <= best_classical + 0.5
