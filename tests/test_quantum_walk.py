"""Tests for quantum walk module."""

import jax.numpy as jnp
import pytest

from qcccm.core.quantum_walk import (
    QuantumWalkParams,
    biased_coin,
    classical_vs_quantum_spreading,
    hadamard_coin,
    quantum_walk_evolution,
    quantum_walk_fpt_density,
    shift_operator,
)


class TestCoinOperators:
    def test_hadamard_unitary(self):
        """Hadamard coin should be unitary: H†H = I."""
        H = hadamard_coin()
        product = jnp.conj(H).T @ H
        assert jnp.allclose(product, jnp.eye(2, dtype=jnp.complex64), atol=1e-5)

    def test_biased_coin_unitary(self):
        """Biased coin should be unitary for any angle."""
        for theta in [0.0, jnp.pi / 6, jnp.pi / 4, jnp.pi / 3]:
            C = biased_coin(theta)
            product = jnp.conj(C).T @ C
            assert jnp.allclose(product, jnp.eye(2, dtype=jnp.complex64), atol=1e-5)

    def test_hadamard_is_biased_pi4(self):
        """biased_coin(π/4) should equal hadamard_coin()."""
        H = hadamard_coin()
        C = biased_coin(jnp.pi / 4)
        assert jnp.allclose(H, C, atol=1e-5)


class TestShiftOperator:
    def test_shift_inverse(self):
        """S_left and S_right should be inverses (on periodic lattice)."""
        N = 10
        S_left, S_right = shift_operator(N)
        product = S_left @ S_right
        assert jnp.allclose(product, jnp.eye(N, dtype=jnp.complex64), atol=1e-5)

    def test_shift_action(self):
        """S_right should shift a basis vector one position to the right."""
        N = 5
        _, S_right = shift_operator(N)
        e2 = jnp.zeros(N, dtype=jnp.complex64).at[2].set(1.0)
        shifted = S_right @ e2
        expected = jnp.zeros(N, dtype=jnp.complex64).at[3].set(1.0)
        assert jnp.allclose(shifted, expected, atol=1e-5)


class TestQuantumWalkEvolution:
    def test_probability_normalization(self):
        """Probability should sum to 1 at each time step."""
        params = QuantumWalkParams(n_sites=51, n_steps=20, start_pos=25)
        probs = quantum_walk_evolution(params)
        sums = jnp.sum(probs, axis=1)
        assert jnp.allclose(sums, 1.0, atol=1e-4)

    def test_output_shape(self):
        """Output should be (n_steps + 1, n_sites)."""
        params = QuantumWalkParams(n_sites=31, n_steps=10, start_pos=15)
        probs = quantum_walk_evolution(params)
        assert probs.shape == (11, 31)

    def test_initial_condition(self):
        """At t=0, all probability should be at start position."""
        params = QuantumWalkParams(n_sites=21, n_steps=5, start_pos=10)
        probs = quantum_walk_evolution(params)
        assert probs[0, 10] == pytest.approx(1.0, abs=1e-5)

    def test_symmetric_hadamard(self):
        """Hadamard walk with symmetric initial coin state should spread symmetrically."""
        params = QuantumWalkParams(
            coin_angle=jnp.pi / 4,
            n_sites=101,
            n_steps=30,
            start_pos=50,
        )
        probs = quantum_walk_evolution(params)
        # Check symmetry at last time step
        center = 50
        final = probs[-1]
        assert jnp.allclose(final[:center], jnp.flip(final[center + 1:]), atol=1e-4)

    def test_ballistic_spreading(self):
        """Quantum walk variance should grow faster than classical (∝ t vs ∝ t²)."""
        params = QuantumWalkParams(n_sites=201, n_steps=50, start_pos=100)
        probs = quantum_walk_evolution(params)
        positions = jnp.arange(201) - 100

        var_early = float(probs[10] @ (positions**2) - (probs[10] @ positions) ** 2)
        var_late = float(probs[50] @ (positions**2) - (probs[50] @ positions) ** 2)

        # Quantum: var ∝ t², so var(50)/var(10) ≈ 25
        # Classical: var ∝ t, so ratio would be 5
        ratio = var_late / max(var_early, 1e-10)
        assert ratio > 10, f"Variance ratio {ratio} too low for ballistic spreading"


class TestFPTDensity:
    def test_fpt_nonnegative(self):
        """FPT density should be non-negative everywhere."""
        params = QuantumWalkParams(
            n_sites=31, n_steps=40, start_pos=15, absorbing_right=25,
        )
        fpt = quantum_walk_fpt_density(params, boundary="right")
        assert jnp.all(fpt >= -1e-8)

    def test_fpt_sums_to_less_than_one(self):
        """Total FPT probability should be ≤ 1 (some walkers may not reach boundary)."""
        params = QuantumWalkParams(
            n_sites=31, n_steps=60, start_pos=15, absorbing_right=25,
        )
        fpt = quantum_walk_fpt_density(params, boundary="right")
        assert float(jnp.sum(fpt)) <= 1.0 + 1e-5

    def test_no_boundary_raises(self):
        """Should raise if no absorbing boundary is set."""
        params = QuantumWalkParams(n_sites=21, n_steps=10, start_pos=10)
        with pytest.raises(ValueError, match="No absorbing boundary"):
            quantum_walk_fpt_density(params, boundary="right")


class TestClassicalVsQuantumSpreading:
    def test_output_shapes(self):
        """Should return three arrays of matching length."""
        times, c_var, q_var = classical_vs_quantum_spreading(n_sites=51, n_steps=20)
        assert times.shape == (21,)
        assert c_var.shape == (21,)
        assert q_var.shape == (21,)

    def test_quantum_faster_than_classical(self):
        """At late times, quantum variance should exceed classical."""
        times, c_var, q_var = classical_vs_quantum_spreading(n_sites=201, n_steps=50)
        # At t=50, quantum should be well ahead
        assert float(q_var[-1]) > float(c_var[-1])
