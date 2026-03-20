"""Tests for the classical-quantum bridge module."""

import jax.numpy as jnp
import pytest

from qcccm.core.density_matrix import (
    pure_state_density_matrix,
    von_neumann_entropy,
)
from qcccm.models.bridge import (
    beliefs_to_density_matrix,
    density_matrix_to_beliefs,
    quantum_efe,
    stochastic_to_unitary,
)


class TestSzegedyConstruction:
    def test_output_is_unitary(self):
        """Szegedy unitary should satisfy U†U = I."""
        # Simple 2×2 column-stochastic matrix
        B = jnp.array([[0.7, 0.3], [0.3, 0.7]])
        U = stochastic_to_unitary(B)
        product = jnp.conj(U).T @ U
        assert jnp.allclose(product, jnp.eye(4, dtype=jnp.complex64), atol=1e-4)

    def test_identity_transition(self):
        """Identity stochastic matrix should give a valid unitary."""
        B = jnp.eye(3)
        U = stochastic_to_unitary(B)
        product = jnp.conj(U).T @ U
        assert jnp.allclose(product, jnp.eye(9, dtype=jnp.complex64), atol=1e-4)

    def test_uniform_transition(self):
        """Uniform stochastic matrix should give a valid unitary."""
        n = 3
        B = jnp.ones((n, n)) / n
        U = stochastic_to_unitary(B)
        product = jnp.conj(U).T @ U
        assert jnp.allclose(product, jnp.eye(n * n, dtype=jnp.complex64), atol=1e-4)

    def test_output_dimension(self):
        """Szegedy unitary on n-state system should be n² × n²."""
        B = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        U = stochastic_to_unitary(B)
        assert U.shape == (4, 4)

        B3 = jnp.ones((3, 3)) / 3
        U3 = stochastic_to_unitary(B3)
        assert U3.shape == (9, 9)


class TestBeliefConversion:
    def test_roundtrip(self):
        """beliefs → density matrix → beliefs should be identity."""
        beliefs = jnp.array([0.2, 0.5, 0.3])
        rho = beliefs_to_density_matrix(beliefs)
        recovered = density_matrix_to_beliefs(rho)
        assert jnp.allclose(recovered, beliefs, atol=1e-5)

    def test_diagonal_density_matrix(self):
        """beliefs_to_density_matrix should produce a diagonal matrix."""
        beliefs = jnp.array([0.6, 0.4])
        rho = beliefs_to_density_matrix(beliefs)
        # Off-diagonal should be zero
        assert rho[0, 1] == pytest.approx(0.0, abs=1e-7)
        assert rho[1, 0] == pytest.approx(0.0, abs=1e-7)

    def test_entropy_match(self):
        """von Neumann entropy of diagonal ρ should match Shannon entropy of beliefs."""
        beliefs = jnp.array([0.3, 0.5, 0.2])
        rho = beliefs_to_density_matrix(beliefs)
        vn = von_neumann_entropy(rho)
        shannon = -jnp.sum(beliefs * jnp.log(beliefs))
        assert vn == pytest.approx(float(shannon), abs=1e-4)

    def test_coherent_state_not_diagonal(self):
        """density_matrix_to_beliefs on a coherent state loses off-diagonal info."""
        psi = jnp.array([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)
        rho = pure_state_density_matrix(psi)
        beliefs = density_matrix_to_beliefs(rho)
        # Both components should be 0.5
        assert jnp.allclose(beliefs, jnp.array([0.5, 0.5]), atol=1e-5)
        # But the original rho has off-diagonal elements
        assert abs(rho[0, 1]) > 0.4


class TestQuantumEFE:
    def test_classical_limit(self):
        """With diagonal ρ and commuting U, quantum EFE should behave like classical EFE."""
        beliefs = jnp.array([0.5, 0.5])
        rho = beliefs_to_density_matrix(beliefs)
        # Identity transition (stay in place)
        U = jnp.eye(2, dtype=jnp.complex64)
        preferred = beliefs_to_density_matrix(jnp.array([0.9, 0.1]))

        G = quantum_efe(rho, U, preferred)
        # Should be finite and real
        assert jnp.isfinite(G)

    def test_preferred_state_minimizes_efe(self):
        """Being in the preferred state with identity transition should give lower EFE."""
        preferred = beliefs_to_density_matrix(jnp.array([0.9, 0.1]))
        U = jnp.eye(2, dtype=jnp.complex64)

        # Already at preferred state
        G_preferred = quantum_efe(preferred, U, preferred)
        # Far from preferred state
        rho_far = beliefs_to_density_matrix(jnp.array([0.1, 0.9]))
        G_far = quantum_efe(rho_far, U, preferred)

        # Pragmatic term should be smaller when already at preferred state
        assert G_preferred < G_far

    def test_efe_is_scalar(self):
        """quantum_efe should return a scalar."""
        rho = beliefs_to_density_matrix(jnp.array([0.5, 0.5]))
        U = jnp.eye(2, dtype=jnp.complex64)
        pref = beliefs_to_density_matrix(jnp.array([0.8, 0.2]))
        G = quantum_efe(rho, U, pref)
        assert G.shape == ()
