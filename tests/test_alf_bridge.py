"""Tests for the ALF bridge module (no ALF import required)."""

import jax.numpy as jnp
import numpy as np
import pytest

from qcccm.models.alf_bridge import (
    alf_quantum_efe,
    beliefs_to_quantum_state,
    evaluate_all_policies,
    preferences_to_density_matrix,
    transition_to_unitary,
)


class TestBeliefsToQuantumState:
    def test_zero_quantumness_is_diagonal(self):
        """q=0 should produce a diagonal density matrix."""
        beliefs = jnp.array([0.6, 0.3, 0.1])
        rho = beliefs_to_quantum_state(beliefs, quantumness=0.0)
        assert jnp.allclose(rho, jnp.diag(beliefs.astype(jnp.complex64)), atol=1e-5)

    def test_positive_quantumness_has_coherences(self):
        """q>0 should produce non-zero off-diagonal elements."""
        beliefs = jnp.array([0.6, 0.4])
        rho = beliefs_to_quantum_state(beliefs, quantumness=0.5)
        assert float(jnp.abs(rho[0, 1])) > 0.1

    def test_trace_is_one(self):
        """Density matrix trace should always be 1."""
        for q in [0.0, 0.3, 0.7, 1.0]:
            beliefs = jnp.array([0.5, 0.3, 0.2])
            rho = beliefs_to_quantum_state(beliefs, quantumness=q)
            assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-5)

    def test_positive_semidefinite(self):
        """Density matrix should have non-negative eigenvalues."""
        beliefs = jnp.array([0.7, 0.2, 0.1])
        rho = beliefs_to_quantum_state(beliefs, quantumness=0.5)
        eigvals = jnp.linalg.eigvalsh(rho).real
        assert jnp.all(eigvals >= -1e-5)


class TestTransitionToUnitary:
    def test_output_is_unitary(self):
        """Polar decomposition should produce a unitary."""
        B_a = jnp.array([[0.7, 0.3], [0.3, 0.7]])
        U = transition_to_unitary(B_a)
        product = jnp.conj(U).T @ U
        assert jnp.allclose(product, jnp.eye(2, dtype=jnp.complex64), atol=1e-4)

    def test_identity_transition(self):
        """Identity transition should give identity (or close) unitary."""
        B_a = jnp.eye(3)
        U = transition_to_unitary(B_a)
        product = jnp.conj(U).T @ U
        assert jnp.allclose(product, jnp.eye(3, dtype=jnp.complex64), atol=1e-4)


class TestPreferencesToDensityMatrix:
    def test_trace_is_one(self):
        """Preferred state should have trace 1."""
        C = jnp.array([2.0, 0.0, -1.0])
        rho = preferences_to_density_matrix(C)
        assert jnp.trace(rho).real == pytest.approx(1.0, abs=1e-5)

    def test_peaked_preferences(self):
        """Strong preference should concentrate on preferred outcome."""
        C = jnp.array([10.0, 0.0])
        rho = preferences_to_density_matrix(C)
        assert float(rho[0, 0].real) > 0.9


class TestALFQuantumEFE:
    def test_returns_finite_scalar(self):
        """quantum EFE should return a finite scalar."""
        A = jnp.eye(3, dtype=jnp.float32)
        B = jnp.stack([jnp.eye(3)] * 2, axis=-1)
        C = jnp.array([1.0, 0.0, -1.0])
        beliefs = jnp.array([0.5, 0.3, 0.2])
        G = alf_quantum_efe(A, B, C, beliefs, action=0, quantumness=0.0)
        assert jnp.isfinite(G)
        assert G.shape == ()

    def test_quantumness_changes_efe(self):
        """Different quantumness should give different EFE values."""
        A = jnp.eye(3, dtype=jnp.float32)
        B = jnp.array([
            [[0.8, 0.2], [0.1, 0.7], [0.1, 0.1]],
            [[0.1, 0.1], [0.8, 0.2], [0.1, 0.7]],
            [[0.1, 0.7], [0.1, 0.1], [0.8, 0.2]],
        ])
        C = jnp.array([2.0, 0.0, -1.0])
        beliefs = jnp.array([0.5, 0.3, 0.2])
        G_classical = float(alf_quantum_efe(A, B, C, beliefs, 0, quantumness=0.0))
        G_quantum = float(alf_quantum_efe(A, B, C, beliefs, 0, quantumness=0.5))
        assert G_classical != pytest.approx(G_quantum, abs=1e-4)


class TestEvaluateAllPolicies:
    def test_output_shape(self):
        """Should return one EFE value per policy."""
        A = jnp.eye(2, dtype=jnp.float32)
        B = jnp.stack([jnp.eye(2)] * 2, axis=-1)
        C = jnp.array([1.0, -1.0])
        beliefs = jnp.array([0.5, 0.5])
        policies = np.array([[[0], [0]], [[1], [1]], [[0], [1]]])
        G = evaluate_all_policies(A, B, C, beliefs, policies)
        assert G.shape == (3,)
        assert np.all(np.isfinite(G))

    def test_different_actions_give_different_efe(self):
        """Distinct transitions should produce distinct EFE values."""
        A = jnp.eye(3, dtype=jnp.float32)
        B = jnp.zeros((3, 3, 2))
        B = B.at[:, :, 0].set(jnp.array([
            [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]
        ]))
        B = B.at[:, :, 1].set(jnp.array([
            [0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]
        ]))
        C = jnp.array([3.0, 0.0, -3.0])
        beliefs = jnp.array([0.5, 0.3, 0.2])
        policies = np.array([[[0]], [[1]]])
        G = evaluate_all_policies(A, B, C, beliefs, policies)
        assert G[0] != pytest.approx(G[1], abs=1e-3)
