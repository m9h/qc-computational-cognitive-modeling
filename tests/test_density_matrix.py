"""Tests for density matrix operations."""

import jax.numpy as jnp
import pytest

from quantum_cognition.core.density_matrix import (
    fidelity,
    maximally_mixed,
    partial_trace,
    pure_state_density_matrix,
    purity,
    quantum_mutual_information,
    quantum_relative_entropy,
    von_neumann_entropy,
)
from quantum_cognition.core.states import bell_state, computational_basis, plus_state


class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self):
        """Pure state |0⟩ should have S = 0."""
        psi = computational_basis(2, 0)
        rho = pure_state_density_matrix(psi)
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-5)

    def test_maximally_mixed_max_entropy(self):
        """Maximally mixed state I/d should have S = log(d)."""
        for d in [2, 4, 8]:
            rho = maximally_mixed(d)
            expected = jnp.log(d)
            assert von_neumann_entropy(rho) == pytest.approx(float(expected), abs=1e-4)

    def test_diagonal_equals_shannon(self):
        """For diagonal ρ, von Neumann entropy = Shannon entropy."""
        probs = jnp.array([0.3, 0.5, 0.2])
        rho = jnp.diag(probs.astype(jnp.complex64))
        shannon = -jnp.sum(probs * jnp.log(probs))
        assert von_neumann_entropy(rho) == pytest.approx(float(shannon), abs=1e-5)

    def test_bell_state_zero_entropy(self):
        """Bell state is pure → S = 0."""
        psi = bell_state("phi+")
        rho = pure_state_density_matrix(psi)
        assert von_neumann_entropy(rho) == pytest.approx(0.0, abs=1e-5)


class TestPartialTrace:
    def test_product_state(self):
        """Partial trace of |0⟩⊗|+⟩ over second system gives |0⟩⟨0|."""
        psi_0 = computational_basis(2, 0)
        psi_plus = plus_state()
        psi_product = jnp.kron(psi_0, psi_plus)
        rho = pure_state_density_matrix(psi_product)

        rho_a = partial_trace(rho, (2, 2), trace_out=1)
        expected = pure_state_density_matrix(psi_0)
        assert jnp.allclose(rho_a, expected, atol=1e-5)

    def test_bell_state_reduced_is_maximally_mixed(self):
        """Tracing out one qubit of |Φ+⟩ gives I/2."""
        psi = bell_state("phi+")
        rho = pure_state_density_matrix(psi)
        rho_a = partial_trace(rho, (2, 2), trace_out=1)
        expected = maximally_mixed(2)
        assert jnp.allclose(rho_a, expected, atol=1e-5)

    def test_trace_preserves_trace(self):
        """Partial trace should preserve Tr(ρ) = 1."""
        psi = bell_state("psi-")
        rho = pure_state_density_matrix(psi)
        rho_a = partial_trace(rho, (2, 2), trace_out=1)
        assert jnp.trace(rho_a).real == pytest.approx(1.0, abs=1e-5)

    def test_asymmetric_dims(self):
        """Partial trace with different subsystem dimensions."""
        # 2 ⊗ 3 system
        rho = maximally_mixed(6)
        rho_a = partial_trace(rho, (2, 3), trace_out=1)
        assert rho_a.shape == (2, 2)
        assert jnp.trace(rho_a).real == pytest.approx(1.0, abs=1e-5)


class TestQuantumMutualInformation:
    def test_product_state_zero_mi(self):
        """Product state should have I(A:B) = 0."""
        psi_0 = computational_basis(2, 0)
        psi_1 = computational_basis(2, 1)
        psi_product = jnp.kron(psi_0, psi_1)
        rho = pure_state_density_matrix(psi_product)
        mi = quantum_mutual_information(rho, (2, 2))
        assert mi == pytest.approx(0.0, abs=1e-4)

    def test_bell_state_max_mi(self):
        """Bell state should have I(A:B) = 2 log 2."""
        psi = bell_state("phi+")
        rho = pure_state_density_matrix(psi)
        mi = quantum_mutual_information(rho, (2, 2))
        assert mi == pytest.approx(2.0 * float(jnp.log(2.0)), abs=1e-4)


class TestPurity:
    def test_pure_state(self):
        """Pure state has Tr(ρ²) = 1."""
        psi = plus_state()
        rho = pure_state_density_matrix(psi)
        assert purity(rho) == pytest.approx(1.0, abs=1e-5)

    def test_maximally_mixed(self):
        """Maximally mixed state has Tr(ρ²) = 1/d."""
        for d in [2, 4]:
            rho = maximally_mixed(d)
            assert purity(rho) == pytest.approx(1.0 / d, abs=1e-5)


class TestFidelity:
    def test_same_state(self):
        """F(ρ, ρ) = 1."""
        psi = plus_state()
        rho = pure_state_density_matrix(psi)
        assert fidelity(rho, rho) == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_states(self):
        """F(|0⟩⟨0|, |1⟩⟨1|) = 0."""
        rho0 = pure_state_density_matrix(computational_basis(2, 0))
        rho1 = pure_state_density_matrix(computational_basis(2, 1))
        assert fidelity(rho0, rho1) == pytest.approx(0.0, abs=1e-4)

    def test_symmetry(self):
        """F(ρ, σ) = F(σ, ρ)."""
        rho = pure_state_density_matrix(plus_state())
        sigma = maximally_mixed(2)
        assert fidelity(rho, sigma) == pytest.approx(float(fidelity(sigma, rho)), abs=1e-4)


class TestQuantumRelativeEntropy:
    def test_same_state_zero(self):
        """S(ρ ‖ ρ) = 0."""
        rho = maximally_mixed(2)
        assert quantum_relative_entropy(rho, rho) == pytest.approx(0.0, abs=1e-4)

    def test_positive(self):
        """S(ρ ‖ σ) ≥ 0 (Klein's inequality)."""
        rho = pure_state_density_matrix(plus_state())
        sigma = maximally_mixed(2)
        assert quantum_relative_entropy(rho, sigma) >= -1e-5
