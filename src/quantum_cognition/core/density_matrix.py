"""Density matrix operations: entropy, partial trace, fidelity.

All functions operate on JAX arrays and are jit-compatible where noted.
Density matrices are represented as (d, d) complex arrays.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import Array


class DensityMatrix(NamedTuple):
    """Lightweight wrapper pairing a density matrix with its subsystem dimensions."""

    rho: Array  # (d, d) complex array
    dims: tuple[int, ...] = ()  # subsystem dimensions, e.g. (2, 2) for two qubits


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def pure_state_density_matrix(psi: Array) -> Array:
    """Return |ψ⟩⟨ψ| from a state vector."""
    return jnp.outer(psi, jnp.conj(psi))


def maximally_mixed(d: int) -> Array:
    """Return the maximally mixed state I/d."""
    return jnp.eye(d, dtype=jnp.complex64) / d


# ---------------------------------------------------------------------------
# Entropy & information measures
# ---------------------------------------------------------------------------


@jax.jit
def von_neumann_entropy(rho: Array) -> Array:
    """S(ρ) = −Tr(ρ log ρ), computed via eigenvalues.

    For a diagonal ρ this equals Shannon entropy of the diagonal.
    Returns a scalar.
    """
    eigenvalues = jnp.linalg.eigvalsh(rho).real
    # Clip to avoid log(0); eigenvalues of a valid ρ are in [0, 1]
    eigenvalues = jnp.clip(eigenvalues, 1e-12, None)
    return -jnp.sum(eigenvalues * jnp.log(eigenvalues))


@jax.jit
def quantum_relative_entropy(rho: Array, sigma: Array) -> Array:
    """S(ρ ‖ σ) = Tr(ρ (log ρ − log σ)).

    Assumes supp(ρ) ⊆ supp(σ). Returns a scalar.
    """
    log_rho = _safe_logm(rho)
    log_sigma = _safe_logm(sigma)
    return jnp.trace(rho @ (log_rho - log_sigma)).real


def _safe_logm(A: Array) -> Array:
    """Matrix logarithm via eigendecomposition, clamping small eigenvalues."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.clip(eigvals.real, 1e-12, None)
    return (eigvecs * jnp.log(eigvals)[None, :]) @ jnp.conj(eigvecs).T


# ---------------------------------------------------------------------------
# Partial trace
# ---------------------------------------------------------------------------


def partial_trace(rho: Array, dims: Sequence[int], trace_out: int) -> Array:
    """Trace out one subsystem of a composite density matrix.

    Args:
        rho: Density matrix of shape (d, d) where d = prod(dims).
        dims: Tuple of subsystem dimensions, e.g. (2, 3) for qubit ⊗ qutrit.
        trace_out: Index of the subsystem to trace out (0-indexed).

    Returns:
        Reduced density matrix for the remaining subsystems.
    """
    dims = tuple(dims)
    n = len(dims)
    d = int(jnp.prod(jnp.array(dims)))

    if rho.shape != (d, d):
        raise ValueError(f"rho shape {rho.shape} incompatible with dims {dims} (product={d}).")

    # Reshape into a tensor with 2n indices: (d0, d1, ..., d_{n-1}, d0, d1, ..., d_{n-1})
    tensor = rho.reshape(*dims, *dims)

    # Contract axis `trace_out` with axis `trace_out + n`
    result = jnp.trace(tensor, axis1=trace_out, axis2=trace_out + n)

    # Remaining dimensions
    remaining = [d for i, d in enumerate(dims) if i != trace_out]
    d_remaining = int(jnp.prod(jnp.array(remaining))) if remaining else 1
    return result.reshape(d_remaining, d_remaining)


# ---------------------------------------------------------------------------
# Quantum mutual information
# ---------------------------------------------------------------------------


def quantum_mutual_information(rho_ab: Array, dims: tuple[int, int]) -> Array:
    """I(A:B) = S(ρ_A) + S(ρ_B) − S(ρ_AB) for a bipartite state.

    Args:
        rho_ab: Joint density matrix.
        dims: (d_A, d_B) subsystem dimensions.

    Returns:
        Quantum mutual information (scalar).
    """
    rho_a = partial_trace(rho_ab, dims, trace_out=1)
    rho_b = partial_trace(rho_ab, dims, trace_out=0)
    return von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) - von_neumann_entropy(rho_ab)


# ---------------------------------------------------------------------------
# Purity & fidelity
# ---------------------------------------------------------------------------


@jax.jit
def purity(rho: Array) -> Array:
    """Tr(ρ²). Equal to 1 for pure states, 1/d for maximally mixed."""
    return jnp.trace(rho @ rho).real


@jax.jit
def fidelity(rho: Array, sigma: Array) -> Array:
    """F(ρ, σ) = (Tr √(√ρ σ √ρ))².

    Computed via eigendecomposition for numerical stability.
    """
    sqrt_rho = _matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    # Eigenvalues of the inner product; take sqrt, sum, square
    eigvals = jnp.linalg.eigvalsh(inner).real
    eigvals = jnp.clip(eigvals, 0.0, None)
    return jnp.sum(jnp.sqrt(eigvals)) ** 2


def _matrix_sqrt(A: Array) -> Array:
    """Hermitian positive-semidefinite matrix square root via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.clip(eigvals.real, 0.0, None)
    return (eigvecs * jnp.sqrt(eigvals)[None, :]) @ jnp.conj(eigvecs).T
