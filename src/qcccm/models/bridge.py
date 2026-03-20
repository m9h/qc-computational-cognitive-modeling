"""Bridge between classical and quantum representations.

Key conversions:
- Stochastic matrix → unitary operator (Szegedy construction)
- Classical belief vector ↔ density matrix
- Quantum expected free energy
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.core.density_matrix import von_neumann_entropy, quantum_relative_entropy


# ---------------------------------------------------------------------------
# Szegedy walk: stochastic matrix → unitary
# ---------------------------------------------------------------------------


def stochastic_to_unitary(B: Array) -> Array:
    """Convert a column-stochastic matrix B to a unitary via the Szegedy construction.

    The Szegedy walk operates on a doubled Hilbert space H ⊗ H (dimension n²).
    Given B with columns summing to 1, the unitary is:

        U = S · (2|ψ_B⟩⟨ψ_B| − I)

    where |ψ_B⟩ = Σ_j |j⟩ ⊗ |π_j⟩ with |π_j⟩ = Σ_i √(B_ij) |i⟩,
    and S is the swap operator on the two registers.

    Args:
        B: (n, n) column-stochastic matrix (columns sum to 1).

    Returns:
        U: (n², n²) unitary matrix.
    """
    n = B.shape[0]

    # Build the "walk states" |π_j⟩ = √B[:,j] for each column j
    sqrt_B = jnp.sqrt(jnp.clip(B, 0.0, None))

    # |ψ_B⟩ = Σ_j |j⟩ ⊗ |π_j⟩ — this is an n×n matrix, flattened to n²
    # Construct the projector Π = |ψ_B⟩⟨ψ_B| column by column
    # Actually we need: for each j, the vector |j⟩⊗|π_j⟩ in the n² space
    # Build the isometry W: |j⟩ → |j⟩⊗|π_j⟩
    W = jnp.zeros((n * n, n), dtype=jnp.complex64)
    for j in range(n):
        # |j⟩ ⊗ |π_j⟩ occupies indices [j*n : (j+1)*n] in the n² space
        W = W.at[j * n : (j + 1) * n, j].set(sqrt_B[:, j].astype(jnp.complex64))

    # Reflection R = 2 W W† − I
    R = 2.0 * W @ jnp.conj(W).T - jnp.eye(n * n, dtype=jnp.complex64)

    # Swap operator on H ⊗ H
    S = _swap_operator(n)

    # Szegedy unitary
    U = S @ R
    return U


def _swap_operator(n: int) -> Array:
    """Swap operator on H⊗H: S|i,j⟩ = |j,i⟩. Returns (n², n²) matrix."""
    S = jnp.zeros((n * n, n * n), dtype=jnp.complex64)
    for i in range(n):
        for j in range(n):
            S = S.at[j * n + i, i * n + j].set(1.0)
    return S


# ---------------------------------------------------------------------------
# Belief ↔ density matrix
# ---------------------------------------------------------------------------


def beliefs_to_density_matrix(beliefs: Array) -> Array:
    """Convert a classical probability vector to a diagonal density matrix.

    This is the "fully decohered" quantum state — no off-diagonal elements,
    von Neumann entropy equals Shannon entropy.

    Args:
        beliefs: (n,) probability vector summing to 1.

    Returns:
        rho: (n, n) diagonal density matrix.
    """
    return jnp.diag(beliefs.astype(jnp.complex64))


def density_matrix_to_beliefs(rho: Array) -> Array:
    """Extract the diagonal of a density matrix as a classical belief vector.

    This is equivalent to measuring in the computational basis.

    Args:
        rho: (n, n) density matrix.

    Returns:
        beliefs: (n,) real probability vector.
    """
    return jnp.diag(rho).real


# ---------------------------------------------------------------------------
# Quantum expected free energy
# ---------------------------------------------------------------------------


@jax.jit
def quantum_efe(
    rho: Array,
    transition_unitary: Array,
    preferred_state: Array,
) -> Array:
    """Quantum expected free energy for a single policy (one-step lookahead).

    G = epistemic_value + pragmatic_value

    Epistemic value: negative von Neumann entropy of predicted state (drives exploration).
    Pragmatic value: divergence from preferred outcome (drives exploitation).

    In the classical limit (diagonal ρ, diagonal preferred), this reduces to
    the standard EFE from active inference.

    Args:
        rho: (n, n) current density matrix (beliefs as quantum state).
        transition_unitary: (n, n) unitary evolution operator for this policy.
        preferred_state: (n, n) density matrix encoding preferred outcomes.

    Returns:
        G: scalar expected free energy (lower = better).
    """
    # Predicted state after transition
    rho_pred = transition_unitary @ rho @ jnp.conj(transition_unitary).T

    # Epistemic term: −S(ρ_pred) — low entropy states are informative
    epistemic = -von_neumann_entropy(rho_pred)

    # Pragmatic term: S(ρ_pred ‖ ρ_pref) = Tr(ρ_pred (log ρ_pred − log ρ_pref))
    pragmatic = quantum_relative_entropy(rho_pred, preferred_state)

    return epistemic + pragmatic
