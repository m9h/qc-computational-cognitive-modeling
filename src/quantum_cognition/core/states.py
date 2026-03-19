"""State preparation utilities: basis states, Bell states, GHZ states.

All states are returned as JAX arrays (complex64) for compatibility with
the rest of the quantum_cognition library.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def computational_basis(n: int, i: int) -> Array:
    """Return |i⟩ in an n-dimensional Hilbert space.

    Args:
        n: Dimension of the Hilbert space.
        i: Index of the basis state (0-indexed).

    Returns:
        Column vector of shape (n,) with 1 at position i.
    """
    return jnp.eye(n, dtype=jnp.complex64)[i]


def plus_state() -> Array:
    """|+⟩ = (|0⟩ + |1⟩) / √2."""
    return jnp.array([1.0, 1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)


def minus_state() -> Array:
    """|−⟩ = (|0⟩ − |1⟩) / √2."""
    return jnp.array([1.0, -1.0], dtype=jnp.complex64) / jnp.sqrt(2.0)


def bell_state(kind: str = "phi+") -> Array:
    """Return a two-qubit Bell state as a 4-component vector.

    Args:
        kind: One of "phi+", "phi-", "psi+", "psi-".

    Returns:
        State vector of shape (4,).
    """
    s = jnp.sqrt(2.0)
    states = {
        "phi+": jnp.array([1, 0, 0, 1], dtype=jnp.complex64) / s,
        "phi-": jnp.array([1, 0, 0, -1], dtype=jnp.complex64) / s,
        "psi+": jnp.array([0, 1, 1, 0], dtype=jnp.complex64) / s,
        "psi-": jnp.array([0, 1, -1, 0], dtype=jnp.complex64) / s,
    }
    if kind not in states:
        raise ValueError(f"Unknown Bell state '{kind}'. Use one of {list(states.keys())}.")
    return states[kind]


def ghz_state(n: int) -> Array:
    """Return an n-qubit GHZ state: (|00…0⟩ + |11…1⟩) / √2.

    Args:
        n: Number of qubits (≥2).

    Returns:
        State vector of shape (2**n,).
    """
    if n < 2:
        raise ValueError(f"GHZ state requires n >= 2, got {n}.")
    d = 2**n
    state = jnp.zeros(d, dtype=jnp.complex64)
    state = state.at[0].set(1.0 / jnp.sqrt(2.0))
    state = state.at[d - 1].set(1.0 / jnp.sqrt(2.0))
    return state
