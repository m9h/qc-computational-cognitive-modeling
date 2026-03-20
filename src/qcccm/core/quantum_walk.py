"""Discrete-time quantum walk on a line with absorbing boundaries.

The state lives in H_coin ⊗ H_position (dimension 2 × N).
We represent it as a (2, N) complex array where axis 0 is coin (|L⟩, |R⟩)
and axis 1 is position on a 1-D lattice of size N.

Key functions:
- quantum_walk_evolution: full time evolution via jax.lax.scan
- quantum_walk_fpt_density: first-passage time density at absorbing boundaries
  (spectral method — eigendecomposition of restricted evolution operator)
- classical_vs_quantum_spreading: side-by-side variance comparison
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class QuantumWalkParams(NamedTuple):
    """Parameters for a discrete-time quantum walk on a line."""

    coin_angle: float = jnp.pi / 4  # Hadamard = π/4
    n_sites: int = 101  # odd → symmetric about center
    n_steps: int = 50
    start_pos: int = 50  # initial position (0-indexed)
    absorbing_left: int | None = None  # absorbing boundary position
    absorbing_right: int | None = None


# ---------------------------------------------------------------------------
# Coin operators
# ---------------------------------------------------------------------------


def hadamard_coin() -> Array:
    """Standard Hadamard coin: H = (1/√2)[[1,1],[1,-1]]."""
    return jnp.array([[1, 1], [1, -1]], dtype=jnp.complex64) / jnp.sqrt(2.0)


def biased_coin(theta: float) -> Array:
    """Biased coin parameterised by angle θ.

    C(θ) = [[cos θ, sin θ], [sin θ, −cos θ]]

    θ = π/4 recovers the Hadamard coin.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, s], [s, -c]], dtype=jnp.complex64)


# ---------------------------------------------------------------------------
# Shift operator
# ---------------------------------------------------------------------------


def shift_operator(n_sites: int) -> tuple[Array, Array]:
    """Return (S_left, S_right) permutation matrices for the position register.

    S_left shifts position index down by 1 (with wraparound).
    S_right shifts position index up by 1 (with wraparound).
    """
    S_right = jnp.roll(jnp.eye(n_sites, dtype=jnp.complex64), shift=1, axis=0)
    S_left = jnp.roll(jnp.eye(n_sites, dtype=jnp.complex64), shift=-1, axis=0)
    return S_left, S_right


# ---------------------------------------------------------------------------
# Single-step evolution
# ---------------------------------------------------------------------------


def _qw_step(state: Array, coin: Array, S_left: Array, S_right: Array) -> Array:
    """Apply one step: coin ⊗ I then conditional shift.

    Args:
        state: (2, N) array — coin ⊗ position amplitudes.
        coin: (2, 2) coin operator.
        S_left, S_right: (N, N) shift matrices.

    Returns:
        Updated (2, N) state.
    """
    # Apply coin to coin register
    coined = coin @ state  # (2, N)

    # Conditional shift: |L⟩ component shifts left, |R⟩ shifts right
    new_state = jnp.stack([
        S_left @ coined[0],   # |L⟩ → shift left
        S_right @ coined[1],  # |R⟩ → shift right
    ])
    return new_state


# ---------------------------------------------------------------------------
# Full evolution
# ---------------------------------------------------------------------------


def quantum_walk_evolution(params: QuantumWalkParams, coin: Array | None = None) -> Array:
    """Run a quantum walk and return the full trajectory of probability distributions.

    Args:
        params: Walk parameters.
        coin: (2, 2) coin operator. Defaults to biased_coin(params.coin_angle).

    Returns:
        probs: (n_steps + 1, n_sites) array of position probability distributions.
    """
    N = params.n_sites
    if coin is None:
        coin = biased_coin(params.coin_angle)

    S_left, S_right = shift_operator(N)

    # Initial state: coin in |R⟩ (index 1), position at start_pos
    state = jnp.zeros((2, N), dtype=jnp.complex64)
    state = state.at[0, params.start_pos].set(1.0 / jnp.sqrt(2.0))  # |L⟩ component
    state = state.at[1, params.start_pos].set(1.0j / jnp.sqrt(2.0))  # |R⟩ component (×i for symmetry)

    def scan_fn(state, _):
        new_state = _qw_step(state, coin, S_left, S_right)
        prob = jnp.sum(jnp.abs(new_state) ** 2, axis=0)
        return new_state, prob

    initial_prob = jnp.sum(jnp.abs(state) ** 2, axis=0)
    final_state, prob_trajectory = jax.lax.scan(scan_fn, state, None, length=params.n_steps)

    # Prepend initial distribution
    return jnp.concatenate([initial_prob[None, :], prob_trajectory], axis=0)


# ---------------------------------------------------------------------------
# First-passage time density (spectral method)
# ---------------------------------------------------------------------------


def quantum_walk_fpt_density(
    params: QuantumWalkParams,
    coin: Array | None = None,
    boundary: str = "right",
) -> Array:
    """Compute first-passage time density at an absorbing boundary.

    Uses the "survival probability" method: run the walk with the absorbing
    site zeroed out after each step. FPT density at time t is the probability
    that leaks into the absorbing site at step t.

    Args:
        params: Walk parameters. Must have absorbing_left or absorbing_right set.
        coin: (2, 2) coin operator.
        boundary: Which boundary to compute FPT for ("left" or "right").

    Returns:
        fpt: (n_steps,) array — first-passage time density for t = 1, …, n_steps.
    """
    N = params.n_sites
    if coin is None:
        coin = biased_coin(params.coin_angle)

    if boundary == "right":
        absorb = params.absorbing_right
    else:
        absorb = params.absorbing_left

    if absorb is None:
        raise ValueError(f"No absorbing boundary set for '{boundary}'.")

    S_left, S_right = shift_operator(N)

    # Initial state
    state = jnp.zeros((2, N), dtype=jnp.complex64)
    state = state.at[0, params.start_pos].set(1.0 / jnp.sqrt(2.0))
    state = state.at[1, params.start_pos].set(1.0j / jnp.sqrt(2.0))

    def scan_fn(state, _):
        new_state = _qw_step(state, coin, S_left, S_right)
        # Probability arriving at absorbing site this step
        arrival_prob = jnp.sum(jnp.abs(new_state[:, absorb]) ** 2)
        # Zero out the absorbing site (absorb the probability)
        new_state = new_state.at[:, absorb].set(0.0)
        return new_state, arrival_prob

    _, fpt_density = jax.lax.scan(scan_fn, state, None, length=params.n_steps)
    return fpt_density


# ---------------------------------------------------------------------------
# Classical vs quantum spreading comparison
# ---------------------------------------------------------------------------


def classical_vs_quantum_spreading(
    n_sites: int = 101,
    n_steps: int = 50,
    coin_angle: float = jnp.pi / 4,
) -> tuple[Array, Array, Array]:
    """Compare variance growth of classical RW vs quantum walk.

    Returns:
        times: (n_steps + 1,) array of time steps.
        classical_var: (n_steps + 1,) classical random walk variance (= t).
        quantum_var: (n_steps + 1,) quantum walk variance (∝ t²).
    """
    center = n_sites // 2
    params = QuantumWalkParams(
        coin_angle=coin_angle,
        n_sites=n_sites,
        n_steps=n_steps,
        start_pos=center,
    )

    # Quantum walk
    qw_probs = quantum_walk_evolution(params)
    positions = jnp.arange(n_sites) - center
    qw_mean = qw_probs @ positions
    qw_var = qw_probs @ (positions**2) - qw_mean**2

    # Classical random walk: variance = t
    times = jnp.arange(n_steps + 1, dtype=jnp.float32)
    classical_var = times  # Var(X_t) = t for unbiased RW with step ±1

    return times, classical_var, qw_var
