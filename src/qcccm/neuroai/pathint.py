"""PATHINT — deterministic path-integral density propagation (Ingber).

The non-Monte-Carlo path integral: a probability density is propagated forward by
repeatedly folding a short-time Gaussian **transition kernel** (Ingber's banded
matrix ``T_ij``, smni21_hybrid eq. 12–13)::

    P(x, t+Δt) = ∫ G(x | x'; Δt) P(x', t) dx'  ≈  T · P

with ``G(x | x'; Δt) = N(x ; x' + g(x')·Δt, σ²Δt)`` from drift ``g`` and diffusion
``σ²``. This is the classical density analogue of qPATHINT (which folds a *complex*
amplitude kernel), and the rung between MPPI control and SMNI/CMI in the
"path integrals for general computation" ladder.

Pure JAX: the fold is `T @ P` under `jax.lax.scan`, so the whole propagation is
`jit`-able and differentiable.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def build_transition_matrix(x_grid: Array, drift: Array, diffusion: float,
                            dt: float) -> Array:
    """Short-time Gaussian transition kernel T (column-stochastic).

    T[i, j] = N(x_i ; x_j + drift_j·dt, diffusion·dt), normalised over i so each
    source column j sums to 1.

    Args:
        x_grid: (G,) spatial grid (uniform).
        drift: (G,) drift g(x_j) evaluated at each grid point.
        diffusion: scalar diffusion σ² (variance rate).
        dt: time step.
    """
    var = jnp.maximum(diffusion * dt, 1e-12)
    mu = x_grid + drift * dt                       # (G,) predicted mean per source
    diff = x_grid[:, None] - mu[None, :]           # (target i, source j)
    K = jnp.exp(-0.5 * diff ** 2 / var)
    return K / jnp.sum(K, axis=0, keepdims=True)   # column-stochastic


def propagate(P0: Array, T: Array, n_steps: int) -> Array:
    """Fold the density forward n_steps via P ← T·P (renormalised).

    Returns the trajectory (n_steps, G); does not include the initial P0.
    """
    def step(P: Array, _):
        Pn = T @ P
        Pn = Pn / jnp.sum(Pn)
        return Pn, Pn

    _, traj = jax.lax.scan(step, P0, None, length=n_steps)
    return traj


def linear_drift(x_grid: Array, a: float, b: float = 0.0) -> Array:
    """Convenience: linear (Ornstein–Uhlenbeck) drift g(x) = a·x + b on the grid."""
    return a * x_grid + b


def delta_density(x_grid: Array, x0: float) -> Array:
    """Normalised point-mass density at the grid node nearest x0."""
    P = jnp.zeros_like(x_grid).at[jnp.argmin(jnp.abs(x_grid - x0))].set(1.0)
    return P / jnp.sum(P)


def moments(P: Array, x_grid: Array) -> tuple[Array, Array]:
    """Mean and variance of a (possibly batched, ..., G) density over x_grid."""
    mean = jnp.sum(P * x_grid, axis=-1)
    var = jnp.sum(P * x_grid ** 2, axis=-1) - mean ** 2
    return mean, var
