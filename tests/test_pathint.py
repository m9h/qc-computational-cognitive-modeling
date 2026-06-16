"""Tests for qcccm.neuroai.pathint — deterministic PATHINT density propagation.

Validated against analytic results: free diffusion spreads with var = σ²·t, and an
Ornstein–Uhlenbeck drift relaxes to the stationary variance σ²/(2k).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qcccm.neuroai import pathint as pi


def _grid(n=401, half=8.0):
    return jnp.linspace(-half, half, n)


def test_transition_matrix_column_stochastic():
    x = _grid()
    T = pi.build_transition_matrix(x, pi.linear_drift(x, 0.0), diffusion=1.0, dt=0.01)
    assert T.shape == (x.size, x.size)
    assert jnp.allclose(jnp.sum(T, axis=0), 1.0, atol=1e-5)   # columns sum to 1
    assert jnp.all(T >= 0)


def test_free_diffusion_variance_grows_linearly():
    x = _grid()
    D, dt, n = 1.0, 0.01, 50            # total time 0.5 → var ≈ 0.5
    T = pi.build_transition_matrix(x, pi.linear_drift(x, 0.0), D, dt)
    P0 = pi.delta_density(x, 0.0)
    traj = pi.propagate(P0, T, n)
    mean, var = pi.moments(traj[-1], x)
    assert abs(float(mean)) < 0.05
    assert abs(float(var) - D * dt * n) < 0.1     # ≈ 0.5


def test_ou_reaches_stationary_variance():
    x = _grid()
    k, D, dt, n = 2.0, 1.0, 0.01, 800             # OU: g(x) = -k x
    T = pi.build_transition_matrix(x, pi.linear_drift(x, -k), D, dt)
    P0 = pi.delta_density(x, 3.0)                  # start off-center
    traj = pi.propagate(P0, T, n)
    mean, var = pi.moments(traj[-1], x)
    assert abs(float(mean)) < 0.1                  # relaxes to 0
    assert abs(float(var) - D / (2 * k)) < 0.05    # stationary var ≈ 0.25


def test_normalisation_preserved():
    x = _grid()
    T = pi.build_transition_matrix(x, pi.linear_drift(x, -1.0), 1.0, 0.01)
    traj = pi.propagate(pi.delta_density(x, 1.0), T, 100)
    assert jnp.allclose(jnp.sum(traj, axis=1), 1.0, atol=1e-5)


def test_propagation_is_jittable_and_differentiable():
    x = _grid(201, 6.0)
    P0 = pi.delta_density(x, 0.0)

    def final_var(diffusion):
        T = pi.build_transition_matrix(x, pi.linear_drift(x, -1.0), diffusion, 0.01)
        traj = pi.propagate(P0, T, 50)
        _, var = pi.moments(traj[-1], x)
        return var

    g = jax.grad(final_var)(1.0)                   # d(final var)/d(diffusion)
    assert jnp.isfinite(g) and g > 0               # more diffusion → more spread
