"""Tests for qcccm.models.smni_nonlinear — nonlinear SMNI drift via fit_mle.

Validates that MLE (the ASA replacement) recovers a known tanh drift and that the
nonlinear fit beats the linear one when the data is genuinely nonlinear.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qcccm.fitting.mle import MLEConfig, fit_mle
from qcccm.models import smni, smni_nonlinear as nl


def test_g_channel_form():
    th = jnp.array([0.5, 0.1, 2.0, 1.5])
    M = jnp.array([0.0, 1.0, -1.0])
    g = nl._g_channel(th, M)
    expected = 0.5 * M + 0.1 + 2.0 * jnp.tanh(1.5 * M)
    assert jnp.allclose(g, expected)


def test_mle_recovers_nonlinear_drift():
    # synthetic (V, M) with a strong tanh component + small noise
    key = jax.random.PRNGKey(0)
    M = jax.random.normal(key, (20, 200))
    true = jnp.array([0.3, -0.2, 1.8, 1.2])
    V = nl._g_channel(true, M) + 0.05 * jax.random.normal(
        jax.random.PRNGKey(1), M.shape)

    cfg = MLEConfig(n_restarts=3, compute_hessian=False, max_iter=400)
    init = jnp.array([0.0, 0.0, 0.0, 1.0])
    res = fit_mle(lambda t: nl._profile_nll(t, V, M), init,
                  bounds=nl._BOUNDS, n_observations=V.size, config=cfg)
    fitted = np.asarray(res.params["x"])
    # the nonlinear gain β must be recovered as clearly nonzero
    assert abs(fitted[2]) > 1.0

    # and the nonlinear NLL must beat the linear-only fit (β=γ=0 frozen)
    lin_init = jnp.array([float(fitted[0]), float(fitted[1]), 0.0, 0.0])
    nll_nonlin = float(nl._profile_nll(jnp.array(fitted), V, M))
    nll_lin = float(nl._profile_nll(lin_init, V, M))
    assert nll_nonlin < nll_lin


def test_fit_nonlinear_drift_end_to_end():
    M = jax.random.normal(jax.random.PRNGKey(2), (8, 3, 64))
    p = nl.fit_nonlinear_drift(M, config=MLEConfig(
        n_restarts=1, compute_hessian=False, max_iter=100))
    assert p.a.shape == (3,) and p.inv_var.shape == (3,)
    assert jnp.all(jnp.isfinite(p.beta)) and jnp.all(p.inv_var > 0)
    cmi = nl.canonical_momenta(M, p)
    assert cmi.shape == M.shape and jnp.all(jnp.isfinite(cmi))


def test_linear_baseline_matches_smni_diagonal_spirit():
    # per-channel linear baseline returns β=0 and finite precisions
    M = jax.random.normal(jax.random.PRNGKey(3), (6, 4, 50))
    p = nl.fit_linear_drift_perchannel(M)
    assert jnp.allclose(p.beta, 0.0)
    assert jnp.all(p.inv_var > 0)
