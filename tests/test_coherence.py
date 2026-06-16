"""Tests for qcccm.neuroai.coherence — the quantum-coherence sweep.

Validates that the entropy feature picks up *off-diagonal* (coherence) group
differences: a synthetic dataset where groups differ only in cross-channel
correlation (identical marginal variances) must show ~no separation at q=0
(diagonal only) and strong separation at q=1 (full coherence).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from qcccm.neuroai import coherence as co


def test_density_matrix_validity():
    X = jax.random.normal(jax.random.PRNGKey(0), (6, 128))
    rho = co.trial_density_matrix(X)
    assert rho.shape == (6, 6)
    assert jnp.allclose(jnp.trace(rho).real, 1.0, atol=1e-5)
    assert jnp.allclose(rho, jnp.conj(rho).T, atol=1e-5)          # Hermitian
    evals = jnp.linalg.eigvalsh(rho).real
    assert jnp.all(evals > -1e-6)                                 # PSD


def test_interpolation_endpoints_valid():
    X = jax.random.normal(jax.random.PRNGKey(1), (5, 100))
    rho = co.trial_density_matrix(X)
    for q in (0.0, 0.5, 1.0):
        rq = co.interpolate_coherence(rho, q)
        assert jnp.allclose(jnp.trace(rq).real, 1.0, atol=1e-5)
        assert jnp.all(jnp.linalg.eigvalsh(rq).real > -1e-6)
    assert jnp.allclose(co.interpolate_coherence(rho, 1.0), rho, atol=1e-6)
    assert jnp.allclose(jnp.diagonal(co.interpolate_coherence(rho, 0.0)),
                        jnp.diagonal(rho), atol=1e-6)


def _synthetic(n=40, C=6, T=400, r=0.5, seed=0):
    """Two groups with identical (unit) marginal variances, differing ONLY in
    off-diagonal correlation. Sampled from N(0, Σ) via Cholesky:
    control Σ=I; alcoholic Σ=(1−r)I + r·11ᵀ (equicorrelation, unit diagonal)."""
    C_eq = (1 - r) * jnp.eye(C) + r * jnp.ones((C, C))   # unit diagonal
    L = jnp.linalg.cholesky(C_eq)
    key = jax.random.PRNGKey(seed)
    trials, groups = [], []
    for _ in range(n):
        key, k1, k2 = jax.random.split(key, 3)
        trials.append(jax.random.normal(k1, (C, T)))          # control: Σ=I
        groups.append("control")
        trials.append(L @ jax.random.normal(k2, (C, T)))      # alcoholic: Σ=equicorr
        groups.append("alcoholic")
    return jnp.stack(trials), np.array(groups)


def test_coherence_separates_only_with_q():
    M, groups = _synthetic()
    sweep = co.coherence_sweep(M, groups, qs=(0.0, 1.0))
    # diagonal-only (q=0): marginal variances match → weak separation
    # full coherence (q=1): correlation differs → strong separation
    assert abs(sweep[0.0]) < 2.0
    assert abs(sweep[1.0]) > 3.0
    assert abs(sweep[1.0]) > abs(sweep[0.0])
