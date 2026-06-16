"""Nonlinear SMNI drift fit via fit_mle — the modern ASA replacement.

Ingber's SMNI drift is nonlinear (a ``tanh`` saturation: g ∝ M + N·tanh F). The
linear surrogate in :mod:`qcccm.models.smni` drops that. Here we fit a per-channel
nonlinear drift

    g_c(M) = a·M + b + β·tanh(γ·M)            (nests the linear model at β=0)

by **maximum likelihood with autodiff gradients** (:func:`qcccm.fitting.mle.fit_mle`)
— exactly what replaces Ingber's Adaptive Simulated Annealing (no hand-derived
gradients, no 6-day single-core runs). The diffusion variance is profiled out, so
each channel is a clean 4-parameter MLE; the canonical momenta (CMI) are then
``Π_c = (Ṁ_c − g_c)/σ̂²_c`` as before.

Use :func:`fit_nonlinear_drift` (per-channel ``fit_mle``) vs the closed-form linear
baseline to test whether the nonlinearity adds held-out group discriminability.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from qcccm.fitting.mle import MLEConfig, fit_mle
from qcccm.models import smni


class ChannelDrift(NamedTuple):
    """Per-channel nonlinear drift params + diffusion. Each array is (C,)."""
    a: Array
    b: Array
    beta: Array
    gamma: Array
    inv_var: Array


def _g_channel(theta: Array, M_c: Array) -> Array:
    """g_c(M) = a·M + b + β·tanh(γ·M).  theta=(a,b,β,γ); M_c any shape."""
    a, b, beta, gamma = theta
    return a * M_c + b + beta * jnp.tanh(gamma * M_c)


def _profile_nll(theta: Array, V_c: Array, M_c: Array) -> Array:
    """Gaussian profile NLL (variance profiled out): ½ N log(mean resid²)."""
    r = V_c - _g_channel(theta, M_c)
    return 0.5 * r.size * jnp.log(jnp.mean(r ** 2) + 1e-12)


_BOUNDS = [(-20.0, 20.0), (-20.0, 20.0), (-20.0, 20.0), (-8.0, 8.0)]
_CFG = MLEConfig(n_restarts=2, compute_hessian=False, max_iter=300)


def fit_nonlinear_drift(M: Array, dt: float = smni.DT,
                        config: MLEConfig = _CFG) -> ChannelDrift:
    """Fit the per-channel nonlinear drift by MLE (fit_mle) over trials M (N,C,T)."""
    V = np.asarray(smni.velocity(M, dt))
    M = np.asarray(M)
    C = M.shape[1]
    a = np.zeros(C); b = np.zeros(C); beta = np.zeros(C); gamma = np.zeros(C)
    inv_var = np.zeros(C)
    for c in range(C):
        Vc = jnp.asarray(V[:, c, :]); Mc = jnp.asarray(M[:, c, :])
        # linear least-squares init (a, b); β=0, γ=1
        A = np.vstack([M[:, c, :].ravel(), np.ones(M[:, c, :].size)]).T
        a0, b0 = np.linalg.lstsq(A, V[:, c, :].ravel(), rcond=None)[0]
        init = jnp.array([a0, b0, 0.0, 1.0])
        res = fit_mle(lambda th: _profile_nll(th, Vc, Mc), init,
                      bounds=_BOUNDS, n_observations=Vc.size, config=config)
        th = np.asarray(res.params["x"])
        a[c], b[c], beta[c], gamma[c] = th
        r = np.asarray(Vc) - np.asarray(_g_channel(jnp.array(th), Mc))
        inv_var[c] = 1.0 / max(float(np.mean(r ** 2)), np.finfo(float).eps)
    return ChannelDrift(a=jnp.array(a), b=jnp.array(b), beta=jnp.array(beta),
                        gamma=jnp.array(gamma), inv_var=jnp.array(inv_var))


def fit_linear_drift_perchannel(M: Array, dt: float = smni.DT) -> ChannelDrift:
    """Closed-form per-channel LINEAR baseline (β=0): the nested comparison model."""
    V = np.asarray(smni.velocity(M, dt)); Mn = np.asarray(M)
    C = Mn.shape[1]
    a = np.zeros(C); b = np.zeros(C); inv_var = np.zeros(C)
    for c in range(C):
        A = np.vstack([Mn[:, c, :].ravel(), np.ones(Mn[:, c, :].size)]).T
        sol = np.linalg.lstsq(A, V[:, c, :].ravel(), rcond=None)[0]
        a[c], b[c] = sol
        r = V[:, c, :].ravel() - (A @ sol)
        inv_var[c] = 1.0 / max(float(np.mean(r ** 2)), np.finfo(float).eps)
    return ChannelDrift(a=jnp.array(a), b=jnp.array(b), beta=jnp.zeros(C),
                        gamma=jnp.ones(C), inv_var=jnp.array(inv_var))


def drift(M: Array, p: ChannelDrift) -> Array:
    """g(M) per channel, broadcast over (N, C, T)."""
    a = p.a[None, :, None]; b = p.b[None, :, None]
    beta = p.beta[None, :, None]; gamma = p.gamma[None, :, None]
    return a * M + b + beta * jnp.tanh(gamma * M)


def canonical_momenta(M: Array, p: ChannelDrift, dt: float = smni.DT) -> Array:
    """CMI with the (non)linear per-channel drift: Π = Σ⁻¹(Ṁ − g(M))."""
    innov = smni.velocity(M, dt) - drift(M, p)
    return innov * p.inv_var[None, :, None]
