"""SMNI Canonical Momenta Indicators (CMI) — JAX port.

Statistical Mechanics of Neocortical Interactions (Ingber 1981–2021). The
short-time evolution of neocortical fields ``M(t)`` is a Gaussian conditional
probability with Lagrangian

    L(M, Ṁ) = 1/2 (Ṁ - g(M))ᵀ Σ⁻¹ (Ṁ - g(M))

with drift ``g(M)`` and diffusion ``Σ``. The **canonical momenta** conjugate to
the fields are

    Π = ∂L/∂Ṁ = Σ⁻¹ (Ṁ - g(M))                      ← the CMI (Ingber 1997)

Ingber (`smni21_hybrid.pdf`, p.11) gives exactly ``Πⁱ = ∂L/∂(∂qⁱ/∂t)`` and
"CMI = Πⁱ", so this matches the canonical-momentum definition. In the
path-integral / FEP bridge these Π are the object hypothesised to equal the
free-energy action's conjugate momenta (``models/bridge.py``).

This is the differentiable JAX port of the reference implementation in
``smni-eeg/src/cmi.py``. The drift here is the tractable **linear (Ornstein–
Uhlenbeck) surrogate** ``g(M) = A·M + b``; its MLE is closed-form (least
squares). A nonlinear SMNI drift can replace :func:`fit_linear_drift` and be fit
with :mod:`qcccm.fitting.mle` (the modern ASA replacement) via
:func:`smni_log_likelihood`.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

FS_HZ = 256.0          # SMNI_CMI EEG sampling rate
DT = 1.0 / FS_HZ       # seconds per sample (3.906 ms)


class SMNIDrift(NamedTuple):
    """Fitted linear-drift SMNI surrogate (diagonal diffusion)."""
    A: Array           # (C, C) drift coupling
    b: Array           # (C,) drift bias
    inv_var: Array     # (C,) diagonal precision = 1 / diffusion variance


def velocity(M: Array, dt: float = DT) -> Array:
    """Field velocity Ṁ via central differences along the last (time) axis."""
    return jnp.gradient(M, dt, axis=-1)


def _design(M: Array) -> tuple[Array, Array]:
    """Pool trials (N, C, T) into regression matrices X (states), V (velocities)."""
    V = velocity(M)
    C = M.shape[1]
    X = jnp.moveaxis(M, 1, -1).reshape(-1, C)
    Y = jnp.moveaxis(V, 1, -1).reshape(-1, C)
    return X, Y


def fit_linear_drift(M: Array) -> SMNIDrift:
    """Closed-form MLE of the linear drift g(M)=A·M+b and diagonal diffusion.

    For a linear-Gaussian SDE the least-squares fit *is* the maximum-likelihood
    estimate, so no iterative optimisation is needed for this surrogate.

    Args:
        M: trials, shape (N, C, T).
    """
    X, V = _design(M)
    C = X.shape[1]
    Xa = jnp.concatenate([X, jnp.ones((X.shape[0], 1), X.dtype)], axis=1)
    W, *_ = jnp.linalg.lstsq(Xa, V, rcond=None)        # (C+1, C)
    A = W[:C].T
    b = W[C]
    R = V - (X @ A.T + b)
    var = jnp.maximum(jnp.var(R, axis=0), jnp.finfo(R.dtype).eps)
    return SMNIDrift(A=A, b=b, inv_var=1.0 / var)


def drift(M: Array, params: SMNIDrift) -> Array:
    """g(M) = A·M + b applied per time sample. (N, C, T) -> (N, C, T)."""
    return jnp.einsum("ij,njt->nit", params.A, M) + params.b[None, :, None]


def canonical_momenta(M: Array, params: SMNIDrift, dt: float = DT) -> Array:
    """CMI: Π = Σ⁻¹ (Ṁ - g(M)). Trials (N, C, T) -> momenta (N, C, T)."""
    innov = velocity(M, dt) - drift(M, params)
    return innov * params.inv_var[None, :, None]       # diagonal Σ⁻¹


def momentum_magnitude(cmi: Array) -> Array:
    """Per-trial, per-time |Π| = sqrt(Σ_c Πc²). (N, C, T) -> (N, T)."""
    return jnp.sqrt(jnp.sum(cmi ** 2, axis=1))


def smni_log_likelihood(M: Array, params: SMNIDrift, dt: float = DT) -> Array:
    """Scalar Gaussian short-time path-integral log-likelihood of trials M.

    The momenta :func:`canonical_momenta` are the stationary points of the action
    whose density this evaluates. Suitable for :mod:`qcccm.fitting.mle` when the
    drift is made nonlinear (the linear case is already closed-form above).
    """
    innov = velocity(M, dt) - drift(M, params)
    ll = -0.5 * (innov ** 2 * params.inv_var[None, :, None]
                 - jnp.log(params.inv_var)[None, :, None]
                 + jnp.log(2.0 * jnp.pi))
    return jnp.sum(ll)
