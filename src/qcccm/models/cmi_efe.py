"""CMI ↔ free-energy momenta — the analytic core of the bridge.

The project's central hypothesis (research/smni-eeg/docs/PATH_INTEGRAL.md) is that
Ingber's **Canonical Momenta Indicators** are the conjugate momenta of Friston's
free-energy *action*. In the Gaussian / Laplace generative model this is not a
guess but an identity, and this module proves it *computationally* via autodiff.

For the SMNI short-time Lagrangian

    L(q, q̇) = ½ (q̇ - g(q))ᵀ Σ⁻¹ (q̇ - g(q))

the canonical/conjugate momentum is, by definition,

    Π = ∂L/∂q̇ = Σ⁻¹ (q̇ - g(q))                         (= the CMI)

The same quantity is the FEP "momentum": under a Laplace-encoded variational free
energy with a Gaussian generative model, the free energy is the precision-weighted
prediction error ½ εᵀΣ⁻¹ε with ε = q̇ - g(q) (generalized motion minus flow), and
its gradient w.r.t. generalized velocity is exactly Σ⁻¹ε. So at this order **CMI ≡
FEP conjugate momentum** — they are literally the same object.

Here we compute Π by `jax.grad` of the Lagrangian (not by the closed form) and the
test asserts it matches :func:`qcccm.models.smni.canonical_momenta`. The
*interesting* science is only where this breaks: a non-diagonal (coherent) density
matrix in :func:`qcccm.models.bridge.quantum_efe` gives a quantum free-energy whose
momentum differs from the classical CMI — that divergence, swept by the
``quantumness`` knob, is "what quantum adds" made measurable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.models import smni
from qcccm.models.smni import DT, SMNIDrift


def smni_lagrangian(qdot: Array, q: Array, drift: SMNIDrift) -> Array:
    """SMNI short-time Lagrangian at one point: ½ (q̇-g)ᵀ Σ⁻¹ (q̇-g)."""
    g = drift.A @ q + drift.b
    return 0.5 * jnp.sum(drift.inv_var * (qdot - g) ** 2)


# Π = ∂L/∂q̇  — the conjugate momentum, by automatic differentiation
_momentum_point = jax.grad(smni_lagrangian, argnums=0)


def canonical_momentum_autodiff(M: Array, drift: SMNIDrift, dt: float = DT) -> Array:
    """CMI computed as ∂L/∂q̇ via autodiff, over a trial tensor (N, C, T).

    Equivalent to :func:`qcccm.models.smni.canonical_momenta`, but obtained by
    differentiating the action's Lagrangian — making explicit that CMI *is* the
    conjugate momentum (Ingber 1997; FEP path-integral, Friston et al. 2023).
    """
    V = smni.velocity(M, dt)

    def point(qd: Array, q: Array) -> Array:
        return _momentum_point(qd, q, drift)

    over_time = jax.vmap(point, in_axes=(1, 1), out_axes=1)      # (C,T)
    over_trials = jax.vmap(over_time, in_axes=(0, 0), out_axes=0)  # (N,C,T)
    return over_trials(V, M)


def free_energy_action(M: Array, drift: SMNIDrift, dt: float = DT) -> Array:
    """The SMNI action = Σ_t L = the (negative-log) Gaussian path density, i.e. the
    Laplace variational free energy of the trajectory. CMI = its conjugate momenta.
    """
    V = smni.velocity(M, dt)
    g = smni.drift(M, drift)
    return 0.5 * jnp.sum(drift.inv_var[None, :, None] * (V - g) ** 2)
