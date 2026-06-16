"""Tests for qcccm.models.cmi_efe — CMI as the conjugate momentum of the action.

Proves the Gaussian-limit case of the central hypothesis: the CMI obtained by
autodiff of the SMNI Lagrangian (Π = ∂L/∂q̇) equals the closed-form canonical
momenta, i.e. CMI *is* the conjugate momentum the FEP path-integral uses.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from qcccm.models import smni, cmi_efe


def _data(key=jax.random.PRNGKey(0), n=6, C=4, T=48):
    return jax.random.normal(key, (n, C, T))


def test_autodiff_momentum_equals_closed_form():
    M = _data()
    d = smni.fit_linear_drift(M)
    cmi_closed = smni.canonical_momenta(M, d)
    cmi_autodiff = cmi_efe.canonical_momentum_autodiff(M, d)
    assert cmi_autodiff.shape == cmi_closed.shape
    assert jnp.allclose(cmi_autodiff, cmi_closed, atol=1e-5)


def test_action_is_nonnegative_and_finite():
    M = _data(jax.random.PRNGKey(1))
    d = smni.fit_linear_drift(M)
    S = cmi_efe.free_energy_action(M, d)
    assert jnp.isfinite(S) and S >= 0.0


def test_lagrangian_grad_is_precision_weighted_error():
    # single point: ∂L/∂q̇ must equal Σ⁻¹ (q̇ - g)
    C = 3
    d = smni.SMNIDrift(A=jnp.zeros((C, C)), b=jnp.zeros(C),
                       inv_var=jnp.array([1.0, 2.0, 4.0]))
    q = jnp.array([0.0, 0.0, 0.0])
    qdot = jnp.array([1.0, 1.0, 1.0])
    mom = jax.grad(cmi_efe.smni_lagrangian, argnums=0)(qdot, q, d)
    assert jnp.allclose(mom, d.inv_var * (qdot - (d.A @ q + d.b)))
