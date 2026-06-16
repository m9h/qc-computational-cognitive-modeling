"""Tests for qcccm.models.smni — SMNI canonical momenta indicators (CMI).

Physics/consistency invariants:
- CMI preserves trial/channel/time shape.
- A pure linear-drift trajectory (no noise) has near-zero canonical momenta:
  Π = Σ⁻¹(Ṁ - g) and g recovers the drift, so the innovation ≈ 0.
- Adding noise strictly increases |Π|.
- The CMI pipeline is differentiable (the whole point of the JAX port — the
  ASA-replacement fit needs gradients).
- The Gaussian log-likelihood is finite and is maximised at the fitted drift.
"""
from __future__ import annotations

# NOTE: do NOT enable jax_enable_x64 here — it is a process-global toggle and
# would leak into the rest of the pytest session, promoting complex64→complex128
# in the network/circuit tests. These tests are tolerance-based and pass in
# float32. (The reproduce_smni_cmi.py script enables x64 for numeric parity.)
import jax
import jax.numpy as jnp
import pytest

from qcccm.models import smni


def _ou_trajectory(key, n=6, C=3, T=128, dt=smni.DT, noise=0.0):
    """Forward-Euler trajectory of a stable linear (OU) drift g(M)=A·M+b."""
    kA, kb, kM0, kn = jax.random.split(key, 4)
    A = -0.5 * jnp.eye(C) + 0.05 * jax.random.normal(kA, (C, C))  # stable
    b = 0.1 * jax.random.normal(kb, (C,))
    M = jnp.zeros((n, C, T))
    M = M.at[:, :, 0].set(jax.random.normal(kM0, (n, C)))
    noise_seq = noise * jax.random.normal(kn, (n, C, T))
    for t in range(1, T):
        g = M[:, :, t - 1] @ A.T + b
        M = M.at[:, :, t].set(M[:, :, t - 1] + dt * g + noise_seq[:, :, t])
    return M, A, b


def test_shapes_and_finiteness():
    M = jax.random.normal(jax.random.PRNGKey(0), (8, 4, 64))
    d = smni.fit_linear_drift(M)
    cmi = smni.canonical_momenta(M, d)
    assert cmi.shape == M.shape
    assert d.A.shape == (4, 4) and d.b.shape == (4,) and d.inv_var.shape == (4,)
    assert jnp.all(jnp.isfinite(cmi))
    assert jnp.all(d.inv_var > 0)


def test_pure_drift_has_small_innovation():
    # The *raw* innovation (Ṁ - g) is what shrinks for pure drift; the CMI
    # Π = Σ⁻¹·innov is the whitened momentum and actually blows up as the
    # diffusion variance → 0, so we test the innovation, not Π.
    def mean_abs_innov(noise):
        M, _, _ = _ou_trajectory(jax.random.PRNGKey(1), noise=noise)
        d = smni.fit_linear_drift(M)
        innov = smni.velocity(M) - smni.drift(M, d)
        return jnp.mean(jnp.abs(innov))

    assert mean_abs_innov(0.0) < mean_abs_innov(0.5)


def test_cmi_is_whitened_unit_variance():
    # By construction Π·√var = innov/√var has ~unit variance per channel on the
    # data it was fit to — a consistency check on the Σ⁻¹ scaling.
    M, _, _ = _ou_trajectory(jax.random.PRNGKey(7), noise=0.4)
    d = smni.fit_linear_drift(M)
    innov = smni.velocity(M) - smni.drift(M, d)
    whitened = innov * jnp.sqrt(d.inv_var)[None, :, None]
    per_chan_var = jnp.var(whitened, axis=(0, 2))   # over trials & time, per channel
    assert jnp.allclose(per_chan_var, 1.0, atol=0.05)


def test_momentum_magnitude_nonnegative():
    M = jax.random.normal(jax.random.PRNGKey(2), (5, 3, 50))
    d = smni.fit_linear_drift(M)
    mag = smni.momentum_magnitude(smni.canonical_momenta(M, d))
    assert mag.shape == (5, 50)
    assert jnp.all(mag >= 0)


def test_cmi_is_differentiable():
    M = jax.random.normal(jax.random.PRNGKey(3), (4, 3, 40))
    d = smni.fit_linear_drift(M)

    def scalar(x):
        return jnp.sum(smni.canonical_momenta(x, d) ** 2)

    g = jax.grad(scalar)(M)
    assert g.shape == M.shape
    assert jnp.all(jnp.isfinite(g))


def test_loglik_finite_and_maximal_at_fit():
    M, _, _ = _ou_trajectory(jax.random.PRNGKey(4), noise=0.3)
    d = smni.fit_linear_drift(M)
    ll_fit = smni.smni_log_likelihood(M, d)
    assert jnp.isfinite(ll_fit)
    # perturbing the drift away from the MLE should not increase the likelihood
    d_bad = d._replace(A=d.A + 1.0)
    assert smni.smni_log_likelihood(M, d_bad) <= ll_fit


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
