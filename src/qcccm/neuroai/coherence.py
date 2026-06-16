"""Quantum-coherence sweep: does off-diagonal structure add discriminative power?

The honest operationalization of "what does quantum add?" for the SMNI_CMI EEG
(see research/smni-eeg/docs/CMI_EFE.md). Each trial's multichannel signal is
encoded as a **density matrix** from its channel covariance; a quantumness knob
``q ∈ [0, 1]`` interpolates from the diagonal-only (classical / decohered) state to
the full state with off-diagonal **coherences**:

    ρ(q) = (1−q)·diag(ρ) + q·ρ        # convex ⇒ valid density matrix for q∈[0,1]

A scalar feature (von Neumann entropy) is taken per trial, and we measure
alcoholic-vs-control separation as ``q`` sweeps 0→1. At ``q=0`` only per-channel
variances enter (classical); at ``q>0`` cross-channel coherence enters. If the
group separation does **not** improve with ``q``, that is a clean negative result
on "quantum adds something" for this dataset — and either way it is falsifiable.

Note: this measures whether *off-diagonal density-matrix structure* helps, which is
the concrete content of the quantum question here. The policy-level
``bridge.quantum_efe`` is a different (action-oriented) object, reserved for a task
with an explicit policy structure.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.core.density_matrix import von_neumann_entropy


def trial_density_matrix(X: Array) -> Array:
    """Channel-covariance density matrix of one trial.

    Args:
        X: (C, T) multichannel signal (channels × time).

    Returns:
        ρ: (C, C) Hermitian PSD density matrix, trace 1 (complex).
    """
    Xc = X - jnp.mean(X, axis=1, keepdims=True)
    cov = (Xc @ Xc.T) / (X.shape[1] - 1)
    cov = cov / jnp.trace(cov)
    return cov.astype(jnp.complex64)


def interpolate_coherence(rho: Array, q: float) -> Array:
    """ρ(q) = (1−q)·diag(ρ) + q·ρ — convex blend, valid density matrix for q∈[0,1]."""
    diag = jnp.diag(jnp.diagonal(rho))
    return (1.0 - q) * diag + q * rho


def entropy_features(M: Array, q: float) -> Array:
    """Per-trial von Neumann entropy of ρ(q). M: (N, C, T) -> (N,)."""
    def feat(X: Array) -> Array:
        return von_neumann_entropy(interpolate_coherence(trial_density_matrix(X), q))
    return jax.vmap(feat)(M)


def welch_t(a: Array, c: Array) -> float:
    a = jnp.asarray(a); c = jnp.asarray(c)
    return float((jnp.mean(a) - jnp.mean(c)) /
                 jnp.sqrt(jnp.var(a, ddof=1) / a.size + jnp.var(c, ddof=1) / c.size))


def coherence_sweep(M: Array, groups, qs=(0.0, 0.25, 0.5, 0.75, 1.0)
                    ) -> dict[float, float]:
    """Welch t (alcoholic vs control) on the entropy feature for each q.

    Returns {q: t}. |t| growing with q ⇒ off-diagonal coherence adds discriminability.
    """
    import numpy as np
    groups = np.asarray(groups)
    a_mask = groups == "alcoholic"
    c_mask = groups == "control"
    out: dict[float, float] = {}
    for q in qs:
        f = np.asarray(entropy_features(M, float(q)))
        out[float(q)] = welch_t(f[a_mask], f[c_mask])
    return out
