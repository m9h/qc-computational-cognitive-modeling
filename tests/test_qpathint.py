"""Tests for qcccm.neuroai.qpathint — quantum path-integral (complex-kernel) propagation.

qPATHINT is PATHINT with a *complex* short-time kernel (Feynman propagator): it folds
a complex amplitude ψ forward instead of a real density. Validated against the analytic
free-particle Gaussian wave-packet, whose probability width spreads as
    σ(t)² = σ0² ( 1 + (ħ t / (2 m σ0²))² ).
Written red-first (module does not exist yet).
"""
from __future__ import annotations

import jax.numpy as jnp

from qcccm.neuroai import qpathint as qp


def _grid(n=4096, half=60.0):
    x = jnp.linspace(-half, half, n)
    return x, float(x[1] - x[0])


def test_free_particle_gaussian_spreads_analytically():
    x, dx = _grid()
    sigma0, m, hbar, t = 1.0, 1.0, 1.0, 2.0
    psi0 = qp.gaussian_packet(x, x0=0.0, sigma=sigma0, k0=0.0)
    K = qp.free_propagator(x, t, m=m, hbar=hbar)
    psi_t = qp.apply(K, psi0, dx)
    mean, var = qp.moments(psi_t, x, dx)
    expected = sigma0 ** 2 * (1 + (hbar * t / (2 * m * sigma0 ** 2)) ** 2)  # = 2.0
    assert abs(float(mean)) < 0.2
    assert abs(float(var) - expected) < 0.2


def test_norm_preserved():
    x, dx = _grid()
    psi0 = qp.gaussian_packet(x, 0.0, 1.0, 0.0)
    K = qp.free_propagator(x, 1.5, m=1.0, hbar=1.0)
    psi_t = qp.apply(K, psi0, dx)
    norm = float(jnp.sum(jnp.abs(psi_t) ** 2) * dx)
    assert abs(norm - 1.0) < 0.05


def test_moving_packet_translates():
    x, dx = _grid()
    k0, t, m, hbar = 2.0, 2.0, 1.0, 1.0
    psi0 = qp.gaussian_packet(x, x0=0.0, sigma=1.5, k0=k0)
    psi_t = qp.apply(qp.free_propagator(x, t, m=m, hbar=hbar), psi0, dx)
    mean, _ = qp.moments(psi_t, x, dx)
    assert abs(float(mean) - hbar * k0 * t / m) < 0.5    # group velocity ħk0/m


def test_amplitude_is_complex():
    x, _ = _grid(256, 10.0)
    psi = qp.gaussian_packet(x, 0.0, 1.0, 1.0)
    assert jnp.iscomplexobj(psi)
