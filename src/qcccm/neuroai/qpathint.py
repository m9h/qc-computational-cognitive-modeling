"""Quantum path-integral (qPATHINT) propagation — complex-kernel extension of PATHINT.

PATHINT folds a real probability density through a Gaussian transition kernel; **qPATHINT**
folds a complex amplitude ψ through the **Feynman short-time propagator** — the quantum rung
of the path-integral ladder, and the substrate for Ingber's Ca²⁺ wave-packet modeling
(see smni-cmi `QPATHINT_REIMAGINED.md`). Pure JAX (complex), GPU-ready.

Free-particle short-time/whole-time propagator (exact for H = p²/2m):
    K(x, x'; t) = √(m / 2πiħt) · exp( i m (x−x')² / 2ħt )
and ψ(x, t) = ∫ K(x, x'; t) ψ(x', 0) dx'  ≈  (K · ψ) dx  on a grid.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def gaussian_packet(x: Array, x0: float = 0.0, sigma: float = 1.0,
                    k0: float = 0.0) -> Array:
    """Normalised complex Gaussian wave-packet; |ψ|² has position std `sigma`,
    mean `x0`, and group momentum `k0`."""
    psi = jnp.exp(-((x - x0) ** 2) / (4 * sigma ** 2) + 1j * k0 * x)
    dx = x[1] - x[0]
    psi = psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx)
    return psi.astype(jnp.complex64)


def free_propagator(x: Array, t: float, m: float = 1.0, hbar: float = 1.0) -> Array:
    """Free-particle complex propagator K(x, x'; t) on the grid → (G, G) complex."""
    d = x[:, None] - x[None, :]
    pref = jnp.sqrt(m / (2j * jnp.pi * hbar * t))
    return (pref * jnp.exp(1j * m * d ** 2 / (2 * hbar * t))).astype(jnp.complex64)


def apply(K: Array, psi: Array, dx: float) -> Array:
    """One application of a propagator kernel: ψ ← (K · ψ) dx."""
    return (K @ psi) * dx


def split_step(psi: Array, x: Array, dt: float, potential: Array | None = None,
               m: float = 1.0, hbar: float = 1.0) -> Array:
    """One split-operator (Strang) step: ½ potential → free kinetic (FFT) → ½ potential.
    Robust multi-step propagation with a potential V(x) (e.g. for bound/qPATHINT runs)."""
    dx = x[1] - x[0]
    n = x.shape[0]
    k = 2 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
    Vh = jnp.zeros_like(x) if potential is None else potential
    psi = psi * jnp.exp(-1j * Vh * dt / (2 * hbar))
    psi = jnp.fft.ifft(jnp.fft.fft(psi) * jnp.exp(-1j * hbar * k ** 2 * dt / (2 * m)))
    psi = psi * jnp.exp(-1j * Vh * dt / (2 * hbar))
    return psi.astype(jnp.complex64)


def prob_density(psi: Array, dx: float) -> Array:
    """Normalised |ψ|²."""
    p = jnp.abs(psi) ** 2
    return p / (jnp.sum(p) * dx)


def moments(psi: Array, x: Array, dx: float):
    """Mean and variance of |ψ|² (dx cancels)."""
    p = jnp.abs(psi) ** 2
    Z = jnp.sum(p)
    mean = jnp.sum(x * p) / Z
    var = jnp.sum(x ** 2 * p) / Z - mean ** 2
    return mean, var
