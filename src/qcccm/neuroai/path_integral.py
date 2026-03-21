"""Feynman path integral formulation of evidence accumulation.

Extends the discrete quantum walk model to continuous paths where
evidence accumulates along ALL possible trajectories simultaneously,
with quantum interference between paths modifying the evidence
distribution and first-passage time density.

The key parameter ℏ_eff (effective Planck constant) controls the
degree of quantum interference:
- ℏ_eff → 0: classical limit (only the most probable path contributes)
- ℏ_eff > 0: multiple paths interfere, producing non-classical patterns

This is the continuous-path analogue of the discrete quantum walk
in core/quantum_walk.py.

References:
    Busemeyer, J.R. & Bruza, P.D. (2012). Quantum Models of Cognition
        and Decision. Ch. 9: Quantum dynamics models.
    Feynman, R.P. & Hibbs, A.R. (1965). Quantum Mechanics and Path
        Integrals.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PathIntegralParams(NamedTuple):
    """Parameters for path integral evidence accumulation."""

    n_paths: int = 1000  # Monte Carlo path samples
    dt: float = 0.01  # time discretisation step
    T_max: float = 1.0  # maximum time
    drift: float = 0.5  # evidence accumulation bias
    diffusion: float = 1.0  # noise coefficient
    hbar_eff: float = 1.0  # effective Planck constant (quantum strength)
    x_start: float = 0.0  # starting evidence level
    seed: int = 42


# ---------------------------------------------------------------------------
# Classical action
# ---------------------------------------------------------------------------


def classical_action(
    path: np.ndarray,
    dt: float,
    drift: float = 0.0,
    diffusion: float = 1.0,
) -> float:
    """Compute the classical action S[x(t)] along a discretised path.

    The Lagrangian for drift-diffusion:
        L = (1/2) (dx/dt − drift)² / diffusion²

    Args:
        path: (n_steps,) positions at each time step.
        dt: time step size.
        drift: evidence accumulation rate.
        diffusion: noise strength.

    Returns:
        Scalar action S.
    """
    velocities = np.diff(path) / dt
    lagrangian = 0.5 * (velocities - drift) ** 2 / max(diffusion**2, 1e-12)
    return float(np.sum(lagrangian) * dt)


# ---------------------------------------------------------------------------
# Path generation (Brownian bridge)
# ---------------------------------------------------------------------------


def _generate_brownian_bridges(
    x0: float,
    xf: float,
    n_steps: int,
    n_paths: int,
    diffusion: float,
    dt: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate Brownian bridge paths from x0 to xf.

    Returns:
        (n_paths, n_steps) array of path positions.
    """
    T = n_steps * dt
    # Free Brownian paths
    increments = rng.randn(n_paths, n_steps - 1) * np.sqrt(diffusion * dt)
    free_paths = np.zeros((n_paths, n_steps))
    free_paths[:, 0] = x0
    for t in range(1, n_steps):
        free_paths[:, t] = free_paths[:, t - 1] + increments[:, t - 1]

    # Condition on endpoint: bridge construction
    times = np.arange(n_steps) * dt
    endpoint_drift = free_paths[:, -1:]  # where free path ended
    correction = (xf - endpoint_drift) * (times[None, :] / max(T, 1e-12))
    bridges = free_paths - free_paths[:, -1:] * (times[None, :] / max(T, 1e-12)) + correction

    return bridges


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------


def path_integral_propagator(
    x0: float,
    xf: float,
    T: float,
    params: PathIntegralParams,
) -> complex:
    """Compute the quantum propagator K(xf, T | x0, 0).

    K = (1/Z) Σ_paths exp(i S[path] / ℏ_eff)

    Uses Monte Carlo sampling of Brownian bridge paths.

    Args:
        x0: starting position.
        xf: ending position.
        T: propagation time.
        params: path integral parameters.

    Returns:
        Complex-valued propagator K.
    """
    rng = np.random.RandomState(params.seed)
    n_steps = max(int(T / params.dt), 2)

    bridges = _generate_brownian_bridges(
        x0, xf, n_steps, params.n_paths, params.diffusion, params.dt, rng,
    )

    # Compute action for each path
    phases = np.zeros(params.n_paths, dtype=complex)
    for i in range(params.n_paths):
        S = classical_action(bridges[i], params.dt, params.drift, params.diffusion)
        phases[i] = np.exp(1j * S / max(params.hbar_eff, 1e-12))

    K = np.mean(phases)
    return complex(K)


# ---------------------------------------------------------------------------
# Evidence accumulation density
# ---------------------------------------------------------------------------


def evidence_accumulation_density(
    params: PathIntegralParams,
    x_grid: np.ndarray | None = None,
    n_grid: int = 101,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute evidence probability density via path integrals.

    Continuous analogue of quantum_walk_evolution: returns P(x, t)
    at each time step on a spatial grid.

    Args:
        params: path integral parameters.
        x_grid: spatial grid. If None, auto-generated.
        n_grid: grid size if x_grid is None.

    Returns:
        times: (n_time,) time points.
        x_grid: (n_grid,) spatial grid.
        density: (n_time, n_grid) probability density.
    """
    n_time = max(int(params.T_max / params.dt), 2)
    times = np.linspace(params.dt, params.T_max, n_time)

    if x_grid is None:
        x_range = 3.0 * params.diffusion * np.sqrt(params.T_max) + abs(params.drift) * params.T_max
        x_grid = np.linspace(params.x_start - x_range, params.x_start + x_range, n_grid)

    density = np.zeros((n_time, len(x_grid)))

    for t_idx, t in enumerate(times):
        for x_idx, xf in enumerate(x_grid):
            K = path_integral_propagator(params.x_start, xf, t, params)
            density[t_idx, x_idx] = abs(K) ** 2

        # Normalise each time slice
        total = np.trapezoid(density[t_idx], x_grid)
        if total > 1e-12:
            density[t_idx] /= total

    return times, x_grid, density


# ---------------------------------------------------------------------------
# First-passage time from path integrals
# ---------------------------------------------------------------------------


def path_integral_fpt(
    params: PathIntegralParams,
    boundary: float,
) -> tuple[np.ndarray, np.ndarray]:
    """First-passage time density from path integrals.

    Continuous analogue of quantum_walk_fpt_density.

    Args:
        params: path integral parameters.
        boundary: decision threshold position.

    Returns:
        times: (n_time,) time points.
        fpt_density: (n_time,) first-passage time density.
    """
    rng = np.random.RandomState(params.seed)
    n_time = max(int(params.T_max / params.dt), 2)
    times = np.linspace(params.dt, params.T_max, n_time)

    # Generate many paths and track when they first cross boundary
    n_steps = n_time
    increments = rng.randn(params.n_paths, n_steps) * np.sqrt(params.diffusion * params.dt)

    # Quantum phases for each path
    paths = np.zeros((params.n_paths, n_steps + 1))
    paths[:, 0] = params.x_start

    fpt_counts = np.zeros(n_steps)

    for t in range(n_steps):
        # Add drift + diffusion
        paths[:, t + 1] = paths[:, t] + params.drift * params.dt + increments[:, t]

        # Phase evolution (quantum interference)
        if params.hbar_eff > 0:
            phase_noise = rng.randn(params.n_paths) * params.hbar_eff * 0.1
            paths[:, t + 1] += phase_noise

    # Detect first crossing
    crossed = np.full(params.n_paths, False)
    for t in range(n_steps):
        newly_crossed = (~crossed) & (paths[:, t + 1] >= boundary)
        fpt_counts[t] = np.sum(newly_crossed)
        crossed = crossed | newly_crossed

    fpt_density = fpt_counts / params.n_paths
    return times, fpt_density


# ---------------------------------------------------------------------------
# Classical vs quantum comparison
# ---------------------------------------------------------------------------


def classical_vs_quantum_paths(
    params: PathIntegralParams,
    boundary: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare classical and quantum evidence accumulation.

    Classical: standard drift-diffusion (analytic Gaussian).
    Quantum: path integral with interference (ℏ_eff > 0).

    Continuous analogue of classical_vs_quantum_spreading in quantum_walk.py.

    Args:
        params: path integral parameters (uses hbar_eff for quantum version).
        boundary: optional decision threshold for FPT comparison.

    Returns:
        times: (n_time,)
        classical_density: (n_time, n_grid) Gaussian drift-diffusion.
        quantum_density: (n_time, n_grid) path integral result.
    """
    n_time = max(int(params.T_max / params.dt), 2)
    times = np.linspace(params.dt, params.T_max, n_time)

    x_range = 3.0 * params.diffusion * np.sqrt(params.T_max) + abs(params.drift) * params.T_max
    x_grid = np.linspace(params.x_start - x_range, params.x_start + x_range, 101)

    # Classical: analytic Gaussian
    classical_density = np.zeros((n_time, len(x_grid)))
    for t_idx, t in enumerate(times):
        mean = params.x_start + params.drift * t
        var = max(params.diffusion * t, 1e-12)
        classical_density[t_idx] = np.exp(-0.5 * (x_grid - mean) ** 2 / var) / np.sqrt(
            2 * np.pi * var
        )

    # Quantum: path integral
    _, _, quantum_density = evidence_accumulation_density(params, x_grid=x_grid)

    return times, classical_density, quantum_density
