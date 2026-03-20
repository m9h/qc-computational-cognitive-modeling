"""Spin glass order parameters for multi-agent systems.

These diagnostics distinguish:
- Paramagnet (m=0, q_EA=0): agents fluctuate randomly
- Ferromagnet (m>0, q_EA>0): agents agree (consensus)
- Spin glass (m≈0, q_EA>0): agents frozen in disordered pattern (stuck disagreement)

The spin glass phase is the interesting one for social science: agents are stuck
in locally stable but globally suboptimal configurations, resistant to perturbation.
"""

from __future__ import annotations

import numpy as np


def edwards_anderson_q(spin_trajectory: np.ndarray) -> float:
    """Edwards-Anderson order parameter: q_EA = (1/N) sum_i <s_i>_t^2.

    Measures the degree of local freezing. Positive when individual spins
    have a preferred orientation over time, even if the global magnetization
    is zero (the spin glass signature).

    Args:
        spin_trajectory: (n_snapshots, n_agents) array of spin configurations.

    Returns:
        q_EA in [0, 1]. 0 = paramagnetic, 1 = fully frozen.
    """
    time_avg = np.mean(spin_trajectory, axis=0)  # <s_i>_t for each agent
    return float(np.mean(time_avg**2))


def overlap(config_a: np.ndarray, config_b: np.ndarray) -> float:
    """Replica overlap between two spin configurations.

    q_ab = (1/N) sum_i s_i^a s_i^b

    In physics, this measures similarity between two equilibrium states
    found from different random initializations (replicas).
    In social science, it measures agreement between two independent
    simulation runs of the same social network.

    Args:
        config_a, config_b: (N,) spin configurations.

    Returns:
        q in [-1, 1]. +1 = identical, -1 = anti-aligned, 0 = uncorrelated.
    """
    return float(np.mean(config_a * config_b))


def overlap_distribution(
    trajectories: list[np.ndarray],
    n_samples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Compute the overlap distribution P(q) from multiple replicas.

    The shape of P(q) distinguishes phases:
    - Paramagnet: single peak at q=0
    - Ferromagnet: two peaks at q=±m (magnetization)
    - Spin glass (replica symmetric): single peak at q=q_EA
    - Spin glass (RSB): broad/multi-peaked distribution

    Args:
        trajectories: List of (n_snapshots, N) arrays from independent replicas.
        n_samples: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        (n_samples,) array of overlap values.
    """
    rng = np.random.default_rng(seed)
    n_replicas = len(trajectories)
    overlaps = np.zeros(n_samples)

    for k in range(n_samples):
        # Pick two different replicas
        a, b = rng.choice(n_replicas, size=2, replace=False)
        # Pick a random time snapshot from each
        t_a = rng.integers(0, trajectories[a].shape[0])
        t_b = rng.integers(0, trajectories[b].shape[0])
        overlaps[k] = overlap(trajectories[a][t_a], trajectories[b][t_b])

    return overlaps


def glass_susceptibility(
    spin_trajectory: np.ndarray,
    temperature: float,
) -> float:
    """Spin glass susceptibility chi_SG = N * Var(q_EA) / T.

    Diverges at the glass transition T_g. Analogous to the magnetic
    susceptibility that diverges at T_c, but for the glass order parameter.

    Args:
        spin_trajectory: (n_snapshots, N) spin configurations.
        temperature: Current temperature.

    Returns:
        chi_SG (scalar).
    """
    N = spin_trajectory.shape[1]
    # Compute q_EA in time windows
    window_size = max(spin_trajectory.shape[0] // 10, 1)
    q_values = []
    for start in range(0, spin_trajectory.shape[0] - window_size, window_size):
        window = spin_trajectory[start : start + window_size]
        q_values.append(edwards_anderson_q(window))
    q_values = np.array(q_values)
    return float(N * np.var(q_values) / max(temperature, 1e-10))


def binder_cumulant(spin_trajectory: np.ndarray) -> float:
    """Binder cumulant U_L = 1 - <q^4> / (3 <q^2>^2).

    Dimensionless ratio that is size-independent at the critical temperature.
    Crossing of U_L curves for different system sizes pinpoints T_c or T_g.

    Args:
        spin_trajectory: (n_snapshots, N) spin configurations.

    Returns:
        U_L in [0, 2/3]. 0 = Gaussian, 2/3 = delta function.
    """
    time_avg = np.mean(spin_trajectory, axis=0)
    q2 = np.mean(time_avg**2)
    q4 = np.mean(time_avg**4)
    return float(1.0 - q4 / (3.0 * q2**2 + 1e-20))
