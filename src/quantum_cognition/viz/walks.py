"""Visualization for quantum walks: probability evolution, spreading comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from jax import Array


def plot_qw_probability_evolution(
    probs: Array,
    times: list[int] | None = None,
    ax: plt.Axes | None = None,
    title: str = "Quantum Walk Probability Distribution",
) -> Figure:
    """Plot probability distribution snapshots at selected time steps.

    Args:
        probs: (n_steps + 1, n_sites) probability array from quantum_walk_evolution.
        times: Which time steps to plot. Defaults to 5 evenly spaced snapshots.
        ax: Matplotlib axes. Created if None.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    probs_np = np.asarray(probs)
    n_steps, n_sites = probs_np.shape
    center = n_sites // 2
    positions = np.arange(n_sites) - center

    if times is None:
        times = np.linspace(0, n_steps - 1, 5, dtype=int).tolist()

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    else:
        fig = ax.get_figure()

    for t in times:
        ax.plot(positions, probs_np[t], label=f"t = {t}", alpha=0.8)

    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_spreading_comparison(
    times: Array,
    classical_var: Array,
    quantum_var: Array,
    ax: plt.Axes | None = None,
) -> Figure:
    """Plot variance growth: classical O(t) vs quantum O(t²).

    Args:
        times: (n_steps + 1,) time array.
        classical_var: Classical RW variance.
        quantum_var: Quantum walk variance.
        ax: Matplotlib axes.

    Returns:
        The matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    t_np = np.asarray(times)
    ax.plot(t_np, np.asarray(classical_var), "b--", label="Classical RW: Var ∝ t", linewidth=2)
    ax.plot(t_np, np.asarray(quantum_var), "r-", label="Quantum Walk: Var ∝ t²", linewidth=2)
    ax.set_xlabel("Time steps")
    ax.set_ylabel("Variance")
    ax.set_title("Ballistic vs Diffusive Spreading")
    ax.legend()
    return fig


def plot_fpt_density(
    fpt: Array,
    ax: plt.Axes | None = None,
    title: str = "First-Passage Time Density (Quantum Walk)",
) -> Figure:
    """Plot first-passage time density showing interference fringes.

    Args:
        fpt: (n_steps,) FPT density array.
        ax: Matplotlib axes.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    fpt_np = np.asarray(fpt)
    t = np.arange(1, len(fpt_np) + 1)
    ax.bar(t, fpt_np, color="steelblue", alpha=0.7, width=0.8)
    ax.set_xlabel("Time step")
    ax.set_ylabel("FPT density")
    ax.set_title(title)
    return fig
