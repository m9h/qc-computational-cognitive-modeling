"""Bloch sphere visualization for single-qubit states."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from jax import Array


def bloch_coordinates(rho: Array) -> tuple[float, float, float]:
    """Extract Bloch sphere coordinates (x, y, z) from a single-qubit density matrix.

    ρ = (I + x σ_x + y σ_y + z σ_z) / 2

    Args:
        rho: (2, 2) density matrix.

    Returns:
        (x, y, z) Bloch vector components.
    """
    rho_np = np.asarray(rho)
    x = 2.0 * rho_np[0, 1].real
    y = 2.0 * rho_np[0, 1].imag  # note: σ_y has −i, so y = 2 Im(ρ_{01})
    z = (rho_np[0, 0] - rho_np[1, 1]).real
    return float(x), float(-y), float(z)


def plot_bloch_sphere(
    states: list[Array],
    labels: list[str] | None = None,
    title: str = "Bloch Sphere",
) -> Figure:
    """Plot one or more qubit states on the Bloch sphere.

    Args:
        states: List of (2, 2) density matrices.
        labels: Optional labels for each state.
        title: Plot title.

    Returns:
        Matplotlib Figure with 3D Bloch sphere.
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Draw wire sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.08, color="gray")

    # Draw axes
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], "k-", alpha=0.2)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], "k-", alpha=0.2)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], "k-", alpha=0.2)

    # Label poles
    ax.text(0, 0, 1.3, "|0⟩", ha="center", fontsize=12)
    ax.text(0, 0, -1.3, "|1⟩", ha="center", fontsize=12)
    ax.text(1.3, 0, 0, "|+⟩", ha="center", fontsize=10)
    ax.text(-1.3, 0, 0, "|−⟩", ha="center", fontsize=10)

    # Plot states
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(states), 10)))
    for i, rho in enumerate(states):
        x, y, z = bloch_coordinates(rho)
        label = labels[i] if labels else None
        ax.scatter([x], [y], [z], s=80, color=colors[i], label=label, zorder=5)
        # Arrow from origin
        ax.plot([0, x], [0, y], [0, z], color=colors[i], alpha=0.6, linewidth=1.5)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])

    if labels:
        ax.legend(loc="upper left")

    return fig
