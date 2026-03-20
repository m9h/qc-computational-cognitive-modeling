"""Spin glass Hamiltonians for social multi-agent systems.

Maps social interaction networks to spin Hamiltonians:
    H = -sum_{ij} J_ij Z_i Z_j  -  sum_i h_i Z_i  -  Gamma sum_i X_i

where:
    J_ij  = social coupling (positive: agreement incentive, negative: disagreement)
    h_i   = agent's private prior / external influence
    Gamma = transverse field (quantum exploration / bounded rationality)
    s_i   = agent opinion (+1/-1)

Supports:
    - Sherrington-Kirkpatrick (SK): fully connected, J_ij ~ N(0, 1/sqrt(N))
    - Edwards-Anderson (EA): nearest-neighbor on lattice, J_ij ~ N(0, 1) or ±J
    - Arbitrary social network topology with custom couplings
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class SocialSpinGlassParams(NamedTuple):
    """Parameters defining a social spin glass.

    Attributes:
        n_agents: Number of agents (spins).
        adjacency: (N, N) binary adjacency matrix of the social network.
        J: (N, N) coupling matrix (can be disordered).
        fields: (N,) local fields (agent priors).
        transverse_field: Strength of quantum fluctuations (Gamma).
        temperature: Noise / bounded rationality.
        seed: Random seed for disorder realization.
    """

    n_agents: int
    adjacency: np.ndarray
    J: np.ndarray
    fields: np.ndarray
    transverse_field: float = 0.0
    temperature: float = 1.0
    seed: int = 42


def sk_couplings(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Sherrington-Kirkpatrick model: fully connected, Gaussian J_ij.

    J_ij ~ N(0, 1/sqrt(N)), symmetric, zero diagonal.
    Returns (adjacency, J).
    """
    rng = np.random.default_rng(seed)
    adjacency = np.ones((n, n)) - np.eye(n)
    J_upper = rng.normal(0, 1.0 / np.sqrt(n), size=(n, n))
    J = (J_upper + J_upper.T) / 2
    np.fill_diagonal(J, 0.0)
    return adjacency, J


def ea_couplings(
    n: int,
    topology: str = "square",
    disorder: str = "gaussian",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Edwards-Anderson model: nearest-neighbor on lattice, disordered J_ij.

    Args:
        n: Number of spins. For 'square' topology, uses L=int(sqrt(n)) with L^2 spins.
        topology: 'square' (2D), 'chain' (1D), or 'complete'.
        disorder: 'gaussian' (J ~ N(0,1)), 'bimodal' (J = ±1 uniform), or 'uniform' (J ~ U[-1,1]).
        seed: Random seed.

    Returns:
        (adjacency, J) — both (N, N) arrays.
    """
    rng = np.random.default_rng(seed)

    if topology == "square":
        L = int(np.sqrt(n))
        n = L * L  # enforce square
        adjacency = _square_lattice_adjacency(L)
    elif topology == "chain":
        adjacency = np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    elif topology == "complete":
        adjacency = np.ones((n, n)) - np.eye(n)
    else:
        raise ValueError(f"Unknown topology '{topology}'")

    # Generate disordered couplings on edges
    J_raw = np.zeros((n, n))
    rows, cols = np.where(np.triu(adjacency) > 0)
    n_bonds = len(rows)

    if disorder == "gaussian":
        values = rng.normal(0, 1.0, size=n_bonds)
    elif disorder == "bimodal":
        values = rng.choice([-1.0, 1.0], size=n_bonds)
    elif disorder == "uniform":
        values = rng.uniform(-1.0, 1.0, size=n_bonds)
    else:
        raise ValueError(f"Unknown disorder type '{disorder}'")

    for k, (i, j) in enumerate(zip(rows, cols)):
        J_raw[i, j] = values[k]
        J_raw[j, i] = values[k]

    return adjacency.astype(float), J_raw


def _square_lattice_adjacency(L: int) -> np.ndarray:
    """2D square lattice adjacency matrix with periodic boundary conditions."""
    N = L * L
    adj = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            site = i * L + j
            # Right neighbor
            adj[site, i * L + (j + 1) % L] = 1
            adj[i * L + (j + 1) % L, site] = 1
            # Down neighbor
            adj[site, ((i + 1) % L) * L + j] = 1
            adj[((i + 1) % L) * L + j, site] = 1
    return adj


def social_hamiltonian_classical(params: SocialSpinGlassParams, spins: np.ndarray) -> float:
    """Compute the classical energy H(s) for a spin configuration.

    H = -sum_{ij} J_ij s_i s_j - sum_i h_i s_i
    """
    interaction = -0.5 * np.sum(params.J * np.outer(spins, spins))
    field = -np.sum(params.fields * spins)
    return float(interaction + field)


def social_hamiltonian_pennylane(params: SocialSpinGlassParams):
    """Build a PennyLane Hamiltonian for VQE/QAOA.

    H = -sum_{ij} J_ij Z_i Z_j - sum_i h_i Z_i - Gamma sum_i X_i

    Returns:
        qml.Hamiltonian ready for qml.expval().
    """
    import pennylane as qml

    coeffs = []
    ops = []
    n = params.n_agents

    # ZZ interaction terms
    for i in range(n):
        for j in range(i + 1, n):
            if params.adjacency[i, j] > 0 and abs(params.J[i, j]) > 1e-12:
                coeffs.append(-float(params.J[i, j]))
                ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # Z field terms (agent priors)
    for i in range(n):
        if abs(params.fields[i]) > 1e-12:
            coeffs.append(-float(params.fields[i]))
            ops.append(qml.PauliZ(i))

    # X transverse field (quantum fluctuations)
    if params.transverse_field != 0:
        for i in range(n):
            coeffs.append(-float(params.transverse_field))
            ops.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, ops)


def frustration_index(adjacency: np.ndarray, J: np.ndarray) -> float:
    """Fraction of frustrated triangles in the social network.

    A triangle (i,j,k) is frustrated if the product J_ij * J_jk * J_ik < 0.
    Returns a value in [0, 1]; 0 = no frustration, 1 = maximally frustrated.
    """
    n = adjacency.shape[0]
    n_triangles = 0
    n_frustrated = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 0:
                continue
            for k in range(j + 1, n):
                if adjacency[j, k] == 0 or adjacency[i, k] == 0:
                    continue
                n_triangles += 1
                if J[i, j] * J[j, k] * J[i, k] < 0:
                    n_frustrated += 1
    return n_frustrated / max(n_triangles, 1)
