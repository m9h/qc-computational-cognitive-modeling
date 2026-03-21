"""Density matrix representations of neural population activity.

Maps N-neuron populations to N-qubit quantum states, where diagonal
elements encode firing rates and off-diagonal elements encode
inter-neuron correlations and coherences.

For N neurons with binary firing, the Hilbert space has dimension 2^N.
The density matrix ρ generalises the classical joint firing distribution:
- Diagonal: P(firing pattern k) = product of marginals (independent neurons)
- Off-diagonal: coherences encoding correlations beyond pairwise statistics

This connects to Wolff et al. (2025) "Quantum Computing for Neuroscience"
argument that quantum mathematical frameworks better accommodate
context-dependence and multi-pathway causation in neural dynamics.

References:
    Wolff, A., Choquette, A., Northoff, G., Iriki, A. & Dumas, G. (2025).
        Quantum Computing for Neuroscience. OSF Preprints.
    Schneidman, E. et al. (2006). Weak pairwise correlations imply strongly
        correlated network states in a neural population. Nature.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
from jax import Array

from qcccm.core.density_matrix import (
    fidelity,
    partial_trace,
    von_neumann_entropy,
)


# ---------------------------------------------------------------------------
# Construction: firing rates → density matrix
# ---------------------------------------------------------------------------


def firing_rates_to_density_matrix(
    rates: Array,
    correlations: Array | None = None,
) -> Array:
    """Convert neural firing rates to a density matrix.

    Maps N firing rates (marginal probabilities) to a 2^N × 2^N
    density matrix. Without correlations, assumes independent neurons
    (product state). With correlations, adds off-diagonal coherences.

    Args:
        rates: (N,) firing rates in [0, 1].
        correlations: (N, N) pairwise correlation matrix, entries in [-1, 1].
            Diagonal is ignored. None for independent neurons.

    Returns:
        (2^N, 2^N) density matrix with Tr(ρ) = 1.
    """
    rates = jnp.asarray(rates)
    n = rates.shape[0]

    if n > 8:
        warnings.warn(
            f"N={n} neurons creates a {2**n}×{2**n} density matrix. "
            "Consider mean-field approximation for N > 8.",
            stacklevel=2,
        )

    rates_clipped = jnp.clip(rates, 1e-12, 1.0 - 1e-12)

    # Compute product-state probabilities for each of 2^N basis states
    d = 2**n
    indices = jnp.arange(d)
    # Bit j of index k: 1 if neuron j fires in pattern k
    bits = (indices[:, None] >> jnp.arange(n)[None, :]) & 1  # (2^N, N)
    log_probs = (
        bits * jnp.log(rates_clipped) + (1 - bits) * jnp.log(1 - rates_clipped)
    )
    probs = jnp.exp(jnp.sum(log_probs, axis=1))  # (2^N,)
    probs = probs / jnp.sum(probs)  # normalise

    # Start with diagonal density matrix
    rho = jnp.diag(probs.astype(jnp.complex64))

    # Add correlations as off-diagonal coherences
    if correlations is not None:
        correlations = jnp.asarray(correlations)
        sqrt_p = jnp.sqrt(probs)

        # For each pair of basis states that differ in correlated neurons,
        # add coherence proportional to correlation strength
        for i_neuron in range(n):
            for j_neuron in range(i_neuron + 1, n):
                c = correlations[i_neuron, j_neuron]
                if abs(c) < 1e-10:
                    continue
                # Basis states that differ in neurons i and j
                # contribute coherence proportional to c * sqrt(p_k * p_l)
                flip_mask = (1 << i_neuron) | (1 << j_neuron)
                for k in range(d):
                    partner = k ^ flip_mask
                    if partner > k:
                        coherence = c * sqrt_p[k] * sqrt_p[partner]
                        rho = rho.at[k, partner].add(coherence.astype(jnp.complex64))
                        rho = rho.at[partner, k].add(coherence.astype(jnp.complex64))

        # Project to nearest valid density matrix (clip eigenvalues)
        eigvals, eigvecs = jnp.linalg.eigh(rho)
        eigvals = jnp.clip(eigvals.real, 0.0, None)
        rho = (eigvecs * eigvals[None, :]) @ jnp.conj(eigvecs).T
        rho = rho / jnp.trace(rho)

    return rho


# ---------------------------------------------------------------------------
# Analysis: entropy, mutual information, decoding
# ---------------------------------------------------------------------------


def neural_entropy(rho: Array) -> Array:
    """Von Neumann entropy of a neural population state.

    Quantifies uncertainty in neural activity. For a classical (diagonal)
    state, equals Shannon entropy of the firing pattern distribution.
    Coherences between neurons reduce entropy below the classical value.

    Args:
        rho: (2^N, 2^N) density matrix.

    Returns:
        Scalar entropy S(ρ) ≥ 0.
    """
    return von_neumann_entropy(rho)


def neural_mutual_information(
    rho: Array,
    n_neurons: int,
    partition: tuple[tuple[int, ...], tuple[int, ...]],
) -> Array:
    """Quantum mutual information between two neural subpopulations.

    I(A:B) = S(ρ_A) + S(ρ_B) − S(ρ_AB)

    Args:
        rho: (2^N, 2^N) full density matrix.
        n_neurons: total number of neurons N.
        partition: ((neurons_in_A), (neurons_in_B)) e.g. ((0,1), (2,3)).

    Returns:
        Scalar mutual information I(A:B) ≥ 0.
    """
    # Trace out neurons not in A to get ρ_A
    neurons_a, neurons_b = partition
    keep_a = set(neurons_a)
    keep_b = set(neurons_b)

    # Trace out complement of A (in reverse order to preserve indices)
    rho_a = rho
    traced_a = 0
    for k in range(n_neurons - 1, -1, -1):
        if k not in keep_a:
            current_dims = tuple(
                2 for j in range(n_neurons - traced_a)
            )
            rho_a = partial_trace(rho_a, current_dims, trace_out=k - traced_a if k >= traced_a else k)
            traced_a += 1

    # Trace out complement of B
    rho_b = rho
    traced_b = 0
    for k in range(n_neurons - 1, -1, -1):
        if k not in keep_b:
            current_dims = tuple(
                2 for j in range(n_neurons - traced_b)
            )
            rho_b = partial_trace(rho_b, current_dims, trace_out=k - traced_b if k >= traced_b else k)
            traced_b += 1

    return von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) - von_neumann_entropy(rho)


def decode_neural_state(rho: Array, n_neurons: int) -> tuple[Array, Array]:
    """Extract firing rates and pairwise correlations from a density matrix.

    Inverse of firing_rates_to_density_matrix: recovers marginal
    firing rates and pairwise correlation structure.

    Args:
        rho: (2^N, 2^N) density matrix.
        n_neurons: number of neurons N.

    Returns:
        rates: (N,) marginal firing rates.
        correlations: (N, N) pairwise correlations.
    """
    d = 2**n_neurons
    rates = jnp.zeros(n_neurons)
    correlations = jnp.zeros((n_neurons, n_neurons))

    # Firing rate for neuron j: Tr(ρ |1⟩⟨1|_j ⊗ I_rest)
    # = sum over basis states where neuron j fires
    diag = jnp.diag(rho).real
    indices = jnp.arange(d)

    for j in range(n_neurons):
        mask = ((indices >> j) & 1).astype(jnp.float32)
        rates = rates.at[j].set(jnp.sum(diag * mask))

    # Pairwise correlations: Cov(j,k) = <n_j n_k> - <n_j><n_k>
    for j in range(n_neurons):
        for k in range(j + 1, n_neurons):
            mask_jk = (((indices >> j) & 1) & ((indices >> k) & 1)).astype(jnp.float32)
            joint = jnp.sum(diag * mask_jk)
            cov = joint - rates[j] * rates[k]
            correlations = correlations.at[j, k].set(cov)
            correlations = correlations.at[k, j].set(cov)

    return rates, correlations


def neural_fidelity_trajectory(rho_trajectory: Array) -> Array:
    """Track state fidelity between consecutive time steps.

    Args:
        rho_trajectory: (T, d, d) density matrices over time.

    Returns:
        (T-1,) fidelity between consecutive states.
    """
    T = rho_trajectory.shape[0]
    fidelities = jnp.array([
        fidelity(rho_trajectory[t], rho_trajectory[t + 1])
        for t in range(T - 1)
    ])
    return fidelities
