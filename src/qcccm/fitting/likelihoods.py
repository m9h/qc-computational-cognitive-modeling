"""Log-likelihood functions for quantum cognition models.

Each function returns a scalar log-likelihood suitable for use with
jax.grad and scipy.optimize.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.core.quantum_walk import QuantumWalkParams, quantum_walk_fpt_density


# ---------------------------------------------------------------------------
# General choice log-likelihood
# ---------------------------------------------------------------------------


@jax.jit
def choice_log_likelihood(predicted_probs: Array, choices: Array) -> Array:
    """Log-likelihood of observed choices given predicted probabilities.

    L = Σ_t log p(choice_t | params)

    Args:
        predicted_probs: (n_trials, n_options) predicted probabilities.
        choices: (n_trials,) observed choice indices.

    Returns:
        Scalar total log-likelihood.
    """
    selected = predicted_probs[jnp.arange(choices.shape[0]), choices]
    return jnp.sum(jnp.log(jnp.clip(selected, 1e-12, None)))


# ---------------------------------------------------------------------------
# Quantum walk reaction time log-likelihood
# ---------------------------------------------------------------------------


def quantum_walk_rt_log_likelihood(
    coin_angle: Array,
    observed_rts: Array,
    n_sites: int = 101,
    n_steps: int = 200,
    boundary_pos: int = 75,
    dt: float = 0.01,
) -> Array:
    """Log-likelihood of observed RTs under a quantum walk FPT model.

    Discretises observed RTs into time steps, then evaluates the FPT
    density of a quantum walk with a given coin angle.

    Args:
        coin_angle: scalar coin bias angle (the free parameter).
        observed_rts: (n_trials,) reaction times in seconds.
        n_sites: lattice size for the quantum walk.
        n_steps: maximum time steps.
        boundary_pos: position of absorbing boundary.
        dt: time discretisation step (seconds per lattice step).

    Returns:
        Scalar log-likelihood.
    """
    start = n_sites // 2
    params = QuantumWalkParams(
        coin_angle=coin_angle,
        n_sites=n_sites,
        n_steps=n_steps,
        start_pos=start,
        absorbing_right=boundary_pos,
    )
    fpt = quantum_walk_fpt_density(params, boundary="right")

    # Convert RTs to discrete time step indices
    step_indices = jnp.clip(
        jnp.round(observed_rts / dt).astype(jnp.int32) - 1, 0, n_steps - 1
    )
    likelihoods = fpt[step_indices]
    return jnp.sum(jnp.log(jnp.clip(likelihoods, 1e-12, None)))


# ---------------------------------------------------------------------------
# Interference model log-likelihood
# ---------------------------------------------------------------------------


def interference_log_likelihood(
    gamma: Array,
    path_probs: Array,
    observed_choices: Array,
) -> Array:
    """Log-likelihood for a quantum interference decision model.

    The model adds an interference term scaled by γ to classical
    path probabilities:

        P(outcome) = Σ_paths P_classical(path→outcome)
                     + 2γ Σ_{i<j} √(p_i p_j) cos(φ_ij)

    Simplified here assuming equal phases (cos=1) between all path pairs.

    Args:
        gamma: scalar interference strength (0 = classical).
        path_probs: (n_paths, n_outcomes) classical path→outcome probabilities.
        observed_choices: (n_trials,) observed choice indices.

    Returns:
        Scalar log-likelihood.
    """
    n_paths, n_outcomes = path_probs.shape

    # Classical total probability per outcome
    classical = jnp.sum(path_probs, axis=0)  # (n_outcomes,)

    # Interference terms: pairwise √(p_i * p_j) summed over path pairs
    interference = jnp.zeros(n_outcomes)
    for i in range(n_paths):
        for j in range(i + 1, n_paths):
            interference = interference + jnp.sqrt(path_probs[i] * path_probs[j])
    interference = 2.0 * gamma * interference

    # Total probability (clipped to valid range)
    total_probs = jnp.clip(classical + interference, 1e-12, None)
    total_probs = total_probs / jnp.sum(total_probs)  # renormalise

    selected = total_probs[observed_choices]
    return jnp.sum(jnp.log(jnp.clip(selected, 1e-12, None)))
