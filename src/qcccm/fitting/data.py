"""Data containers for behavioral experiments."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class ChoiceData(NamedTuple):
    """Behavioral choice data for fitting quantum cognition models."""

    choices: Array  # (n_trials,) int array of chosen option indices
    n_options: int  # number of choice alternatives
    conditions: Array | None = None  # (n_trials,) condition labels
    reaction_times: Array | None = None  # (n_trials,) RTs in seconds


class FitResult(NamedTuple):
    """Result of a parameter estimation procedure."""

    params: dict[str, Array]  # fitted parameter values
    log_likelihood: float  # log-likelihood at optimum
    n_params: int
    aic: float  # Akaike information criterion
    bic: float  # Bayesian information criterion
    converged: bool
    hessian: Array | None = None  # (n_params, n_params) at optimum
    samples: Array | None = None  # (n_samples, n_params) posterior samples
