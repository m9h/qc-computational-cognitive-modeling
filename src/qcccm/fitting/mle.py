"""Maximum likelihood estimation via JAX autodiff + scipy.optimize."""

from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.optimize import minimize

from qcccm.fitting.data import FitResult
from qcccm.fitting.likelihoods import (
    interference_log_likelihood,
    quantum_walk_rt_log_likelihood,
)


class MLEConfig(NamedTuple):
    """Configuration for MLE optimisation."""

    optimizer: str = "L-BFGS-B"
    max_iter: int = 500
    tol: float = 1e-6
    n_restarts: int = 5
    seed: int = 42
    compute_hessian: bool = True


# ---------------------------------------------------------------------------
# Core MLE fitting
# ---------------------------------------------------------------------------


def fit_mle(
    neg_log_likelihood: Callable[[Array], Array],
    initial_params: Array,
    bounds: Sequence[tuple[float, float]] | None = None,
    n_observations: int = 1,
    config: MLEConfig = MLEConfig(),
) -> FitResult:
    """Maximum likelihood estimation with JAX gradients and multi-start.

    Args:
        neg_log_likelihood: maps parameter array → negative log-likelihood.
            Must be JAX-differentiable.
        initial_params: (n_params,) starting point.
        bounds: per-parameter (lower, upper) bounds.
        n_observations: number of data points (for BIC).
        config: optimisation configuration.

    Returns:
        FitResult with fitted parameters, AIC, BIC.
    """
    grad_fn = jax.grad(neg_log_likelihood)
    n_params = initial_params.shape[0]
    rng = np.random.RandomState(config.seed)

    best_result = None
    best_nll = np.inf

    for i in range(config.n_restarts):
        if i == 0:
            x0 = np.asarray(initial_params, dtype=np.float64)
        else:
            # Random perturbation within bounds
            if bounds is not None:
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                x0 = rng.uniform(lo, hi)
            else:
                x0 = np.asarray(initial_params) + rng.randn(n_params) * 0.5

        def objective(x):
            return float(neg_log_likelihood(jnp.array(x)))

        def gradient(x):
            g = grad_fn(jnp.array(x))
            return np.asarray(g, dtype=np.float64)

        result = minimize(
            objective,
            x0,
            jac=gradient,
            method=config.optimizer,
            bounds=bounds,
            options={"maxiter": config.max_iter, "ftol": config.tol},
        )

        if result.fun < best_nll:
            best_nll = result.fun
            best_result = result

    ll = -best_nll
    aic = 2.0 * n_params - 2.0 * ll
    bic = n_params * np.log(max(n_observations, 1)) - 2.0 * ll

    hessian = None
    if config.compute_hessian:
        hess_fn = jax.hessian(neg_log_likelihood)
        hessian = hess_fn(jnp.array(best_result.x))

    return FitResult(
        params={"x": jnp.array(best_result.x)},
        log_likelihood=ll,
        n_params=n_params,
        aic=aic,
        bic=bic,
        converged=best_result.success,
        hessian=hessian,
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def fit_quantum_walk_to_rts(
    observed_rts: Array,
    n_sites: int = 101,
    n_steps: int = 200,
    config: MLEConfig = MLEConfig(),
) -> FitResult:
    """Fit quantum walk coin angle to observed reaction time data.

    Single free parameter: coin angle θ ∈ [0.05, π/2 − 0.05].

    Args:
        observed_rts: (n_trials,) reaction times.
        n_sites: lattice size.
        n_steps: maximum time steps.
        config: MLE configuration.

    Returns:
        FitResult with optimal coin_angle in params["x"][0].
    """
    def nll(params):
        return -quantum_walk_rt_log_likelihood(
            coin_angle=params[0],
            observed_rts=observed_rts,
            n_sites=n_sites,
            n_steps=n_steps,
        )

    result = fit_mle(
        nll,
        initial_params=jnp.array([jnp.pi / 4]),
        bounds=[(0.05, jnp.pi / 2 - 0.05)],
        n_observations=observed_rts.shape[0],
        config=config,
    )
    result = result._replace(params={"coin_angle": result.params["x"][0]})
    return result


def fit_interference_model(
    observed_choices: Array,
    path_probs: Array,
    config: MLEConfig = MLEConfig(),
) -> FitResult:
    """Fit interference parameter γ to choice data.

    Single free parameter: γ ∈ [−π, π].

    Args:
        observed_choices: (n_trials,) observed choices.
        path_probs: (n_paths, n_outcomes) classical path probabilities.
        config: MLE configuration.

    Returns:
        FitResult with optimal gamma in params["gamma"].
    """
    def nll(params):
        return -interference_log_likelihood(params[0], path_probs, observed_choices)

    result = fit_mle(
        nll,
        initial_params=jnp.array([0.0]),
        bounds=[(-jnp.pi, jnp.pi)],
        n_observations=observed_choices.shape[0],
        config=config,
    )
    result = result._replace(params={"gamma": result.params["x"][0]})
    return result


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------


def model_comparison(
    results: Sequence[FitResult],
    names: Sequence[str],
) -> dict:
    """Compare fitted models via AIC/BIC and Akaike weights.

    Args:
        results: list of FitResult from different models.
        names: model names.

    Returns:
        Dict with model_names, aic_values, bic_values, aic_weights,
        bic_weights, best_model_aic, best_model_bic.
    """
    aic_vals = np.array([r.aic for r in results])
    bic_vals = np.array([r.bic for r in results])

    # Akaike weights: w_i = exp(-Δ_i/2) / Σ exp(-Δ_j/2)
    delta_aic = aic_vals - aic_vals.min()
    aic_weights = np.exp(-delta_aic / 2)
    aic_weights = aic_weights / aic_weights.sum()

    delta_bic = bic_vals - bic_vals.min()
    bic_weights = np.exp(-delta_bic / 2)
    bic_weights = bic_weights / bic_weights.sum()

    return {
        "model_names": list(names),
        "aic_values": aic_vals.tolist(),
        "bic_values": bic_vals.tolist(),
        "aic_weights": aic_weights.tolist(),
        "bic_weights": bic_weights.tolist(),
        "best_model_aic": names[int(np.argmin(aic_vals))],
        "best_model_bic": names[int(np.argmin(bic_vals))],
    }
