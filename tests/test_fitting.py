"""Tests for the fitting module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qcccm.fitting.data import FitResult
from qcccm.fitting.likelihoods import (
    choice_log_likelihood,
    interference_log_likelihood,
)
from qcccm.fitting.mle import MLEConfig, fit_mle, model_comparison


class TestChoiceLogLikelihood:
    def test_perfect_prediction(self):
        """Perfect predictions should give the highest log-likelihood."""
        probs = jnp.array([[0.9, 0.1], [0.1, 0.9]])
        choices = jnp.array([0, 1])
        ll = choice_log_likelihood(probs, choices)
        assert jnp.isfinite(ll)
        assert float(ll) < 0  # log of prob < 1 is negative

    def test_worse_prediction_lower_ll(self):
        """Mismatched predictions should give lower log-likelihood."""
        choices = jnp.array([0, 0, 0])
        good_probs = jnp.array([[0.9, 0.1]] * 3)
        bad_probs = jnp.array([[0.1, 0.9]] * 3)
        ll_good = choice_log_likelihood(good_probs, choices)
        ll_bad = choice_log_likelihood(bad_probs, choices)
        assert float(ll_good) > float(ll_bad)

    def test_differentiable(self):
        """jax.grad should work through choice_log_likelihood."""
        choices = jnp.array([0, 1])

        def loss(logits):
            probs = jax.nn.softmax(logits, axis=1)
            return -choice_log_likelihood(probs, choices)

        logits = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        grad = jax.grad(loss)(logits)
        assert grad.shape == logits.shape
        assert jnp.all(jnp.isfinite(grad))


class TestInterferenceLogLikelihood:
    def test_gamma_zero_classical(self):
        """With γ=0, should give valid classical log-likelihood."""
        path_probs = jnp.array([[0.6, 0.2], [0.1, 0.1]])
        choices = jnp.array([0, 0, 1, 0])
        ll = interference_log_likelihood(jnp.array(0.0), path_probs, choices)
        assert jnp.isfinite(ll)

    def test_nonzero_gamma_different(self):
        """Non-zero γ should give different LL than γ=0."""
        path_probs = jnp.array([[0.4, 0.3], [0.2, 0.1]])
        choices = jnp.array([0, 1, 0])
        ll_0 = interference_log_likelihood(jnp.array(0.0), path_probs, choices)
        ll_1 = interference_log_likelihood(jnp.array(0.3), path_probs, choices)
        assert float(ll_0) != pytest.approx(float(ll_1), abs=1e-6)


class TestFitMLE:
    def test_recovers_known_minimum(self):
        """Should find the minimum of a simple quadratic NLL."""
        true_x = 2.0

        def nll(params):
            return (params[0] - true_x) ** 2

        result = fit_mle(
            nll,
            initial_params=jnp.array([0.0]),
            config=MLEConfig(n_restarts=1, compute_hessian=False),
        )
        assert result.converged
        assert float(result.params["x"][0]) == pytest.approx(true_x, abs=0.1)

    def test_fit_result_fields(self):
        """FitResult should have correct field types."""
        def nll(params):
            return params[0] ** 2

        result = fit_mle(
            nll,
            initial_params=jnp.array([1.0]),
            n_observations=10,
            config=MLEConfig(n_restarts=1, compute_hessian=True),
        )
        assert isinstance(result, FitResult)
        assert result.n_params == 1
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert result.hessian is not None
        assert result.hessian.shape == (1, 1)


class TestModelComparison:
    def test_lower_aic_preferred(self):
        """Model with lower AIC should be preferred."""
        r1 = FitResult(
            params={}, log_likelihood=-10.0, n_params=1,
            aic=22.0, bic=23.0, converged=True,
        )
        r2 = FitResult(
            params={}, log_likelihood=-20.0, n_params=2,
            aic=44.0, bic=46.0, converged=True,
        )
        comp = model_comparison([r1, r2], ["model_a", "model_b"])
        assert comp["best_model_aic"] == "model_a"

    def test_weights_sum_to_one(self):
        """Akaike weights should sum to 1."""
        r1 = FitResult(
            params={}, log_likelihood=-10.0, n_params=1,
            aic=22.0, bic=23.0, converged=True,
        )
        r2 = FitResult(
            params={}, log_likelihood=-15.0, n_params=1,
            aic=32.0, bic=33.0, converged=True,
        )
        comp = model_comparison([r1, r2], ["a", "b"])
        assert sum(comp["aic_weights"]) == pytest.approx(1.0, abs=1e-10)
        assert sum(comp["bic_weights"]) == pytest.approx(1.0, abs=1e-10)
