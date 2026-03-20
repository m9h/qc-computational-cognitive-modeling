"""Cognitive belief update circuits using PennyLane."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import pennylane as qml
from jax import Array

from qcccm.circuits.templates import amplitude_encoding_circuit


# ---------------------------------------------------------------------------
# Belief update circuit
# ---------------------------------------------------------------------------


def belief_update_circuit(
    prior: Array,
    likelihood: Array,
    wires: Sequence[int],
) -> None:
    """Apply belief update: encode prior, rotate by likelihood.

    Encodes prior as amplitude state, then applies a diagonal unitary
    derived from likelihood weights. In the computational basis measurement,
    this yields approximate Bayesian posteriors.

    Args:
        prior: (2^n,) prior probability vector summing to 1.
        likelihood: (2^n,) likelihood weights (will be normalised to phases).
        wires: qubit indices.
    """
    amplitude_encoding_circuit(prior, wires)

    # Convert likelihood to diagonal unitary phases
    safe_lik = jnp.clip(likelihood, 1e-12, None)
    log_lik = jnp.log(safe_lik)
    phases = log_lik - jnp.mean(log_lik)  # centre phases
    diag = jnp.exp(1j * phases)
    qml.DiagonalQubitUnitary(diag, wires=wires)


# ---------------------------------------------------------------------------
# QNode factories
# ---------------------------------------------------------------------------


def make_belief_qnode(
    n_qubits: int,
    device: str = "default.qubit",
    shots: int | None = None,
) -> qml.QNode:
    """Factory: create a QNode for belief update experiments.

    Returns a differentiable QNode with JAX interface that takes
    (prior, likelihood) and returns posterior probabilities.

    Args:
        n_qubits: number of qubits (2^n_qubits outcomes).
        device: PennyLane device string.
        shots: None for analytic, int for sampling.

    Returns:
        QNode callable(prior, likelihood) → (2^n_qubits,) probs.
    """
    dev = qml.device(device, wires=n_qubits, shots=shots)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def circuit(prior, likelihood):
        belief_update_circuit(prior, likelihood, wires)
        return qml.probs(wires=wires)

    return circuit
