"""Quantum interference circuits for decision models."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import pennylane as qml
from jax import Array


class InterferenceParams(NamedTuple):
    """Parameters for a quantum interference decision model."""

    n_outcomes: int  # number of decision outcomes
    gamma: float = 0.0  # interference strength (0 = classical)
    phase: Array | None = None  # per-path phase shifts


# ---------------------------------------------------------------------------
# Interference QNode
# ---------------------------------------------------------------------------


def make_interference_qnode(
    n_outcomes: int,
    device: str = "default.qubit",
    shots: int | None = None,
) -> qml.QNode:
    """Factory: QNode for an interference decision model.

    Circuit: prepare superposition over paths, apply phase rotations
    controlled by γ, measure outcome probabilities.

    Args:
        n_outcomes: number of outcomes (must be power of 2).
        device: PennyLane device string.
        shots: None for analytic.

    Returns:
        QNode callable(probs, gamma) → (n_outcomes,) outcome probs.
    """
    n_qubits = max(1, math.ceil(math.log2(n_outcomes)))
    dev = qml.device(device, wires=n_qubits, shots=shots)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def circuit(probs, gamma):
        # Encode probabilities as amplitudes
        amplitudes = jnp.sqrt(jnp.clip(probs, 0.0, None))
        amplitudes = amplitudes / jnp.linalg.norm(amplitudes)
        qml.MottonenStatePreparation(amplitudes, wires=wires)

        # Interference: Hadamard → phase → Hadamard creates observable interference
        for w in wires:
            qml.Hadamard(wires=w)
            qml.RZ(gamma, wires=w)
            qml.Hadamard(wires=w)

        return qml.probs(wires=wires)

    return circuit


# ---------------------------------------------------------------------------
# Conjunction fallacy circuit
# ---------------------------------------------------------------------------


def conjunction_fallacy_circuit(
    p_a: float,
    p_b: float,
    gamma: float,
    wires: tuple[int, int],
) -> None:
    """Two-qubit circuit for the conjunction fallacy (Tversky & Kahneman).

    Encodes marginal probabilities as single-qubit rotations, then adds
    quantum interference via an entangling RZZ gate controlled by γ.

    When γ ≠ 0, P(A∧B) can exceed P(A) — the conjunction fallacy.

    Args:
        p_a: marginal probability of event A.
        p_b: marginal probability of event B.
        gamma: interference angle.
        wires: two qubit indices.
    """
    # Encode marginals: RY(2 arcsin(√p)) maps |0⟩ to √(1-p)|0⟩ + √p|1⟩
    theta_a = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(p_a, 0.0, 1.0)))
    theta_b = 2.0 * jnp.arcsin(jnp.sqrt(jnp.clip(p_b, 0.0, 1.0)))

    qml.RY(theta_a, wires=wires[0])
    qml.RY(theta_b, wires=wires[1])

    # Entangle then rotate to create observable interference
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(gamma, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def make_conjunction_qnode(
    device: str = "default.qubit",
    shots: int | None = None,
) -> qml.QNode:
    """Factory: QNode for the conjunction fallacy experiment.

    Args:
        device: PennyLane device string.
        shots: None for analytic.

    Returns:
        QNode callable(p_a, p_b, gamma) → (4,) joint probs [00, 01, 10, 11].
    """
    dev = qml.device(device, wires=2, shots=shots)

    @qml.qnode(dev, interface="jax")
    def circuit(p_a, p_b, gamma):
        conjunction_fallacy_circuit(p_a, p_b, gamma, wires=[0, 1])
        return qml.probs(wires=[0, 1])

    return circuit
