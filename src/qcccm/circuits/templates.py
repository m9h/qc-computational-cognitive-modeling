"""Parameterised PennyLane circuit templates for quantum cognition."""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax.numpy as jnp
import pennylane as qml
from jax import Array

from qcccm.models.bridge import stochastic_to_unitary


class CircuitParams(NamedTuple):
    """Configuration for circuit construction."""

    n_qubits: int
    n_layers: int = 1
    interface: str = "jax"


# ---------------------------------------------------------------------------
# Low-level circuit building blocks
# ---------------------------------------------------------------------------


def amplitude_encoding_circuit(probs: Array, wires: Sequence[int]) -> None:
    """Encode a probability vector into qubit amplitudes.

    Maps p = (p_0, …, p_{2^n − 1}) to |ψ⟩ = Σ_i √p_i |i⟩
    using Mottonen state preparation.

    Args:
        probs: (2^n,) probability vector summing to 1.
        wires: qubit indices to use.
    """
    amplitudes = jnp.sqrt(jnp.clip(probs, 0.0, None))
    # Normalise to unit vector (handles numerical noise)
    amplitudes = amplitudes / jnp.linalg.norm(amplitudes)
    qml.MottonenStatePreparation(amplitudes, wires=wires)


def variational_layer(params: Array, wires: Sequence[int]) -> None:
    """Single layer of a hardware-efficient variational ansatz.

    RY rotation on each qubit followed by a CNOT entangling ladder.

    Args:
        params: (n_qubits,) rotation angles.
        wires: qubit indices.
    """
    for i, w in enumerate(wires):
        qml.RY(params[i], wires=w)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])


def coin_operator_circuit(theta: float, wire: int) -> None:
    """Apply a biased coin operator as native gates.

    Circuit equivalent of core.quantum_walk.biased_coin(θ).
    Decomposes C(θ) into RY(2θ) · Z.

    Args:
        theta: coin bias angle (π/4 = Hadamard).
        wire: coin qubit index.
    """
    qml.RY(2.0 * theta, wires=wire)
    qml.PauliZ(wires=wire)


def szegedy_walk_circuit(
    B: Array,
    n_steps: int,
    wires: Sequence[int],
) -> None:
    """Implement n_steps of a Szegedy quantum walk as a circuit.

    Uses QubitUnitary with the matrix from bridge.stochastic_to_unitary.
    This is a fallback approach; structured decomposition for specific
    topologies can be added later.

    Args:
        B: (n, n) column-stochastic matrix.
        n_steps: number of walk steps.
        wires: qubits for the doubled Hilbert space (needs 2⌈log₂n⌉ qubits).
    """
    U = stochastic_to_unitary(B)
    for _ in range(n_steps):
        qml.QubitUnitary(U, wires=wires)
