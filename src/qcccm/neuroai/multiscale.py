"""PennyLane circuits for multiscale neural dynamics simulation.

Implements encoding of neural activity, synaptic coupling, and
multi-step dynamics circuits, with support for hierarchical
micro-meso-macro coarse-graining.

The hierarchy:
- Micro: individual neurons (1 qubit each)
- Meso: cortical columns (groups of micro qubits, coarse-grained)
- Macro: brain regions (groups of meso qubits, further coarse-grained)

Coarse-graining uses partial trace to integrate out within-group
degrees of freedom, producing a reduced density matrix at each scale.

References:
    Wolff, A. et al. (2025). Quantum Computing for Neuroscience.
    Deco, G. et al. (2008). The Dynamic Brain: From Spiking Neurons
        to Neural Masses and Cortical Fields. PLoS Comp Bio.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax.numpy as jnp
import pennylane as qml
from jax import Array

from qcccm.core.density_matrix import partial_trace


class NeuralCircuitParams(NamedTuple):
    """Parameters for a neural dynamics quantum circuit."""

    n_neurons: int = 4
    n_layers: int = 2
    connectivity: str = "all_to_all"  # "all_to_all", "nearest_neighbor"
    noise_strength: float = 0.0  # depolarising noise per layer
    seed: int = 42


class MultiscaleHierarchy(NamedTuple):
    """Three-level hierarchy definition."""

    micro: NeuralCircuitParams
    meso: NeuralCircuitParams
    macro: NeuralCircuitParams
    micro_grouping: tuple[tuple[int, ...], ...]  # micro → meso mapping
    meso_grouping: tuple[tuple[int, ...], ...]  # meso → macro mapping


# ---------------------------------------------------------------------------
# Circuit building blocks
# ---------------------------------------------------------------------------


def neural_encoding_circuit(firing_rates: Array, wires: Sequence[int]) -> None:
    """Encode neural population activity as a quantum state.

    Each neuron's firing rate is encoded as a single-qubit rotation:
    RY(2 arcsin(√rate)) maps |0⟩ to √(1−rate)|0⟩ + √rate|1⟩.

    Args:
        firing_rates: (N,) firing rates in [0, 1], one per wire.
        wires: qubit indices.
    """
    for i, w in enumerate(wires):
        rate = jnp.clip(firing_rates[i], 1e-6, 1.0 - 1e-6)
        theta = 2.0 * jnp.arcsin(jnp.sqrt(rate))
        qml.RY(theta, wires=w)


def synaptic_coupling_circuit(
    weights: Array,
    wires: Sequence[int],
    connectivity: str = "all_to_all",
) -> None:
    """Parameterised entangling circuit representing synaptic connections.

    For each connected pair (i, j), applies CNOT-RY-CNOT sandwich
    with rotation angle proportional to synaptic weight.

    Args:
        weights: (N, N) synaptic weight matrix.
        wires: qubit indices.
        connectivity: "all_to_all" or "nearest_neighbor".
    """
    n = len(wires)
    for i in range(n):
        for j in range(i + 1, n):
            if connectivity == "nearest_neighbor" and abs(i - j) > 1:
                continue
            w_ij = weights[i, j]
            # CNOT-RY-CNOT controlled rotation
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.RY(w_ij, wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def neural_dynamics_circuit(
    firing_rates: Array,
    weights: Array,
    params: NeuralCircuitParams,
    wires: Sequence[int],
) -> None:
    """Full neural dynamics circuit: encode → evolve → (optional noise).

    Args:
        firing_rates: (N,) initial firing rates.
        weights: (N, N) synaptic weight matrix.
        params: circuit configuration.
        wires: qubit indices.
    """
    neural_encoding_circuit(firing_rates, wires)

    for _ in range(params.n_layers):
        synaptic_coupling_circuit(weights, wires, params.connectivity)
        if params.noise_strength > 0:
            for w in wires:
                qml.DepolarizingChannel(params.noise_strength, wires=w)


# ---------------------------------------------------------------------------
# QNode factory
# ---------------------------------------------------------------------------


def make_neural_qnode(
    params: NeuralCircuitParams,
    device: str | None = None,
    shots: int | None = None,
) -> qml.QNode:
    """Factory: create a QNode for neural dynamics simulation.

    Args:
        params: circuit parameters.
        device: PennyLane device. Auto-selects default.mixed if noise > 0.
        shots: None for analytic.

    Returns:
        QNode callable(firing_rates, weights) → (2^N,) probs.
    """
    n = params.n_neurons
    if device is None:
        device = "default.mixed" if params.noise_strength > 0 else "default.qubit"

    dev = qml.device(device, wires=n, shots=shots)
    wires = list(range(n))

    @qml.qnode(dev, interface="jax")
    def circuit(firing_rates, weights):
        neural_dynamics_circuit(firing_rates, weights, params, wires)
        return qml.probs(wires=wires)

    return circuit


# ---------------------------------------------------------------------------
# State tomography
# ---------------------------------------------------------------------------


def neural_state_tomography(
    qnode: qml.QNode,
    firing_rates: Array,
    weights: Array,
    n_qubits: int,
) -> Array:
    """Reconstruct density matrix from circuit output probabilities.

    For small systems, uses the probability vector to construct
    a diagonal density matrix (classical snapshot). For full quantum
    tomography, use qml.state() instead.

    Args:
        qnode: neural dynamics QNode.
        firing_rates: input firing rates.
        weights: synaptic weights.
        n_qubits: number of qubits.

    Returns:
        (2^N, 2^N) density matrix.
    """
    probs = qnode(firing_rates, weights)
    # Construct density matrix from measurement probabilities
    # This gives the diagonal (classical) part — full tomography
    # would require multiple measurement bases
    return jnp.diag(probs.astype(jnp.complex64))


# ---------------------------------------------------------------------------
# Multiscale hierarchy
# ---------------------------------------------------------------------------


def multiscale_hierarchy(
    n_micro: int = 8,
    n_meso: int = 4,
    n_macro: int = 2,
) -> MultiscaleHierarchy:
    """Define a three-level neural hierarchy with coarse-graining.

    Micro qubits are grouped into meso units, which are grouped
    into macro units. Each level has its own NeuralCircuitParams.

    Args:
        n_micro: number of micro-level neurons (individual).
        n_meso: number of meso-level units (columns).
        n_macro: number of macro-level units (regions).

    Returns:
        MultiscaleHierarchy with params and groupings.
    """
    # Group micro qubits into meso units
    per_meso = n_micro // n_meso
    micro_grouping = tuple(
        tuple(range(i * per_meso, (i + 1) * per_meso))
        for i in range(n_meso)
    )

    # Group meso qubits into macro units
    per_macro = n_meso // n_macro
    meso_grouping = tuple(
        tuple(range(i * per_macro, (i + 1) * per_macro))
        for i in range(n_macro)
    )

    return MultiscaleHierarchy(
        micro=NeuralCircuitParams(n_neurons=n_micro, connectivity="nearest_neighbor"),
        meso=NeuralCircuitParams(n_neurons=n_meso, connectivity="all_to_all"),
        macro=NeuralCircuitParams(n_neurons=n_macro, connectivity="all_to_all"),
        micro_grouping=micro_grouping,
        meso_grouping=meso_grouping,
    )


def coarse_grain(
    rho: Array,
    n_qubits: int,
    grouping: tuple[tuple[int, ...], ...],
) -> Array:
    """Coarse-grain a density matrix by tracing out within-group qubits.

    For each group, keeps one representative qubit and traces out
    the rest. The result is a density matrix on the coarser scale.

    Args:
        rho: (2^N, 2^N) fine-scale density matrix.
        n_qubits: number of qubits in rho.
        grouping: tuple of qubit groups, e.g. ((0,1), (2,3)).

    Returns:
        (2^M, 2^M) coarse-grained density matrix, M = len(grouping).
    """
    # Identify qubits to trace out (all but first in each group)
    keep = set()
    for group in grouping:
        keep.add(group[0])  # keep first qubit of each group

    trace_out = sorted(set(range(n_qubits)) - keep, reverse=True)

    result = rho
    current_n = n_qubits
    for qubit_idx in trace_out:
        dims = (2,) * current_n
        result = partial_trace(result, dims, trace_out=qubit_idx)
        current_n -= 1

    return result
