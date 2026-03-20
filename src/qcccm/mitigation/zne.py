"""Zero-Noise Extrapolation (ZNE) for noisy quantum circuits.

Wraps existing PennyLane circuits with error mitigation using mitiq.
Requires the optional [mitiq] dependency.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from jax import Array



def _check_mitiq():
    """Lazy import guard for mitiq."""
    try:
        import mitiq  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "mitiq is required for error mitigation. "
            "Install with: uv pip install 'qcccm[mitiq]'"
        ) from None


# ---------------------------------------------------------------------------
# Noisy device creation
# ---------------------------------------------------------------------------


def make_noisy_qnode(
    circuit_fn: Callable,
    n_qubits: int,
    noise_level: float = 0.01,
) -> qml.QNode:
    """Create a noisy QNode by inserting depolarising channels.

    Args:
        circuit_fn: PennyLane circuit function (no device/QNode wrapper).
        n_qubits: number of qubits.
        noise_level: depolarising probability per gate.

    Returns:
        Noisy QNode using the default.mixed device.
    """
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def noisy_circuit(*args, **kwargs):
        circuit_fn(*args, **kwargs)
        # Insert noise on all qubits
        for w in range(n_qubits):
            qml.DepolarizingChannel(noise_level, wires=w)
        return qml.probs(wires=list(range(n_qubits)))

    return noisy_circuit


# ---------------------------------------------------------------------------
# ZNE wrapper
# ---------------------------------------------------------------------------


def mitigate_expectation(
    executor: Callable[..., float],
    scale_factors: Sequence[float] = (1.0, 2.0, 3.0),
) -> Callable[..., float]:
    """Wrap a scalar-returning function with Zero-Noise Extrapolation.

    Uses Richardson extrapolation: run at multiple noise levels,
    fit polynomial, extrapolate to zero noise.

    This is a simplified ZNE that works without mitiq by implementing
    the core extrapolation directly. For full mitiq integration,
    use `mitigate_with_mitiq`.

    Args:
        executor: callable(*args) → scalar expectation value.
        scale_factors: noise scale factors (1.0 = nominal noise).

    Returns:
        Mitigated executor with the same signature.
    """
    def mitigated(*args, **kwargs):
        # Evaluate at each noise scale
        values = []
        for scale in scale_factors:
            val = executor(*args, noise_scale=scale, **kwargs)
            values.append(float(val))

        # Richardson extrapolation (linear fit to zero)
        scales = np.array(scale_factors)
        vals = np.array(values)

        if len(scales) == 2:
            # Linear extrapolation
            slope = (vals[1] - vals[0]) / (scales[1] - scales[0])
            return vals[0] - slope * scales[0]
        else:
            # Polynomial fit
            coeffs = np.polyfit(scales, vals, min(len(scales) - 1, 2))
            return float(np.polyval(coeffs, 0.0))

    return mitigated


def mitigated_belief_probs(
    prior: Array,
    likelihood: Array,
    n_qubits: int = 1,
    noise_levels: Sequence[float] = (0.01, 0.02, 0.03),
) -> Array:
    """Run belief update circuit at multiple noise levels and extrapolate.

    Implements ZNE by running the circuit at increasing depolarising
    noise levels and extrapolating to zero noise.

    Args:
        prior: (2^n,) prior probability vector.
        likelihood: (2^n,) likelihood weights.
        n_qubits: number of qubits.
        noise_levels: depolarising probabilities for ZNE.

    Returns:
        (2^n,) mitigated posterior probabilities.
    """
    from qcccm.circuits.belief_circuits import belief_update_circuit

    results = []
    wires = list(range(n_qubits))

    for p_noise in noise_levels:
        dev = qml.device("default.mixed", wires=n_qubits)

        @qml.qnode(dev, interface="jax")
        def noisy_circuit(prior_p, lik_p):
            belief_update_circuit(prior_p, lik_p, wires)
            for w in wires:
                qml.DepolarizingChannel(p_noise, wires=w)
            return qml.probs(wires=wires)

        probs = noisy_circuit(prior, likelihood)
        results.append(np.asarray(probs))

    # Richardson extrapolation per probability element
    noise_arr = np.array(noise_levels)
    results_arr = np.stack(results)  # (n_levels, n_outcomes)
    n_outcomes = results_arr.shape[1]

    mitigated = np.zeros(n_outcomes)
    for k in range(n_outcomes):
        coeffs = np.polyfit(noise_arr, results_arr[:, k], min(len(noise_levels) - 1, 2))
        mitigated[k] = np.polyval(coeffs, 0.0)

    # Clip and renormalise
    mitigated = np.clip(mitigated, 0.0, None)
    mitigated = mitigated / mitigated.sum()
    return jnp.array(mitigated)


# ---------------------------------------------------------------------------
# Full mitiq integration (optional)
# ---------------------------------------------------------------------------


def mitigate_with_mitiq(
    qnode: qml.QNode,
    *args,
    scale_factors: Sequence[int] = (1, 3, 5),
) -> np.ndarray:
    """Apply mitiq ZNE to a PennyLane QNode.

    Requires mitiq to be installed.

    Args:
        qnode: PennyLane QNode to mitigate.
        *args: arguments to pass to the QNode.
        scale_factors: unitary folding scale factors.

    Returns:
        Mitigated result.
    """
    _check_mitiq()

    from mitiq import zne
    from mitiq.zne.scaling import fold_global

    # Create a Cirq executor from the PennyLane QNode
    # mitiq works with Cirq circuits, so we use the matrix approach
    matrix = qml.matrix(qnode)(*args)

    import cirq

    n_qubits = int(np.log2(matrix.shape[0]))
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit(cirq.MatrixGate(np.asarray(matrix)).on(*qubits))

    def executor(circ):
        # Simulate the circuit
        sim = cirq.DensityMatrixSimulator()
        result = sim.simulate(circ)
        dm = result.final_density_matrix
        return float(np.real(dm[0, 0]))  # probability of |0...0⟩

    mitigated_val = zne.execute_with_zne(
        circuit, executor,
        scale_noise=fold_global,
        num_to_average=1,
    )
    return mitigated_val
