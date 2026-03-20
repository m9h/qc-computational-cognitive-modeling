"""Optional Qiskit circuit export utilities."""

from __future__ import annotations

import pennylane as qml


# ---------------------------------------------------------------------------
# Circuit analysis (no Qiskit required)
# ---------------------------------------------------------------------------


def circuit_depth_report(qnode: qml.QNode, *args) -> dict:
    """Report gate count and circuit depth for a traced QNode.

    Args:
        qnode: PennyLane QNode to analyse.
        *args: arguments to trace the circuit with.

    Returns:
        Dict with n_qubits, depth, n_gates, gate_types.
    """
    specs = qml.specs(qnode)(*args)
    res = specs.resources
    return {
        "n_qubits": specs.num_device_wires,
        "depth": res.depth,
        "n_gates": sum(res.gate_sizes.values()) if hasattr(res, "gate_sizes") else 0,
        "gate_types": dict(res.gate_types) if hasattr(res, "gate_types") else {},
        "resources": specs,
    }


# ---------------------------------------------------------------------------
# Qiskit export (optional dependency)
# ---------------------------------------------------------------------------


def pennylane_to_qiskit(qnode: qml.QNode, *args):
    """Convert a PennyLane QNode to a Qiskit QuantumCircuit.

    Requires the optional [ibm] dependency (qiskit).

    Args:
        qnode: PennyLane QNode to convert.
        *args: arguments to trace the circuit with.

    Returns:
        qiskit.QuantumCircuit equivalent.

    Raises:
        ImportError: if qiskit is not installed.
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
    except ImportError:
        raise ImportError(
            "Qiskit is required for circuit export. "
            "Install with: uv pip install 'qcccm[ibm]'"
        ) from None

    matrix = qml.matrix(qnode)(*args)
    n_qubits = int(matrix.shape[0]).bit_length() - 1
    qc = QuantumCircuit(n_qubits)
    qc.unitary(Operator(matrix), range(n_qubits))
    return qc
