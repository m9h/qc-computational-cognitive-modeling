"""PennyLane circuit constructors for quantum cognition models."""

from qcccm.circuits.templates import (
    CircuitParams as CircuitParams,
    amplitude_encoding_circuit as amplitude_encoding_circuit,
    coin_operator_circuit as coin_operator_circuit,
    szegedy_walk_circuit as szegedy_walk_circuit,
    variational_layer as variational_layer,
)
from qcccm.circuits.belief_circuits import (
    belief_update_circuit as belief_update_circuit,
    make_belief_qnode as make_belief_qnode,
)
from qcccm.circuits.interference import (
    InterferenceParams as InterferenceParams,
    conjunction_fallacy_circuit as conjunction_fallacy_circuit,
    make_conjunction_qnode as make_conjunction_qnode,
    make_interference_qnode as make_interference_qnode,
)
from qcccm.circuits.export import (
    circuit_depth_report as circuit_depth_report,
    pennylane_to_qiskit as pennylane_to_qiskit,
)
