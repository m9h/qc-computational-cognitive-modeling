"""NeuroAI: quantum computing for computational neuroscience.

Implements the conceptual framework from Wolff, Choquette, Northoff,
Iriki & Dumas (2025) "Quantum Computing for Neuroscience."
"""

from qcccm.neuroai.neural_states import (
    decode_neural_state as decode_neural_state,
    firing_rates_to_density_matrix as firing_rates_to_density_matrix,
    neural_entropy as neural_entropy,
    neural_fidelity_trajectory as neural_fidelity_trajectory,
    neural_mutual_information as neural_mutual_information,
)
from qcccm.neuroai.path_integral import (
    PathIntegralParams as PathIntegralParams,
    classical_action as classical_action,
    classical_vs_quantum_paths as classical_vs_quantum_paths,
    evidence_accumulation_density as evidence_accumulation_density,
    path_integral_fpt as path_integral_fpt,
    path_integral_propagator as path_integral_propagator,
)
from qcccm.neuroai.multiscale import (
    MultiscaleHierarchy as MultiscaleHierarchy,
    NeuralCircuitParams as NeuralCircuitParams,
    coarse_grain as coarse_grain,
    make_neural_qnode as make_neural_qnode,
    multiscale_hierarchy as multiscale_hierarchy,
    neural_dynamics_circuit as neural_dynamics_circuit,
    neural_encoding_circuit as neural_encoding_circuit,
    neural_state_tomography as neural_state_tomography,
    synaptic_coupling_circuit as synaptic_coupling_circuit,
)
from qcccm.neuroai.resource_estimation import (
    NeuralSystemSpec as NeuralSystemSpec,
    QubitEstimate as QubitEstimate,
    MolecularQubitEstimate as MolecularQubitEstimate,
    estimate_neural_qubits as estimate_neural_qubits,
    estimate_molecular_qubits as estimate_molecular_qubits,
    full_resource_table as full_resource_table,
)
from qcccm.neuroai.data_interface import (
    neural_data_to_density_matrix as neural_data_to_density_matrix,
    quantum_neural_analysis as quantum_neural_analysis,
    neural_state_fidelity_over_time as neural_state_fidelity_over_time,
    spike_trains_to_rates as spike_trains_to_rates,
    spike_raster_to_rates as spike_raster_to_rates,
    compute_correlations as compute_correlations,
)
