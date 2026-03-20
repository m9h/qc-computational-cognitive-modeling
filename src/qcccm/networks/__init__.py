"""Multi-agent quantum cognitive networks."""

from qcccm.networks.topology import (
    NetworkTopology as NetworkTopology,
    adjacency_to_stochastic as adjacency_to_stochastic,
    complete_graph as complete_graph,
    random_graph as random_graph,
    ring_graph as ring_graph,
    star_graph as star_graph,
)
from qcccm.networks.multi_agent import (
    MultiAgentState as MultiAgentState,
    NetworkEvolutionParams as NetworkEvolutionParams,
    init_network_state as init_network_state,
    network_evolution as network_evolution,
    network_evolution_step as network_evolution_step,
)
from qcccm.networks.observables import (
    belief_polarization as belief_polarization,
    mean_pairwise_fidelity as mean_pairwise_fidelity,
    network_coherence as network_coherence,
    network_entropy as network_entropy,
    quantum_vs_classical_consensus as quantum_vs_classical_consensus,
)
