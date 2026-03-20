"""Spin glass models for multi-agent social simulation.

Provides the Sherrington-Kirkpatrick (SK) and Edwards-Anderson (EA) models
with both classical (Metropolis) and quantum (VQE/QAOA via PennyLane) solvers.
The mathematical isomorphism between disordered magnets and heterogeneous
multi-agent systems is explicit: J_ij = social coupling, s_i = agent opinion,
ground state = Nash equilibrium, T = bounded rationality.
"""

from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    social_hamiltonian_classical,
    social_hamiltonian_pennylane,
    sk_couplings,
    ea_couplings,
)
from qcccm.spin_glass.order_params import (
    edwards_anderson_q,
    overlap,
    overlap_distribution,
    glass_susceptibility,
    binder_cumulant,
)
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,
    transverse_field_mc,
    vqe_ground_state,
    qaoa_ground_state,
)
