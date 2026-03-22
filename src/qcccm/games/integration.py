"""Bridge between games/agreement and spin_glass solvers.

Maps the multi-agent agreement problem to a SocialSpinGlassParams
instance, enabling classical Metropolis, transverse-field MC,
VQE, and QAOA solutions. Compares classical vs quantum approaches
to finding agreement in frustrated social networks.

This connects:
- games/agreement.py: Ising-style multi-agent agreement dynamics
- spin_glass/hamiltonians.py: SocialSpinGlassParams, energy functions
- spin_glass/solvers.py: Metropolis, TFMC, VQE, QAOA
- spin_glass/order_params.py: q_EA, overlap, Binder cumulant

The key question: can quantum solvers (VQE/QAOA/transverse-field MC)
find better agreement configurations than classical Metropolis in
frustrated social networks? This tests Nayebi's (2025) conjecture
that quantum-enhanced agreement is achievable.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from qcccm.games.agreement import AgreementParams, _generate_couplings
from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    frustration_index,
)
from qcccm.spin_glass.order_params import edwards_anderson_q
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,
    transverse_field_mc,
)


class AgreementBenchmarkResult(NamedTuple):
    """Comparison of classical vs quantum agreement solving."""

    n_agents: int
    frustration: float
    frustration_index_value: float
    classical_energy: float
    classical_spins: np.ndarray
    quantum_energy: float
    quantum_spins: np.ndarray
    quantum_method: str
    classical_q_ea: float  # Edwards-Anderson parameter
    quantum_q_ea: float
    energy_improvement: float  # (classical - quantum) / |classical|


def agreement_to_spin_glass(
    params: AgreementParams,
    objective_index: int = 0,
) -> SocialSpinGlassParams:
    """Convert an agreement problem to a spin glass Hamiltonian.

    Maps the multi-agent agreement problem for a single binary objective
    into a spin glass where:
    - Each agent is a spin s_i = ±1 (opinion on the objective)
    - Couplings J_ij come from the agreement coupling matrix
    - Local fields h_i come from the agent's private prior
    - Temperature = agreement temperature (bounded rationality)

    Args:
        params: agreement problem parameters.
        objective_index: which binary objective to optimise (0-indexed).

    Returns:
        SocialSpinGlassParams ready for spin glass solvers.
    """
    rng = np.random.RandomState(params.seed)
    J = _generate_couplings(params.n_agents, params.frustration, rng)

    # Private priors as local fields
    fields = rng.randn(params.n_agents) * 0.5

    return SocialSpinGlassParams(
        n_agents=params.n_agents,
        adjacency=(np.abs(J) > 0).astype(float),
        J=J.astype(float),
        fields=fields.astype(float),
        transverse_field=0.0,  # set by caller for quantum methods
        temperature=params.temperature,
        seed=params.seed,
    )


def benchmark_agreement_solvers(
    n_agents: int = 10,
    frustration: float = 0.3,
    temperature: float = 1.0,
    transverse_field: float = 1.0,
    n_steps: int = 3000,
    seed: int = 42,
) -> AgreementBenchmarkResult:
    """Run classical vs quantum solvers on a frustrated agreement problem.

    Args:
        n_agents: number of agents.
        frustration: fraction of anti-ferromagnetic bonds.
        temperature: bounded rationality parameter.
        transverse_field: quantum fluctuation strength for TFMC.
        n_steps: MC steps per solver.
        seed: random seed.

    Returns:
        AgreementBenchmarkResult comparing the approaches.
    """
    params = AgreementParams(
        n_agents=n_agents,
        n_objectives=1,
        frustration=frustration,
        temperature=temperature,
        seed=seed,
    )

    sg_params = agreement_to_spin_glass(params)

    # Measure frustration
    f_index = frustration_index(sg_params.adjacency, sg_params.J)

    # Classical Metropolis
    classical_result = metropolis_spin_glass(
        sg_params, n_steps=n_steps, n_equilibrate=n_steps // 3,
    )

    # Quantum: transverse-field MC
    sg_params_quantum = sg_params._replace(transverse_field=transverse_field)
    quantum_result = transverse_field_mc(
        sg_params_quantum, n_steps=n_steps, n_equilibrate=n_steps // 3,
    )

    # Order parameters
    c_qea = edwards_anderson_q(classical_result.trajectory) if classical_result.trajectory is not None else 0.0
    q_qea = edwards_anderson_q(quantum_result.trajectory) if quantum_result.trajectory is not None else 0.0

    improvement = (classical_result.energy - quantum_result.energy) / max(abs(classical_result.energy), 1e-10)

    return AgreementBenchmarkResult(
        n_agents=n_agents,
        frustration=frustration,
        frustration_index_value=f_index,
        classical_energy=classical_result.energy,
        classical_spins=classical_result.spins,
        quantum_energy=quantum_result.energy,
        quantum_spins=quantum_result.spins,
        quantum_method=quantum_result.method,
        classical_q_ea=c_qea,
        quantum_q_ea=q_qea,
        energy_improvement=improvement,
    )


def frustration_sweep(
    n_agents: int = 10,
    frustrations: np.ndarray | None = None,
    temperature: float = 1.0,
    transverse_field: float = 1.0,
    n_steps: int = 3000,
    n_seeds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep frustration level and compare classical vs quantum energy.

    Returns:
        frustrations: (n_points,)
        classical_energies: (n_points,) mean best energy (classical)
        quantum_energies: (n_points,) mean best energy (quantum)
        improvements: (n_points,) mean relative improvement
    """
    if frustrations is None:
        frustrations = np.linspace(0.0, 0.5, 8)

    c_energies = np.zeros(len(frustrations))
    q_energies = np.zeros(len(frustrations))
    improvements = np.zeros(len(frustrations))

    for i, f in enumerate(frustrations):
        c_vals, q_vals = [], []
        for seed in range(n_seeds):
            result = benchmark_agreement_solvers(
                n_agents=n_agents, frustration=f,
                temperature=temperature, transverse_field=transverse_field,
                n_steps=n_steps, seed=seed,
            )
            c_vals.append(result.classical_energy)
            q_vals.append(result.quantum_energy)

        c_energies[i] = np.mean(c_vals)
        q_energies[i] = np.mean(q_vals)
        improvements[i] = np.mean(
            [(c - q) / max(abs(c), 1e-10) for c, q in zip(c_vals, q_vals)]
        )

    return frustrations, c_energies, q_energies, improvements
