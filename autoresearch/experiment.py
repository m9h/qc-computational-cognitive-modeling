"""Autoresearch experiment — THIS FILE IS MODIFIED BY THE AI AGENT.

Each experiment defines a sociophysics model, runs classical and quantum solvers,
computes the quantum advantage metric, and logs results.
"""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from prepare import (
    ExperimentResult,
    compute_quantum_advantage,
    exact_ground_state,
    get_commit_hash,
    log_result,
    print_result,
    run_classical,
    run_pimc,
)
from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    frustration_index,
    sk_couplings,
)
from qcccm.spin_glass.order_params import edwards_anderson_q


def run_experiment():
    """Run a single experiment comparing classical vs quantum solvers."""

    # ===== MODEL DEFINITION (MODIFY THIS) =====

    N = 8
    model_name = "SK"
    topology = "complete"
    disorder = "gaussian"
    temperature = 0.3
    transverse_field = 1.0
    seed = 42

    # Generate couplings
    adj, J = sk_couplings(N, seed=seed)
    fields = np.zeros(N)

    # ===== SOLVER CONFIGURATION (MODIFY THIS) =====

    n_mc_steps = 5000
    n_trotter = 8

    # ===== RUN SOLVERS =====

    # Classical baseline
    params_classical = SocialSpinGlassParams(
        n_agents=N, adjacency=adj, J=J, fields=fields,
        transverse_field=0.0, temperature=temperature, seed=seed,
    )
    classical_result, classical_time = run_classical(params_classical, n_steps=n_mc_steps)

    # Quantum: Path-integral Monte Carlo with transverse field
    params_quantum = SocialSpinGlassParams(
        n_agents=N, adjacency=adj, J=J, fields=fields,
        transverse_field=transverse_field, temperature=temperature, seed=seed,
    )
    quantum_result, quantum_time = run_pimc(
        params_quantum, n_trotter=n_trotter, n_steps=n_mc_steps,
    )

    # Exact ground state
    E_exact, exact_spins = exact_ground_state(params_classical)

    # ===== COMPUTE METRICS =====

    advantage = compute_quantum_advantage(
        classical_result.energy, quantum_result.energy, E_exact,
    )

    # Order parameters from classical trajectory
    q_EA_classical = 0.0
    if classical_result.trajectory is not None:
        q_EA_classical = edwards_anderson_q(classical_result.trajectory)

    q_EA_quantum = 0.0
    if quantum_result.trajectory is not None:
        q_EA_quantum = edwards_anderson_q(quantum_result.trajectory)

    # Frustration
    fi = frustration_index(adj, J)

    # Magnetization of best config
    mag_classical = abs(np.mean(classical_result.spins))

    # ===== LOG AND PRINT =====

    commit = get_commit_hash()
    description = (
        f"SK N={N} T={temperature} Gamma={transverse_field} "
        f"Trotter={n_trotter} seed={seed}"
    )

    # Classical result
    result_classical = ExperimentResult(
        commit=commit, description=f"classical: {description}",
        model=model_name, topology=topology, disorder=disorder,
        n_agents=N, temperature=temperature, transverse_field=0.0,
        method="metropolis", E_best=classical_result.energy, E_exact=E_exact,
        quantum_advantage=0.0, q_EA=q_EA_classical,
        wall_time=classical_time, frustration_index=fi,
        magnetization=mag_classical, seed=seed,
    )
    print_result(result_classical)
    log_result(result_classical)

    # Quantum result
    result_quantum = ExperimentResult(
        commit=commit, description=f"pimc: {description}",
        model=model_name, topology=topology, disorder=disorder,
        n_agents=N, temperature=temperature, transverse_field=transverse_field,
        method="pimc", E_best=quantum_result.energy, E_exact=E_exact,
        quantum_advantage=advantage, q_EA=q_EA_quantum,
        wall_time=quantum_time, frustration_index=fi,
        magnetization=abs(np.mean(quantum_result.spins)), seed=seed,
    )
    print_result(result_quantum)
    log_result(result_quantum)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: quantum_advantage = {advantage:.6f}")
    print(f"  Classical:  E = {classical_result.energy:.6f} ({classical_time:.2f}s)")
    print(f"  Quantum:    E = {quantum_result.energy:.6f} ({quantum_time:.2f}s)")
    print(f"  Exact:      E = {E_exact:.6f}")
    print(f"  Frustration index: {fi:.4f}")
    print(f"  q_EA classical: {q_EA_classical:.4f}")
    print(f"  q_EA quantum:   {q_EA_quantum:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_experiment()
