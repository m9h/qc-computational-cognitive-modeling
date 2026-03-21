"""Autoresearch experiment — THIS FILE IS MODIFIED BY THE AI AGENT.

Experiment 2: Sweep temperature and transverse field on SK model
to find the regime where PIMC outperforms Metropolis.

Hypothesis: At intermediate T (near T_g), Metropolis gets trapped in
metastable states while PIMC can tunnel through barriers. The advantage
should peak where frustration is high and T is just above the freezing
temperature.
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
    """Sweep T and Gamma across multiple disorder seeds on SK and EA models."""

    commit = get_commit_hash()
    n_mc_steps = 5000
    n_trotter = 8

    # ===== SWEEP CONFIGURATION (MODIFY THIS) =====

    configs = [
        # (model, N, topology, disorder, temperatures, gammas, seeds)
        ("SK", 10, "complete", "gaussian",
         [0.1, 0.3, 0.5, 0.8, 1.2, 2.0],
         [0.3, 0.7, 1.0, 1.5, 2.5],
         [42, 123, 7, 999, 31]),

        ("EA_bimodal", 16, "square", "bimodal",
         [0.3, 0.5, 0.8, 1.2],
         [0.5, 1.0, 2.0],
         [42, 123, 7]),
    ]

    best_advantage = -1.0
    best_config = ""

    for model_name, N, topology, disorder, temps, gammas, seeds in configs:
        for seed in seeds:
            # Generate couplings
            if model_name == "SK":
                adj, J = sk_couplings(N, seed=seed)
            else:
                adj, J = ea_couplings(N, topology=topology, disorder=disorder, seed=seed)

            fields = np.zeros(adj.shape[0])
            actual_N = adj.shape[0]
            fi = frustration_index(adj, J)

            # Exact ground state (once per disorder realization)
            params_ref = SocialSpinGlassParams(
                n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                temperature=0.1, seed=seed,
            )
            try:
                E_exact, _ = exact_ground_state(params_ref)
            except ValueError:
                # Too large for brute force — use best Metropolis at very low T
                cold_params = SocialSpinGlassParams(
                    n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                    temperature=0.01, seed=seed,
                )
                cold_result, _ = run_classical(cold_params, n_steps=20000)
                E_exact = cold_result.energy

            for T in temps:
                # Classical baseline
                params_c = SocialSpinGlassParams(
                    n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                    transverse_field=0.0, temperature=T, seed=seed,
                )
                c_result, c_time = run_classical(params_c, n_steps=n_mc_steps)

                q_EA_c = 0.0
                if c_result.trajectory is not None:
                    q_EA_c = edwards_anderson_q(c_result.trajectory)

                for Gamma in gammas:
                    # PIMC
                    params_q = SocialSpinGlassParams(
                        n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                        transverse_field=Gamma, temperature=T, seed=seed,
                    )
                    q_result, q_time = run_pimc(
                        params_q, n_trotter=n_trotter, n_steps=n_mc_steps,
                    )

                    q_EA_q = 0.0
                    if q_result.trajectory is not None:
                        q_EA_q = edwards_anderson_q(q_result.trajectory)

                    advantage = compute_quantum_advantage(
                        c_result.energy, q_result.energy, E_exact,
                    )

                    desc = (f"{model_name} N={actual_N} T={T} Gamma={Gamma} "
                            f"seed={seed} frust={fi:.2f}")

                    result = ExperimentResult(
                        commit=commit, description=desc,
                        model=model_name, topology=topology, disorder=disorder,
                        n_agents=actual_N, temperature=T,
                        transverse_field=Gamma,
                        method="pimc", E_best=q_result.energy, E_exact=E_exact,
                        quantum_advantage=advantage, q_EA=q_EA_q,
                        wall_time=q_time, frustration_index=fi,
                        magnetization=abs(np.mean(q_result.spins)), seed=seed,
                    )
                    print_result(result)
                    log_result(result)

                    if advantage > best_advantage:
                        best_advantage = advantage
                        best_config = desc

                    # Also log the classical baseline (once per T)
                    if Gamma == gammas[0]:
                        c_res = ExperimentResult(
                            commit=commit, description=f"classical: {desc}",
                            model=model_name, topology=topology, disorder=disorder,
                            n_agents=actual_N, temperature=T,
                            transverse_field=0.0,
                            method="metropolis", E_best=c_result.energy,
                            E_exact=E_exact, quantum_advantage=0.0,
                            q_EA=q_EA_c, wall_time=c_time,
                            frustration_index=fi,
                            magnetization=abs(np.mean(c_result.spins)),
                            seed=seed,
                        )
                        log_result(c_res)

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Best quantum advantage: {best_advantage:.6f}")
    print(f"Best config: {best_config}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_experiment()
