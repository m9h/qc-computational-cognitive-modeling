"""
Systematic PIMC hyperparameter validation experiment.
Tests temperature sweep, Trotter convergence, classical limit validation,
and MC sweep ablation to diagnose quantum advantage regime.
"""

import numpy as np
import time
from prepare import (
    ExperimentResult,
    compute_quantum_advantage,
    exact_ground_state,
    get_commit_hash,
    log_result,
    print_result,
    run_classical,
    run_pimc,
    run_vqe,
    run_qaoa,
)
from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    sk_couplings,
    ea_couplings,
    frustration_index,
)
from qcccm.spin_glass.order_params import (
    edwards_anderson_q,
    overlap,
    binder_cumulant,
)


def make_params(N, T, Gamma, adj, J, seed):
    """Create SocialSpinGlassParams with given parameters."""
    return SocialSpinGlassParams(
        n_agents=N,
        adjacency=adj,
        J=J,
        fields=np.zeros(N),
        temperature=T,
        transverse_field=Gamma,
        seed=seed,
    )


def run_experiment():
    commit = get_commit_hash()
    N = 8
    seeds = [42, 123, 456]

    # -----------------------------------------------------------------------
    # Phase 1: Classical limit validation (Gamma=0)
    # Verify PIMC recovers classical solution at multiple temperatures
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Classical limit validation (Gamma=0)")
    print("=" * 60)

    temperatures_phase1 = [0.1, 0.5, 1.0, 5.0, 10.0]

    for seed in seeds[:2]:  # 2 seeds for phase 1 to save time
        adj, J = sk_couplings(N, seed)
        fi = frustration_index(adj, J)

        for T in temperatures_phase1:
            params_classical = make_params(N, T, 0.0, adj, J, seed)
            params_pimc_classical_limit = make_params(N, T, 0.0, adj, J, seed)

            # Run classical solver
            classical_result, wt_classical = run_classical(params_classical, n_steps=5000)

            # Run PIMC with Gamma=0 (classical limit)
            pimc_result, wt_pimc = run_pimc(params_pimc_classical_limit, n_trotter=16, n_steps=5000)

            # Get exact ground state
            E_exact, _ = exact_ground_state(params_classical)

            E_classical = classical_result.energy
            E_pimc = pimc_result.energy

            qa = compute_quantum_advantage(E_classical, E_pimc, E_exact)

            # q_EA from classical trajectory
            q_ea_cl = edwards_anderson_q(classical_result.trajectory) if hasattr(classical_result, 'trajectory') and classical_result.trajectory is not None else 0.0

            result = ExperimentResult(
                commit=commit,
                model="SK",
                topology="complete",
                disorder="gaussian",
                n_agents=N,
                temperature=T,
                transverse_field=0.0,
                frustration=fi,
                seed=seed,
                method="PIMC_classical_limit",
                E_best=E_pimc,
                E_exact=E_exact,
                quantum_advantage=qa,
                q_EA=q_ea_cl,
                wall_time=wt_pimc,
                status="ok",
                magnetization=float(np.mean(pimc_result.spins)) if pimc_result.spins is not None else 0.0,
                frustration_index=fi,
                binder=0.0,
                metadata={
                    "phase": "classical_limit_validation",
                    "E_classical": E_classical,
                    "E_pimc_gamma0": E_pimc,
                    "E_exact": E_exact,
                    "T": T,
                    "n_trotter": 16,
                    "n_sweeps": 5000,
                    "seed": seed,
                    "delta_pimc_vs_classical": float(E_pimc - E_classical),
                },
            )
            print_result(result)
            log_result(result)

    # -----------------------------------------------------------------------
    # Phase 2: Trotter convergence study
    # Fix T=0.5, Gamma=0.5, N=8, vary P in {8, 16, 32}
    # (skip 64 to stay within time budget)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 2: Trotter convergence study (T=0.5, Gamma=0.5, N=8)")
    print("=" * 60)

    T_trotter = 0.5
    Gamma_trotter = 0.5
    trotter_slices = [8, 16, 32]

    for seed in seeds[:2]:
        adj, J = sk_couplings(N, seed)
        fi = frustration_index(adj, J)

        params_classical = make_params(N, T_trotter, 0.0, adj, J, seed)
        classical_result, wt_classical = run_classical(params_classical, n_steps=5000)
        E_exact, _ = exact_ground_state(params_classical)
        E_classical = classical_result.energy

        for P in trotter_slices:
            params_pimc = make_params(N, T_trotter, Gamma_trotter, adj, J, seed)
            pimc_result, wt_pimc = run_pimc(params_pimc, n_trotter=P, n_steps=5000)

            E_pimc = pimc_result.energy
            qa = compute_quantum_advantage(E_classical, E_pimc, E_exact)

            result = ExperimentResult(
                commit=commit,
                model="SK",
                topology="complete",
                disorder="gaussian",
                n_agents=N,
                temperature=T_trotter,
                transverse_field=Gamma_trotter,
                frustration=fi,
                seed=seed,
                method="PIMC_trotter_sweep",
                E_best=E_pimc,
                E_exact=E_exact,
                quantum_advantage=qa,
                q_EA=0.0,
                wall_time=wt_pimc,
                status="ok",
                magnetization=float(np.mean(pimc_result.spins)) if pimc_result.spins is not None else 0.0,
                frustration_index=fi,
                binder=0.0,
                metadata={
                    "phase": "trotter_convergence",
                    "n_trotter": P,
                    "inv_P": 1.0 / P,
                    "E_classical": E_classical,
                    "E_pimc": E_pimc,
                    "E_exact": E_exact,
                    "T": T_trotter,
                    "Gamma": Gamma_trotter,
                    "seed": seed,
                },
            )
            print_result(result)
            log_result(result)

    # -----------------------------------------------------------------------
    # Phase 3: Temperature sweep with fine-grained Gamma sweep
    # T in {0.1, 0.5, 1.0, 5.0}, Gamma in {0.0, 0.2, 0.5, 0.8, 1.0}
    # N=8, 3 seeds, n_trotter=16 (balance accuracy vs speed)
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 3: Temperature x Gamma sweep (N=8)")
    print("=" * 60)

    temperatures_phase3 = [0.1, 0.5, 1.0, 5.0]
    gamma_values_phase3 = [0.0, 0.2, 0.5, 0.8, 1.0]
    n_trotter_phase3 = 16

    for seed in seeds:
        adj, J = sk_couplings(N, seed)
        fi = frustration_index(adj, J)

        for T in temperatures_phase3:
            # Run classical once per (T, seed)
            params_classical = make_params(N, T, 0.0, adj, J, seed)
            classical_result, wt_classical = run_classical(params_classical, n_steps=5000)
            E_exact, _ = exact_ground_state(params_classical)
            E_classical = classical_result.energy

            q_ea_cl = 0.0
            if hasattr(classical_result, 'trajectory') and classical_result.trajectory is not None:
                try:
                    q_ea_cl = edwards_anderson_q(classical_result.trajectory)
                except Exception:
                    q_ea_cl = 0.0

            for Gamma in gamma_values_phase3:
                params_pimc = make_params(N, T, Gamma, adj, J, seed)
                pimc_result, wt_pimc = run_pimc(params_pimc, n_trotter=n_trotter_phase3, n_steps=5000)

                E_pimc = pimc_result.energy
                qa = compute_quantum_advantage(E_classical, E_pimc, E_exact)

                q_ea_q = 0.0
                if hasattr(pimc_result, 'trajectory') and pimc_result.trajectory is not None:
                    try:
                        q_ea_q = edwards_anderson_q(pimc_result.trajectory)
                    except Exception:
                        q_ea_q = 0.0

                result = ExperimentResult(
                    commit=commit,
                    model="SK",
                    topology="complete",
                    disorder="gaussian",
                    n_agents=N,
                    temperature=T,
                    transverse_field=Gamma,
                    frustration=fi,
                    seed=seed,
                    method="PIMC_temp_gamma_sweep",
                    E_best=E_pimc,
                    E_exact=E_exact,
                    quantum_advantage=qa,
                    q_EA=q_ea_q,
                    wall_time=wt_pimc,
                    status="ok",
                    magnetization=float(np.mean(pimc_result.spins)) if pimc_result.spins is not None else 0.0,
                    frustration_index=fi,
                    binder=0.0,
                    metadata={
                        "phase": "temp_gamma_sweep",
                        "T": T,
                        "Gamma": Gamma,
                        "n_trotter": n_trotter_phase3,
                        "E_classical": E_classical,
                        "E_pimc": E_pimc,
                        "E_exact": E_exact,
                        "q_EA_classical": q_ea_cl,
                        "q_EA_quantum": q_ea_q,
                        "seed": seed,
                        "wt_classical": wt_classical,
                    },
                )
                print_result(result)
                log_result(result)

    # -----------------------------------------------------------------------
    # Phase 4: MC sweep ablation
    # Fix T=0.5, Gamma=0.5, N=8, P=16, vary n_sweeps in {1000, 5000, 10000}
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 4: MC sweep ablation (T=0.5, Gamma=0.5, N=8)")
    print("=" * 60)

    T_ablation = 0.5
    Gamma_ablation = 0.5
    sweep_counts = [1000, 5000, 10000]

    for seed in seeds[:2]:
        adj, J = sk_couplings(N, seed)
        fi = frustration_index(adj, J)

        params_classical = make_params(N, T_ablation, 0.0, adj, J, seed)
        classical_result, wt_classical = run_classical(params_classical, n_steps=5000)
        E_exact, _ = exact_ground_state(params_classical)
        E_classical = classical_result.energy

        for n_sweeps in sweep_counts:
            params_pimc = make_params(N, T_ablation, Gamma_ablation, adj, J, seed)
            pimc_result, wt_pimc = run_pimc(params_pimc, n_trotter=16, n_steps=n_sweeps)

            E_pimc = pimc_result.energy
            qa = compute_quantum_advantage(E_classical, E_pimc, E_exact)

            result = ExperimentResult(
                commit=commit,
                model="SK",
                topology="complete",
                disorder="gaussian",
                n_agents=N,
                temperature=T_ablation,
                transverse_field=Gamma_ablation,
                frustration=fi,
                seed=seed,
                method="PIMC_sweep_ablation",
                E_best=E_pimc,
                E_exact=E_exact,
                quantum_advantage=qa,
                q_EA=0.0,
                wall_time=wt_pimc,
                status="ok",
                magnetization=float(np.mean(pimc_result.spins)) if pimc_result.spins is not None else 0.0,
                frustration_index=fi,
                binder=0.0,
                metadata={
                    "phase": "sweep_ablation",
                    "n_sweeps": n_sweeps,
                    "n_trotter": 16,
                    "T": T_ablation,
                    "Gamma": Gamma_ablation,
                    "E_classical": E_classical,
                    "E_pimc": E_pimc,
                    "E_exact": E_exact,
                    "seed": seed,
                },
            )
            print_result(result)
            log_result(result)

    # -----------------------------------------------------------------------
    # Phase 5: Best regime follow-up — low T, optimal Gamma, larger N
    # Based on phase 3 findings, try N=10 at T=0.1, Gamma=0.5
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 5: Best regime follow-up (N=10, T=0.1, Gamma=0.5)")
    print("=" * 60)

    N_large = 10
    T_best = 0.1
    Gamma_best = 0.5

    for seed in seeds[:2]:
        adj10, J10 = sk_couplings(N_large, seed)
        fi10 = frustration_index(adj10, J10)

        params_cl10 = make_params(N_large, T_best, 0.0, adj10, J10, seed)
        classical_result10, wt_cl10 = run_classical(params_cl10, n_steps=5000)
        E_exact10, _ = exact_ground_state(params_cl10)
        E_classical10 = classical_result10.energy

        params_pimc10 = make_params(N_large, T_best, Gamma_best, adj10, J10, seed)
        pimc_result10, wt_pimc10 = run_pimc(params_pimc10, n_trotter=16, n_steps=5000)

        E_pimc10 = pimc_result10.energy
        qa10 = compute_quantum_advantage(E_classical10, E_pimc10, E_exact10)

        result = ExperimentResult(
            commit=commit,
            model="SK",
            topology="complete",
            disorder="gaussian",
            n_agents=N_large,
            temperature=T_best,
            transverse_field=Gamma_best,
            frustration=fi10,
            seed=seed,
            method="PIMC_best_regime",
            E_best=E_pimc10,
            E_exact=E_exact10,
            quantum_advantage=qa10,
            q_EA=0.0,
            wall_time=wt_pimc10,
            status="ok",
            magnetization=float(np.mean(pimc_result10.spins)) if pimc_result10.spins is not None else 0.0,
            frustration_index=fi10,
            binder=0.0,
            metadata={
                "phase": "best_regime_followup",
                "N": N_large,
                "T": T_best,
                "Gamma": Gamma_best,
                "n_trotter": 16,
                "E_classical": E_classical10,
                "E_pimc": E_pimc10,
                "E_exact": E_exact10,
                "seed": seed,
            },
        )
        print_result(result)
        log_result(result)

    # -----------------------------------------------------------------------
    # Phase 6: EA bimodal model at low T — more frustrated, different disorder
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Phase 6: EA bimodal model (N=9 square, T=0.1, Gamma sweep)")
    print("=" * 60)

    N_ea = 9  # 3x3 square lattice
    T_ea = 0.1
    gamma_ea_values = [0.0, 0.3, 0.5, 0.8]

    for seed in seeds[:2]:
        adj_ea, J_ea = ea_couplings(N_ea, 'square', 'bimodal', seed)
        fi_ea = frustration_index(adj_ea, J_ea)

        params_cl_ea = make_params(N_ea, T_ea, 0.0, adj_ea, J_ea, seed)
        classical_result_ea, wt_cl_ea = run_classical(params_cl_ea, n_steps=5000)
        E_exact_ea, _ = exact_ground_state(params_cl_ea)
        E_classical_ea = classical_result_ea.energy

        for Gamma_ea in gamma_ea_values:
            params_pimc_ea = make_params(N_ea, T_ea, Gamma_ea, adj_ea, J_ea, seed)
            pimc_result_ea, wt_pimc_ea = run_pimc(params_pimc_ea, n_trotter=16, n_steps=5000)

            E_pimc_ea = pimc_result_ea.energy
            qa_ea = compute_quantum_advantage(E_classical_ea, E_pimc_ea, E_exact_ea)

            result = ExperimentResult(
                commit=commit,
                model="EA",
                topology="square",
                disorder="bimodal",
                n_agents=N_ea,
                temperature=T_ea,
                transverse_field=Gamma_ea,
                frustration=fi_ea,
                seed=seed,
                method="PIMC_EA_bimodal",
                E_best=E_pimc_ea,
                E_exact=E_exact_ea,
                quantum_advantage=qa_ea,
                q_EA=0.0,
                wall_time=wt_pimc_ea,
                status="ok",
                magnetization=float(np.mean(pimc_result_ea.spins)) if pimc_result_ea.spins is not None else 0.0,
                frustration_index=fi_ea,
                binder=0.0,
                metadata={
                    "phase": "ea_bimodal_low_T",
                    "N": N_ea,
                    "T": T_ea,
                    "Gamma": Gamma_ea,
                    "n_trotter": 16,
                    "E_classical": E_classical_ea,
                    "E_pimc": E_pimc_ea,
                    "E_exact": E_exact_ea,
                    "frustration_index": fi_ea,
                    "seed": seed,
                },
            )
            print_result(result)
            log_result(result)

    print("=" * 60)
    print("All phases complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()