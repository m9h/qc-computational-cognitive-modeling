"""Autoresearch experiment — THIS FILE IS MODIFIED BY THE AI AGENT.

Two research questions:

Q1: CONSENSUS TIME — Does quantum tunneling (PIMC) speed up consensus
in frustrated social networks? Measure tau (steps to |m| >= threshold)
for classical Metropolis vs PIMC across temperature and frustration.
Classical tau scales as N^2 (1D) or N*log(N) (2D). If PIMC gives
tau_quantum < tau_classical at any (N, T, frustration), that's a finding.

Q2: QUANTUM MINORITY GAME — Does quantum interference in strategy
selection reduce herding (volatility) in the crowded phase? Sweep
quantumness and alpha = 2^M/N. The classical phase transition at
alpha_c ~ 0.34 separates herding (alpha < alpha_c) from coordination.
Does quantum shift alpha_c or reduce peak volatility?
"""

import numpy as np
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from prepare import (
    ExperimentResult,
    compute_consensus_speedup,
    compute_volatility_reduction,
    get_commit_hash,
    log_result,
    measure_consensus_time,
    print_result,
)
from qcccm.spin_glass.hamiltonians import (
    SocialSpinGlassParams,
    ea_couplings,
    frustration_index,
    sk_couplings,
)
from qcccm.games.minority import (
    MinorityGameParams,
    run_minority_game,
)


def run_experiment():
    commit = get_commit_hash()

    # =================================================================
    # Q1: CONSENSUS TIME — classical Metropolis vs PIMC
    # =================================================================
    print("=" * 60)
    print("Q1: CONSENSUS TIME — tunneling through frustrated barriers")
    print("=" * 60)

    max_steps = 10000
    threshold = 0.8
    n_trotter = 8
    seeds = [42, 123, 7, 999, 31]

    # Sweep: N × T × frustration × Gamma
    consensus_configs = [
        # (N, topology, disorder, frustrations, temperatures, gammas)
        (16, "square", "bimodal", [0.0, 0.2, 0.4], [0.5, 1.0, 1.5, 2.0], [0.5, 1.0, 2.0]),
        (20, "complete", "gaussian", [0.0], [0.3, 0.8, 1.5], [0.5, 1.0, 2.0]),
    ]

    best_speedup = 0.0
    best_consensus_config = ""

    for N, topo, disorder, frusts, temps, gammas in consensus_configs:
        for frust in frusts:
            for seed in seeds[:3]:
                # Generate couplings
                if disorder == "gaussian":
                    adj, J = sk_couplings(N, seed=seed)
                else:
                    adj, J = ea_couplings(N, topology=topo, disorder=disorder, seed=seed)
                    # Apply frustration: flip fraction of bonds
                    if frust > 0:
                        rng = np.random.default_rng(seed + 1000)
                        n_bonds = int(np.sum(adj) / 2)
                        rows, cols = np.where(np.triu(adj) > 0)
                        n_flip = int(frust * len(rows))
                        flip_idx = rng.choice(len(rows), size=n_flip, replace=False)
                        for idx in flip_idx:
                            i, j = rows[idx], cols[idx]
                            J[i, j] *= -1
                            J[j, i] *= -1

                actual_N = adj.shape[0]
                fields = np.zeros(actual_N)
                fi = frustration_index(adj, J)

                for T in temps:
                    # Classical Metropolis consensus time
                    params_c = SocialSpinGlassParams(
                        n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                        transverse_field=0.0, temperature=T, seed=seed,
                    )
                    t0 = time.time()
                    tau_c, mag_c = measure_consensus_time(
                        params_c, n_steps=max_steps, threshold=threshold,
                        use_pimc=False,
                    )
                    time_c = time.time() - t0

                    # Log classical baseline
                    desc_c = (f"consensus classical {topo} N={actual_N} T={T} "
                              f"frust={frust} seed={seed} fi={fi:.2f}")
                    res_c = ExperimentResult(
                        commit=commit, description=desc_c,
                        model=f"consensus_{disorder}", topology=topo,
                        disorder=disorder, n_agents=actual_N, temperature=T,
                        transverse_field=0.0, frustration=frust, seed=seed,
                        method="metropolis",
                        consensus_time=tau_c,
                        magnetization=float(mag_c[-1]) if len(mag_c) > 0 else 0.0,
                        frustration_index=fi, wall_time=time_c,
                    )
                    print_result(res_c)
                    log_result(res_c)

                    for Gamma in gammas:
                        # PIMC consensus time
                        params_q = SocialSpinGlassParams(
                            n_agents=actual_N, adjacency=adj, J=J, fields=fields,
                            transverse_field=Gamma, temperature=T, seed=seed,
                        )
                        t0 = time.time()
                        tau_q, mag_q = measure_consensus_time(
                            params_q, n_steps=max_steps, threshold=threshold,
                            use_pimc=True, n_trotter=n_trotter,
                        )
                        time_q = time.time() - t0

                        speedup = compute_consensus_speedup(tau_c, tau_q, max_steps)

                        desc_q = (f"consensus PIMC {topo} N={actual_N} T={T} "
                                  f"Gamma={Gamma} frust={frust} seed={seed} fi={fi:.2f}")
                        res_q = ExperimentResult(
                            commit=commit, description=desc_q,
                            model=f"consensus_{disorder}", topology=topo,
                            disorder=disorder, n_agents=actual_N, temperature=T,
                            transverse_field=Gamma, frustration=frust, seed=seed,
                            method="pimc",
                            consensus_time=tau_q,
                            consensus_speedup=speedup,
                            magnetization=float(mag_q[-1]) if len(mag_q) > 0 else 0.0,
                            frustration_index=fi, wall_time=time_q,
                        )
                        print_result(res_q)
                        log_result(res_q)

                        if speedup > best_speedup:
                            best_speedup = speedup
                            best_consensus_config = desc_q

    print(f"\nQ1 BEST: speedup={best_speedup:.2f} — {best_consensus_config}")

    # =================================================================
    # Q2: QUANTUM MINORITY GAME — interference vs herding
    # =================================================================
    print("\n" + "=" * 60)
    print("Q2: QUANTUM MINORITY GAME — interference reduces herding?")
    print("=" * 60)

    N_agents = 101
    n_rounds = 500
    n_seeds = 5
    beta = 0.1

    # Sweep quantumness × memory (alpha)
    quantumness_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    memory_values = [1, 2, 3, 4, 5, 6, 7]

    best_vol_reduction = 0.0
    best_minority_config = ""

    for M in memory_values:
        alpha = 2**M / N_agents

        # Classical baseline (average over seeds)
        classical_vols = []
        for seed in range(n_seeds):
            params = MinorityGameParams(
                n_agents=N_agents, memory=M, n_strategies=2,
                n_rounds=n_rounds, seed=seed,
            )
            result = run_minority_game(params, quantumness=0.0, beta=beta)
            classical_vols.append(result.volatility)
        vol_classical = np.mean(classical_vols)

        desc_c = f"minority classical N={N_agents} M={M} alpha={alpha:.4f} beta={beta}"
        res_c = ExperimentResult(
            commit=commit, description=desc_c,
            model="minority_game", topology="complete",
            n_agents=N_agents, temperature=1.0 / beta,
            method="classical",
            volatility=vol_classical,
        )
        print_result(res_c)
        log_result(res_c)

        for q in quantumness_values:
            if q == 0.0:
                continue  # already logged classical

            quantum_vols = []
            for seed in range(n_seeds):
                params = MinorityGameParams(
                    n_agents=N_agents, memory=M, n_strategies=2,
                    n_rounds=n_rounds, seed=seed,
                )
                result = run_minority_game(params, quantumness=q, beta=beta)
                quantum_vols.append(result.volatility)
            vol_quantum = np.mean(quantum_vols)

            vol_red = compute_volatility_reduction(vol_classical, vol_quantum)

            desc_q = (f"minority quantum N={N_agents} M={M} alpha={alpha:.4f} "
                      f"q={q} beta={beta}")
            res_q = ExperimentResult(
                commit=commit, description=desc_q,
                model="minority_game", topology="complete",
                n_agents=N_agents, temperature=1.0 / beta,
                transverse_field=q,  # using Gamma field for quantumness
                method="quantum",
                volatility=vol_quantum,
                volatility_reduction=vol_red,
            )
            print_result(res_q)
            log_result(res_q)

            if vol_red > best_vol_reduction:
                best_vol_reduction = vol_red
                best_minority_config = desc_q

    print(f"\nQ2 BEST: volatility_reduction={best_vol_reduction:.4f} — {best_minority_config}")

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"Q1 best consensus speedup: {best_speedup:.2f}")
    print(f"   config: {best_consensus_config}")
    print(f"Q2 best volatility reduction: {best_vol_reduction:.4f}")
    print(f"   config: {best_minority_config}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_experiment()
