---
category: research
section: methodology
weight: 30
title: "Autonomous Sociophysics Quantum Research Loop"
status: draft
slide_summary: "Karpathy-inspired autonomous agent loop that systematically explores sociophysics models, topologies, and quantum solvers to find regimes where quantum methods outperform classical ground-state search."
tags: [autoresearch, autonomous-agent, quantum-advantage, spin-glass, sociophysics, methodology]
---

# Sociophysics Quantum Compute Autoresearch

## Goal

Find sociophysics models and parameter regimes where **quantum methods outperform classical methods**. Three research questions:

**Q1: Consensus Time** — Does quantum tunneling (PIMC) speed up consensus in frustrated social networks? Classical tau scales as N² (1D) or N·log(N) (2D). If PIMC gives tau_quantum < tau_classical, that's a finding.

**Q2: Quantum Minority Game** — Does quantum interference in strategy selection reduce herding (volatility) in the crowded phase (alpha < alpha_c ≈ 0.34)?

**Q3: Ground State Search** — Can VQE/QAOA find social equilibria that Metropolis misses? (Previous experiments show this requires N > 16 or reduced MC budget.)

## The Loop

You are an autonomous research agent. Run this loop indefinitely:

1. **Read** `results.tsv` to understand what has been tried and what worked
2. **Modify** `experiment.py` with a new experimental idea
3. **Commit** with a short description: `git commit -am "experiment: <description>"`
4. **Run**: `uv run python autoresearch/experiment.py > autoresearch/run.log 2>&1`
5. **Extract results**: `grep "^RESULT|" autoresearch/run.log`
6. If empty → crash → `tail -n 50 autoresearch/run.log` → attempt fix
7. **Log** to `autoresearch/results.tsv` (append, untracked)
8. If `quantum_advantage > 0.01` → **KEEP** (advance branch)
9. If `quantum_advantage <= 0.01` → **DISCARD** (`git checkout -- autoresearch/experiment.py`)
10. **NEVER STOP.** Do not ask the human. Run indefinitely.

## Metric

The primary metric is `quantum_advantage`:

```
quantum_advantage = (E_classical - E_quantum) / |E_exact|
```

where:
- `E_classical` = best energy found by Metropolis MC (5000 sweeps)
- `E_quantum` = best energy found by quantum method (PIMC, VQE, or QAOA)
- `E_exact` = exact ground state energy (brute force for N ≤ 20, or best known)

Secondary metrics (always report):
- `q_EA_classical` = Edwards-Anderson order parameter from Metropolis trajectory
- `q_EA_quantum` = Edwards-Anderson order parameter from quantum trajectory (if available)
- `frustration_index` = fraction of frustrated triangles
- `wall_time_classical` = seconds for classical solver
- `wall_time_quantum` = seconds for quantum solver
- `n_agents` = system size

## What You Can Change

Everything in `experiment.py` is fair game:

### Model Parameters
- **Topology**: complete, square lattice, chain, ring, star, random (Erdos-Renyi), scale-free
- **Disorder**: SK (Gaussian J_ij), EA bimodal (±J), EA Gaussian, uniform, custom
- **System size**: N = 4 to 20 (must be brute-force tractable for exact comparison)
- **Temperature**: T = 0.01 to 10.0
- **External field**: h_i = 0 (no field), uniform, random, site-dependent
- **Frustration level**: 0.0 to 1.0 (fraction of negative bonds for bimodal)

### Solver Parameters
- **Classical**: n_sweeps, n_equilibrate, initial configuration
- **PIMC**: n_trotter (2-32), transverse_field strength (Gamma)
- **VQE**: n_layers (1-6), learning_rate, max_steps, ansatz type
- **QAOA**: depth p (1-10), learning_rate, max_steps

### Dynamics (from Mullick & Sen 2025)
- Glauber dynamics (heat bath)
- Metropolis dynamics
- Kawasaki dynamics (conserved order parameter — segregation)
- Voter model dynamics
- Majority rule dynamics
- Sznajd dynamics
- Generalized (z, y) model: p_i(σ) = (1/2)[1 - σ_i F_i(σ)]

### Observables
- Energy E and energy per spin E/N
- Magnetization |m|
- Edwards-Anderson q_EA
- Overlap distribution P(q) from multiple replicas
- Binder cumulant U_L
- Glass susceptibility χ_SG
- Consensus time τ (for dynamics experiments)
- Exit probability E(x)
- Persistence P(t)

## What You Cannot Change

- `prepare.py` — the infrastructure (qcccm library imports, solver wrappers)
- The metric definition
- The 10-minute wall-clock timeout per experiment
- The requirement to report all secondary metrics

## Strategy Guidance

1. **Start simple**: N=6-8, SK model, compare Metropolis vs PIMC. Find the regime where PIMC wins.
2. **Increase frustration**: Frustrated systems are where quantum tunneling should help most.
3. **Sweep transverse field**: Find optimal Gamma for PIMC at each (N, T, frustration).
4. **Try VQE/QAOA**: For small N (4-8), compare against exact ground state.
5. **Vary topology**: Does quantum advantage depend on graph structure?
6. **Try dynamics**: Consensus time τ — does quantum tunneling speed up consensus in frustrated networks?
7. **Combine insights**: If you find quantum advantage in a specific regime, characterize it systematically.

## Results Format

Each experiment appends one line to `results.tsv`:

```
commit  model  topology  disorder  N  T  Gamma  frustration  method  E_best  E_exact  quantum_advantage  q_EA  wall_time  status  description
```

## If You Get Stuck

- Re-read results.tsv for patterns. What worked? What's unexplored?
- Try combining two previous near-misses
- Try a radically different topology or dynamics
- Increase system size (more room for frustration)
- Try disorder in the external field (random field Ising model)
- Read the correspondence table in program.md and pick a new social scenario

## Reference: Materials ↔ Social Correspondence

| Materials | Social | Hamiltonian |
|---|---|---|
| Ferromagnet | Consensus game | H = -J Σ s_i s_j, J > 0 |
| Antiferromagnet | Competition game | H = -J Σ s_i s_j, J < 0 |
| Spin glass (SK) | Heterogeneous social network | J_ij ~ N(0, 1/√N) |
| Spin glass (EA) | Local trust/distrust network | J_ij = ±1 on lattice |
| Random field Ising | Diverse private information | h_i ~ N(0, σ_h) |
| Transverse field Ising | Quantum cognitive flexibility | H_x = -Γ Σ X_i |
| Frustrated triangle | Social dilemma | Mixed-sign J on triangle |
| Schelling segregation | Kawasaki Ising below T_c | Conserved magnetization |
| Minority game | p-spin glass | Replica RSB at α_c |
