# QCCCM Curriculum — Quantum Compute for Computational Cognitive Modeling

A self-contained curriculum bridging quantum computing and computational cognitive neuroscience. Designed for students with linear algebra and probability background — no quantum mechanics required.

## Prerequisites

- Linear algebra (matrix operations, eigenvalues, inner products)
- Probability (Bayes' rule, Shannon entropy, distributions)
- Python + JAX (basic familiarity)
- Recommended: spinning-up-alf notebooks 01-08 (active inference foundations)

## Notebook Sequence

### Track A: Quantum Foundations

| # | Notebook | What you learn | QCCCM modules used |
|---|----------|---------------|---------------------|
| 01 | [From Bits to Qubits](notebooks/01_bits_to_qubits.ipynb) | Superposition, density matrices, von Neumann entropy, coherence, entanglement, interference | `core/` |
| 02 | [Quantum Walks & Decision-Making](notebooks/02_quantum_walks_decision.ipynb) | Ballistic spreading, FPT as RT distribution, coin angle as evidence strength | `core/`, `viz/` |

### Track B: Quantum Active Inference

| # | Notebook | What you learn | QCCCM modules used |
|---|----------|---------------|---------------------|
| 03 | [Quantum vs Classical EFE](notebooks/03_quantum_vs_classical_efe.ipynb) | Quantum EFE, quantumness parameter, policy selection, density matrix view | `models/`, `core/` |

### Track C: Collective Behavior & Statistical Mechanics

| # | Notebook | What you learn | QCCCM modules used |
|---|----------|---------------|---------------------|
| 04 | [The Quantum Minority Game](notebooks/04_quantum_minority_game.ipynb) | Anti-coordination, herding, phase transition at α_c, quantum strategy selection | `games/` |
| 05 | [Spins, Phases, and Collective Behavior](notebooks/05_spins_phases_collective.ipynb) | Ising model, phase transitions, frustration, multi-agent agreement as spin dynamics, Nayebi bounds | `games/`, `spin_glass/` |

### Track D: NeuroAI

| # | Notebook | What you learn | QCCCM modules used |
|---|----------|---------------|---------------------|
| 06 | [Quantum NeuroAI Pipeline](notebooks/06_quantum_neuroai_pipeline.ipynb) | Spikes → density matrices, qubit resource estimation, quantum entropy/fidelity of neural states | `neuroai/` |

## Connections to spinning-up-alf

| spinning-up-alf topic | QCCCM extension |
|---|---|
| NB 08: Active inference basics | NB 03: Quantum EFE generalises classical EFE |
| NB 10: Multi-agent systems | NB 04-05: Minority game + Ising model scaffold |
| NB 12: HGF perceptual model | `neuroai/path_integral.py`: path integral evidence accumulation |
| NB 14: DDM decision model | NB 02: Quantum walk FPT as quantum DDM |
| NB 16: Metacognition | `models/alf_bridge.py`: quantumness as metacognitive control |

## Connections to external work

| Research group | Paper | QCCCM connection |
|---|---|---|
| Dumas/Wolff (UdeM/Mila) | Quantum Computing for Neuroscience (2025) | `neuroai/` module implements their conceptual framework |
| Nayebi (CMU) | Intrinsic Barriers for Human-AI Alignment (2025) | NB 05 + `games/agreement.py` test the Ω(MN²) bound |
| Challet/Marsili | Minority Games (1997-2005) | NB 04 + `games/minority.py` with quantum extension |
| Busemeyer/Bruza | Quantum Models of Cognition (2012) | Core framework: density matrices as generalised beliefs |

## Module Map

```
qcccm/
├── core/              Density matrices, quantum walks, states (JAX)
├── models/            Classical↔quantum bridge, ALF integration
├── circuits/          PennyLane circuits (belief, interference, export)
├── fitting/           MLE, log-likelihoods, model comparison
├── networks/          Multi-agent topology, evolution, observables
├── games/             Minority game, Ising model, agreement, spin_glass bridge
├── spin_glass/        SK/EA Hamiltonians, VQE/QAOA solvers, order parameters
├── annealing/         QUBO, D-Wave solver, policy optimisation
├── mitigation/        ZNE error mitigation, noisy circuits
├── benchmarks/        JAX JIT timing, scaling analysis
├── neuroai/           Neural states, path integrals, multiscale circuits,
│                      resource estimation, data interface (BL-1/NWB)
└── viz/               Bloch sphere, quantum walk plots
```

## Suggested reading order for self-study

1. **Week 1**: NB 01 (qubits) → NB 02 (walks) — build quantum intuition
2. **Week 2**: NB 03 (EFE) — connect to active inference
3. **Week 3**: NB 05 (Ising) → NB 04 (minority game) — collective behavior
4. **Week 4**: NB 06 (neuroai pipeline) — applications to real data
5. **Ongoing**: Exercises in each notebook, stretch problems connect modules

## For instructors

Each notebook follows the pattern: **concept → simulation → analysis → exercises**. The exercises are graded: Basic (verify understanding), Stretch (connect to other modules or real data). All code is self-contained — students need only `uv sync` to get started.
