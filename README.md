---
category: research
section: introduction
weight: 10
title: "QCCCM: Quantum Compute for Computational Cognitive Modeling"
status: draft
slide_summary: "First open-source quantum cognition library, exploiting the exact Hamiltonian isomorphism between disordered magnets and multi-agent social systems via JAX and PennyLane."
tags: [quantum-computing, cognition, spin-glass, multi-agent, jax, pennylane, sociophysics, game-theory]
---

# QCCCM — Quantum Compute for Computational Cognitive Modeling

> *The same Hamiltonian that describes a disordered magnet describes a society of agents with heterogeneous relationships. The same tools that find ground states of spin glasses find Nash equilibria of social systems. This library makes that isomorphism computational.*

[![Tests](https://github.com/m9h/quantum-cognition/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/quantum-cognition/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![JAX](https://img.shields.io/badge/JAX-0.4%2B-green)
![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-purple)
![Tests](https://img.shields.io/badge/tests-146%20passing-brightgreen)

---

## Vision

There is currently **no open-source quantum cognition Python library**. The field has rich theory (Busemeyer, Pothos, Khrennikov, Fuss & Navarro) but no reusable software. Meanwhile, quantum computing frameworks are mature, hardware access is increasingly free, and the mathematical isomorphism between **disordered magnets** and **multi-agent social systems** is exact.

QCCCM fills this gap — a JAX-native library that lets researchers move between:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│   MATERIALS SCIENCE  │     │  QUANTUM COMPUTING   │     │   SOCIAL SCIENCE    │
│                      │     │                      │     │                     │
│  Spin glass          │◄───►│  VQE / QAOA / D-Wave │◄───►│  Multi-agent game   │
│  Disordered magnet   │     │  PennyLane circuits  │     │  Opinion dynamics   │
│  Phase transitions   │     │  Quantum walks       │     │  Consensus/conflict │
│                      │     │  Error mitigation    │     │                     │
│  H = -Σ Jᵢⱼ sᵢsⱼ   │     │                      │     │  H = -Σ Jᵢⱼ sᵢsⱼ  │
└─────────────────────┘     └──────────────────────┘     └─────────────────────┘
         ▲                                                         ▲
         │              SAME HAMILTONIAN                           │
         └─────────────────────────────────────────────────────────┘
```

### The Materials ↔ Social Correspondence

| Materials Science | Multi-Agent Social Systems |
|---|---|
| Spin s_i ∈ {↑,↓} | Agent opinion s_i ∈ {A,B} |
| Coupling J_ij > 0 (ferromagnetic) | Agreement incentive (conformity) |
| Coupling J_ij < 0 (antiferromagnetic) | Disagreement incentive (competition) |
| Disorder in J_ij | Heterogeneous social relationships |
| Temperature T | Bounded rationality / noise |
| Transverse field Γ | Quantum cognitive flexibility |
| Ground state | Nash equilibrium / social optimum |
| Frustrated triangle | Social dilemma (enemy of my enemy) |
| Spin glass phase (m≈0, q_EA>0) | Polarization lock-in without consensus |
| Phase transition at T_c | Tipping point in public opinion |

---

## Architecture

```
src/qcccm/
├── core/                     # Quantum primitives (JAX, jit-compatible)
│   ├── states.py             #   |0⟩, |+⟩, Bell, GHZ state preparation
│   ├── density_matrix.py     #   von Neumann entropy, partial trace, QMI, purity, fidelity
│   └── quantum_walk.py       #   Hadamard/biased coins, QW evolution (lax.scan), FPT density
│
├── models/                   # Classical ↔ quantum bridges
│   ├── bridge.py             #   Szegedy stochastic→unitary, beliefs↔ρ, quantum EFE
│   └── alf_bridge.py         #   ALF active inference integration, QuantumEFEAgent
│
├── spin_glass/               # Disordered magnet / social simulation
│   ├── hamiltonians.py       #   SK, EA models, PennyLane Hamiltonian construction
│   ├── order_params.py       #   q_EA, P(q) overlap distribution, Binder cumulant, χ_SG
│   └── solvers.py            #   Metropolis, PIMC (Trotter), VQE, QAOA
│
├── games/                    # Multi-agent game theory
│   ├── minority.py           #   Minority game with quantum agents, phase transition at α_c
│   └── agreement.py          #   Ising agreement model, frustration, Schelling segregation
│
├── circuits/                 # PennyLane quantum circuits
│   ├── templates.py          #   Variational ansatze, amplitude encoding
│   ├── interference.py       #   Quantum interference for decision models
│   ├── belief_circuits.py    #   Bayesian belief update on circuit
│   └── export.py             #   PennyLane → Qiskit conversion
│
├── networks/                 # Multi-agent quantum cognitive networks
│   ├── topology.py           #   Complete, ring, star, random graph construction
│   ├── multi_agent.py        #   Density matrix evolution with coupling + decoherence
│   └── observables.py        #   Entropy, polarization, fidelity, coherence
│
├── fitting/                  # Parameter estimation
│   ├── likelihoods.py        #   Choice, QW RT, interference log-likelihoods (JAX)
│   ├── mle.py                #   MLE with JAX autodiff, AIC/BIC model comparison
│   └── data.py               #   ChoiceData, FitResult containers
│
├── annealing/                # Quantum annealing
│   ├── qubo.py               #   EFE → QUBO mapping, policy assignment
│   └── solve.py              #   Brute force, simulated, D-Wave solvers
│
├── mitigation/               # Quantum error mitigation
│   └── zne.py                #   Zero-noise extrapolation (Richardson)
│
├── benchmarks/               # Performance profiling
│   ├── bench_jax.py          #   JIT compilation, walk scaling, density matrix ops
│   └── bench_networks.py     #   Network evolution, observables scaling
│
└── viz/                      # Visualization
    ├── bloch.py              #   Bloch sphere for single-qubit states
    └── walks.py              #   QW probability evolution, spreading, FPT plots

autoresearch/                 # Autonomous research loop (Karpathy-style)
├── program.md                #   Agent instructions — the "skill file"
├── prepare.py                #   Infrastructure (read-only): solvers, metrics, logging
└── experiment.py             #   THE FILE THE AI AGENT MODIFIES

notebooks/
├── 01_bits_to_qubits.ipynb           # Superposition, density matrices, von Neumann entropy
├── 02_quantum_walks_decision.ipynb   # QW vs classical RW, FPT, evidence accumulation
├── 03_quantum_vs_classical_efe.ipynb # Quantum EFE, active inference, policy selection
├── 04_quantum_minority_game.ipynb    # Phase transition at α_c, quantum agent coordination
└── 05_spins_phases_collective.ipynb  # Ising model, agreement dynamics, Nayebi scaling
```

---

## Quick Start

```bash
# Install
uv sync

# Run tests (146 passing)
uv run pytest -v

# Run a spin glass experiment
uv run python autoresearch/experiment.py
```

```python
import jax.numpy as jnp
from qcccm.core import (
    plus_state, pure_state_density_matrix,
    von_neumann_entropy, purity,
    QuantumWalkParams, quantum_walk_evolution,
)
from qcccm.spin_glass import sk_couplings, metropolis_spin_glass, edwards_anderson_q
from qcccm.spin_glass.hamiltonians import SocialSpinGlassParams
import numpy as np

# --- Quantum cognition: density matrices as generalized beliefs ---
psi = plus_state()
rho = pure_state_density_matrix(psi)
print(f"Purity: {purity(rho):.2f}, S(ρ): {von_neumann_entropy(rho):.4f}")

# --- Quantum walk: ballistic spreading ---
params = QuantumWalkParams(n_sites=101, n_steps=50, start_pos=50)
probs = quantum_walk_evolution(params)  # (51, 101) — O(t²) spreading

# --- Spin glass: social equilibrium search ---
N = 12
adj, J = sk_couplings(N, seed=42)
sg_params = SocialSpinGlassParams(N, adj, J, np.zeros(N), temperature=0.3)
result = metropolis_spin_glass(sg_params, n_steps=5000)
print(f"Ground state energy: {result.energy:.4f}")
print(f"Edwards-Anderson q_EA: {edwards_anderson_q(result.trajectory):.4f}")
```

---

## Solvers

Four methods for finding social equilibria / spin glass ground states:

| Solver | Method | Best For | N Range |
|--------|--------|----------|---------|
| **Metropolis MC** | Classical single-spin-flip | Baseline, any system | Any |
| **PIMC** | Path-integral Monte Carlo (Trotter slices) | Quantum tunneling through barriers | Any |
| **VQE** | Variational Quantum Eigensolver (PennyLane) | Small systems, exact ground state | ≤ 16 |
| **QAOA** | Quantum Approximate Optimization (PennyLane) | Combinatorial structure | ≤ 16 |

```python
from qcccm.spin_glass.solvers import (
    metropolis_spin_glass,    # Classical baseline
    transverse_field_mc,      # Quantum-inspired PIMC
    vqe_ground_state,         # PennyLane VQE
    qaoa_ground_state,        # PennyLane QAOA
)
```

---

## Autoresearch

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (45K stars). An autonomous AI research loop that iterates on sociophysics quantum architecture:

```
LOOP FOREVER:
  1. Read results.tsv — what worked? what's unexplored?
  2. Modify experiment.py — new model, parameters, solver
  3. Run experiment (10-min timeout)
  4. Compute quantum_advantage = (E_classical - E_quantum) / |E_exact|
  5. If advantage > 0.01 → KEEP (advance branch)
  6. If not → DISCARD (git reset)
  7. Log to results.tsv
  8. Generate next hypothesis
```

The agent explores the space of sociophysics models (SK, EA, Voter, Majority Rule, Sznajd, Schelling, Minority Game) × topologies (complete, lattice, random, scale-free) × solvers (Metropolis, PIMC, VQE, QAOA) to find regimes where quantum methods outperform classical.

```bash
# Kick off with any LLM coding agent:
# "Read autoresearch/program.md and start the research loop"
uv run python autoresearch/experiment.py
```

---

## Notebooks

| # | Notebook | Topics | Prerequisites |
|---|----------|--------|---------------|
| 01 | From Bits to Qubits | Superposition, density matrices, von Neumann entropy, entanglement | Linear algebra |
| 02 | Quantum Walks & Decision | QW vs classical RW, FPT, interference fringes, evidence accumulation | NB 01, DDM concepts |
| 03 | Quantum EFE | Quantum active inference, policy selection, quantumness parameter | NB 01, AIF concepts |
| 04 | Quantum Minority Game | Phase transition at α_c, quantum agent coordination, volatility | NB 03 |
| 05 | Spins & Collective Behavior | Ising model, agreement dynamics, frustration, Nayebi scaling | NB 04 |

No quantum mechanics prerequisites. Dirac notation introduced gradually (NB 01-02 use matrix notation only).

---

## Optional Dependencies

```bash
uv pip install "qcccm[ibm]"        # Qiskit + Aer (IBM hardware access)
uv pip install "qcccm[annealing]"   # D-Wave Ocean SDK (quantum annealing)
uv pip install "qcccm[mitiq]"       # Mitiq error mitigation
```

---

## Key References

**Sociophysics:**
- Mullick & Sen (2025). "Sociophysics models inspired by the Ising model." [arXiv:2506.23837](https://arxiv.org/abs/2506.23837) — comprehensive review, 118 references
- Brock & Durlauf (2001). "Discrete Choice with Social Interactions." *Rev. Econ. Studies* 68:235 — mean-field Ising = logistic choice
- Challet, Marsili, Zecchina (2000). "Statistical Mechanics of Minority Games." *PRL* 84:1824 — replica solution

**Quantum Cognition:**
- Busemeyer & Bruza (2012). *Quantum Models of Cognition and Decision* — foundational textbook
- Pothos & Busemeyer (2013). "Can quantum probability provide a new direction?" *Psych. Review*
- Khrennikov (2010). *Ubiquitous Quantum Structure* — quantum-like models outside physics

**Quantum Computing for Social/Materials Science:**
- Farhi et al. (2022). "QAOA and the SK Model at Infinite Size." *Quantum* 6:759 — QAOA surpasses SDP at depth p=11
- Abbas et al. (2021). "Power of quantum neural networks." *Nature Comp. Sci.* — QNN effective dimension
- Andreev, Cattan et al. (2023). "pyRiemann-qiskit." *RIO Journal* — Riemannian + quantum classifiers

**Autoresearch:**
- Karpathy (2026). [autoresearch](https://github.com/karpathy/autoresearch) — autonomous AI-driven ML research

---

## License

MIT
