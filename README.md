---
category: research
section: introduction
weight: 10
title: "QCCCM: Quantum Compute for Computational Cognitive Modeling"
status: draft
slide_summary: "First open-source quantum cognition library, exploiting the exact Hamiltonian isomorphism between disordered magnets and multi-agent social systems via JAX and PennyLane."
tags: [quantum-computing, cognition, spin-glass, multi-agent, jax, pennylane, sociophysics, game-theory]
---

<div align="center">

# QCCCM

### Quantum Compute for Computational Cognitive Modeling

*The same Hamiltonian that describes a disordered magnet describes a society of agents with heterogeneous relationships. The same tools that find ground states of spin glasses find Nash equilibria of social systems.*

*This library makes that isomorphism computational.*

[![CI](https://github.com/m9h/qc-computational-cognitive-modeling/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/qc-computational-cognitive-modeling/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/JAX-0.4%2B-green?logo=jax)
![PennyLane](https://img.shields.io/badge/PennyLane-0.35%2B-8B5CF6)
![Tests](https://img.shields.io/badge/tests-281%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## Why This Exists

There is no open-source quantum cognition library. The field has rich theory (Busemeyer, Pothos, Khrennikov, Fuss & Navarro) and 443 sociophysics publications indexed by Scopus (Mullick & Sen, 2025), but no reusable software connecting quantum computing frameworks to cognitive and social models.

QCCCM bridges three domains through a single mathematical framework:

```
                         H = -Σ Jᵢⱼ sᵢsⱼ - Σ hᵢsᵢ - Γ Σ Xᵢ

┌──────────────────────┐                                   ┌──────────────────────┐
│   MATERIALS SCIENCE   │                                   │   SOCIAL SCIENCE     │
│                       │         SAME HAMILTONIAN           │                      │
│  Disordered magnet    │◄─────────────────────────────────►│  Multi-agent system  │
│  Spin glass ground    │                                   │  Nash equilibrium    │
│  state search         │                                   │  Social optimum      │
│                       │                                   │                      │
│  VQE / QAOA / D-Wave  │         SAME TOOLS                │  Consensus dynamics  │
│  Phase transitions    │◄─────────────────────────────────►│  Opinion tipping     │
│  Order parameters     │                                   │  Polarization        │
└──────────────────────┘                                   └──────────────────────┘
```

## Key Results

### Quantum Minority Game: Interference Reduces Herding by 90%

Quantum interference in strategy selection dramatically reduces volatility in the crowded phase — **validated by TDD controls showing this is NOT equivalent to classical noise injection**:

| Phase | Classical volatility | Quantum (q=0.5) | Reduction | Noise control |
|-------|---------------------|------------------|-----------|---------------|
| Herding (α=0.02) | 1.93 | 0.15 | **92%** | Classical at any beta: still 1.49 |
| Herding (α=0.04) | 1.35 | 0.14 | **90%** | Same — beta has zero effect |
| Near α_c (α=0.16) | 0.41 | 0.13 | **68%** | |
| Coordination (α=0.63) | 0.15 | 0.17 | -14% | Quantum hurts here |

The effect is monotonic in quantumness, phase-dependent, and structurally different from classical noise (correlation between classical and quantum action sequences: 0.03).

### Materials ↔ Social Correspondence

| Materials Science | Multi-Agent Social Systems |
|---|---|
| Spin s_i in {up, down} | Agent opinion s_i in {A, B} |
| Coupling J_ij > 0 (ferromagnetic) | Agreement incentive (conformity) |
| Coupling J_ij < 0 (antiferromagnetic) | Disagreement incentive (competition) |
| Disorder in J_ij | Heterogeneous social relationships |
| Temperature T | Bounded rationality / noise |
| Transverse field Gamma | Quantum cognitive flexibility |
| Ground state | Nash equilibrium / social optimum |
| Frustrated triangle | Social dilemma (enemy of my enemy) |
| Spin glass phase (m=0, q_EA>0) | Polarization lock-in without consensus |
| Phase transition at T_c | Tipping point in public opinion |

---

## Installation

```bash
git clone https://github.com/m9h/qc-computational-cognitive-modeling.git
cd qc-computational-cognitive-modeling
uv sync                    # core deps (JAX, PennyLane, numpy)
uv sync --group dev        # + pytest, ruff, mypy, coverage
```

Optional backends:
```bash
uv pip install "qcccm[ibm]"        # Qiskit + Aer for IBM hardware
uv pip install "qcccm[annealing]"   # D-Wave Ocean SDK
uv pip install "qcccm[mitiq]"       # Mitiq error mitigation
```

## Quick Start

```python
import jax.numpy as jnp
import numpy as np
from qcccm.core import plus_state, pure_state_density_matrix, von_neumann_entropy, purity
from qcccm.games.minority import MinorityGameParams, run_minority_game
from qcccm.spin_glass import sk_couplings, metropolis_spin_glass, edwards_anderson_q
from qcccm.spin_glass.hamiltonians import SocialSpinGlassParams

# --- Quantum cognition: density matrices as generalized beliefs ---
rho = pure_state_density_matrix(plus_state())
print(f"Purity: {purity(rho):.2f}, Entropy: {von_neumann_entropy(rho):.4f}")

# --- Minority game: quantum interference reduces herding ---
params = MinorityGameParams(n_agents=101, memory=2, n_rounds=500, seed=42)
classical = run_minority_game(params, quantumness=0.0, beta=0.1)
quantum = run_minority_game(params, quantumness=0.5, beta=0.1)
print(f"Volatility — classical: {classical.volatility:.3f}, quantum: {quantum.volatility:.3f}")

# --- Spin glass: social equilibrium search ---
adj, J = sk_couplings(12, seed=42)
sg = SocialSpinGlassParams(12, adj, J, np.zeros(12), temperature=0.3)
result = metropolis_spin_glass(sg, n_steps=5000)
print(f"Energy: {result.energy:.4f}, q_EA: {edwards_anderson_q(result.trajectory):.4f}")
```

---

## Architecture

```
src/qcccm/
├── core/                     # Quantum primitives (JAX, jit-compatible)
│   ├── states.py             #   Basis, Bell, GHZ state preparation
│   ├── density_matrix.py     #   von Neumann entropy, partial trace, fidelity
│   └── quantum_walk.py       #   Hadamard/biased coins, QW evolution, FPT density
│
├── models/                   # Classical-quantum bridges
│   ├── bridge.py             #   Szegedy construction, beliefs ↔ density matrix, quantum EFE
│   └── alf_bridge.py         #   Active inference integration, QuantumEFEAgent
│
├── spin_glass/               # Disordered magnets / social equilibrium search
│   ├── hamiltonians.py       #   SK, EA models, PennyLane Hamiltonian, frustration index
│   ├── order_params.py       #   q_EA, overlap distribution P(q), Binder cumulant
│   ├── solvers.py            #   Metropolis, PIMC (Trotter), VQE, QAOA
│   └── solvers_jax.py        #   JAX-native solvers with jit/vmap (63x batched speedup)
│
├── games/                    # Multi-agent game theory
│   ├── minority.py           #   Minority game with quantum agents, phase transition
│   └── agreement.py          #   Ising agreement, Schelling segregation, frustration
│
├── circuits/                 # PennyLane quantum circuits
│   ├── templates.py          #   Variational ansatze, amplitude encoding
│   ├── interference.py       #   Quantum interference for decision models
│   ├── belief_circuits.py    #   Bayesian belief update on circuit
│   └── export.py             #   PennyLane → Qiskit conversion
│
├── networks/                 # Multi-agent quantum cognitive networks
│   ├── topology.py           #   Complete, ring, star, random graphs
│   ├── multi_agent.py        #   Density matrix evolution with coupling
│   └── observables.py        #   Entropy, polarization, fidelity, coherence
│
├── neuroai/                  # Quantum neuroscience data analysis
│   ├── data_interface.py     #   DANDI/NWB neural data loading
│   ├── neural_states.py      #   EEG/spike covariance → density matrices
│   ├── multiscale.py         #   Multiscale quantum neural analysis
│   ├── path_integral.py      #   Path integral methods
│   └── resource_estimation.py #  Qubit/gate resource estimates
│
├── fitting/                  # Parameter estimation
│   ├── likelihoods.py        #   Choice, QW RT, interference log-likelihoods
│   ├── mle.py                #   MLE with JAX autodiff, AIC/BIC model comparison
│   └── data.py               #   ChoiceData, FitResult containers
│
├── annealing/                # Quantum annealing
│   ├── qubo.py               #   EFE → QUBO, policy assignment
│   └── solve.py              #   Brute force, simulated, D-Wave
│
├── mitigation/               # Quantum error mitigation
│   └── zne.py                #   Zero-noise extrapolation (Richardson)
│
├── pipeline/                 # Data pipelines
│   └── dandi.py              #   DANDI archive neural data pipeline
│
├── benchmarks/               # Performance profiling
│   └── *.py                  #   JIT, walk scaling, network benchmarks
│
└── viz/                      # Visualization
    ├── bloch.py              #   Bloch sphere
    └── walks.py              #   QW probability, spreading, FPT plots
```

### Autoresearch (Karpathy-style autonomous research loop)

```
autoresearch/
├── program.md        # Agent instructions — research questions and strategy
├── prepare.py        # Infrastructure: solvers, metrics, logging (read-only)
└── experiment.py     # Modified by AI agent each generation
```

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). An AI agent autonomously explores the space of sociophysics models, topologies, and quantum solvers to find regimes where quantum methods outperform classical. Three research questions:

1. **Consensus time** — Does PIMC tunneling speed up consensus in frustrated networks?
2. **Quantum minority game** — Does interference reduce herding in the crowded phase?
3. **Ground state search** — Can VQE/QAOA find equilibria that Metropolis misses?

### Solvers

| Solver | Method | Best For |
|--------|--------|----------|
| **Metropolis MC** | Classical single-spin-flip | Baseline, any N |
| **PIMC** | Path-integral MC (Trotter) | Quantum tunneling through barriers |
| **VQE** | Variational Quantum Eigensolver | Small N, exact ground state |
| **QAOA** | Quantum Approximate Optimization | Combinatorial structure |
| **JAX batched** | vmap over disorder seeds | 63x speedup for parameter sweeps |

---

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 01 | From Bits to Qubits | Superposition, density matrices, von Neumann entropy, entanglement |
| 02 | Quantum Walks & Decision | QW vs classical RW, FPT distributions, evidence accumulation |
| 03 | Quantum EFE | Quantum active inference, policy selection, quantumness sweep |
| 04 | Quantum Minority Game | Phase transition at alpha_c, herding reduction, volatility |
| 05 | Spins & Collective Behavior | Ising model, agreement, frustration, Nayebi scaling |
| 06 | Quantum NeuroAI Pipeline | DANDI neural data, quantum state analysis |

No quantum mechanics prerequisites. Dirac notation introduced gradually.

---

## Testing

```bash
uv run pytest -v                           # 281 tests
uv run pytest --cov=qcccm                  # with coverage
uv run pytest tests/test_physics_invariants.py -v  # physics guardrails
```

The test suite includes **physics invariant tests** that encode what must be true regardless of parameters:
- Classical limits (quantum methods reduce to classical when quantum params → 0)
- Energy bounds (no solver can beat exact ground state)
- Thermodynamic consistency
- Tunneling vs heating diagnostics
- Hamiltonian consistency between PennyLane and classical computations
- Minority game quantum validity (the noise-control experiment)

---

## References

**Sociophysics:**
- Mullick & Sen (2025). "Sociophysics models inspired by the Ising model." [arXiv:2506.23837](https://arxiv.org/abs/2506.23837)
- Brock & Durlauf (2001). "Discrete Choice with Social Interactions." *Rev. Econ. Studies* 68:235
- Challet, Marsili & Zecchina (2000). "Minority Games." *PRL* 84:1824

**Quantum Cognition:**
- Busemeyer & Bruza (2012). *Quantum Models of Cognition and Decision*
- Pothos & Busemeyer (2013). "Can quantum probability provide a new direction?" *Psych. Review*
- Khrennikov (2010). *Ubiquitous Quantum Structure*

**Quantum Computing:**
- Farhi et al. (2022). "QAOA and the SK Model." *Quantum* 6:759
- Abbas et al. (2021). "Power of quantum neural networks." *Nature Comp. Sci.*
- Wolff et al. (2025). "Quantum Computing for Neuroscience." PsyArXiv

**Quantum Operator Learning:**
- Xiao et al. (2025). "Quantum DeepONet." *Quantum* 9:1761

---

## License

[MIT](LICENSE)
