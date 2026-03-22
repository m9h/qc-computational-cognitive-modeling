# QCCCM: Quantum Compute for Computational Cognitive Modeling

## Paper Outline

### Title
Quantum Compute for Computational Cognitive Modeling: A JAX Framework Bridging Quantum Information Theory, Active Inference, and Multi-Agent Game Theory

### Authors
Morgan Hough

### Abstract

We present QCCCM, a JAX-native framework that connects quantum information theory to computational cognitive modeling. The framework implements density matrices as generalized beliefs, quantum walks as evidence accumulation processes, and quantum expected free energy as a policy selection mechanism for active inference agents. We extend the single-agent formalism to multi-agent systems through three converging paths: (1) quantum-enhanced strategy selection in the minority game, where off-diagonal coherences in the strategy density matrix break herding in the crowded phase; (2) frustrated multi-agent agreement modeled as a spin glass, where transverse-field quantum Monte Carlo and variational quantum eigensolvers find lower-energy agreement configurations than classical Metropolis; and (3) a neural data interface that maps spike train recordings from cortical cultures and brain organoids to quantum states, enabling quantum information-theoretic analysis of population dynamics. The framework is accompanied by a six-notebook curriculum that teaches the mathematical scaffold (Ising models, phase transitions, frustration) connecting game theory to statistical mechanics, and a bridge to the Active Learning Framework (ALF) for active inference. All code is open-source, JAX-accelerated, and tested (198 tests).

### 1. Introduction

- The quantum cognition research program \cite{busemeyer2012quantum, pothos2013can, khrennikov2010ubiquitous}
- Gap: no unified computational framework connecting quantum cognition to active inference, multi-agent systems, and real neural data
- QCCCM fills this gap

### 2. Framework Architecture

#### 2.1 Core Quantum Primitives
- Density matrices as generalized beliefs
- Von Neumann entropy, partial trace, fidelity, quantum mutual information
- JAX jit-compiled throughout

#### 2.2 Classical-Quantum Bridge
- `beliefs_to_density_matrix`: probability vectors → diagonal ρ
- `stochastic_to_unitary`: Szegedy construction
- `quantum_efe`: quantum expected free energy
- The `quantumness` parameter q ∈ [0,1]

#### 2.3 Quantum Walks as Evidence Accumulation
- Ballistic vs diffusive spreading
- First-passage time as reaction time distribution
- Connection to drift-diffusion models

### 3. Multi-Agent Extensions

#### 3.1 Quantum Minority Game
- Classical minority game (Challet & Zhang 1997) \cite{challet1997emergence}
- Phase transition at α_c ≈ 0.34
- Quantum agents: density matrix over strategies
- Result: coherence reduces volatility in crowded phase

#### 3.2 Multi-Agent Agreement as a Spin Glass
- Ising model scaffold \cite{brock2001discrete}
- Frustrated agreement (anti-ferromagnetic bonds)
- Connection to Nayebi's Ω(MN²) communication bound \cite{nayebi2025alignment}
- Classical Metropolis vs transverse-field MC vs VQE/QAOA

#### 3.3 Network Evolution
- Mean-field density matrix dynamics on graphs
- Consensus, polarization, coherence observables
- DeGroot vs quantum consensus

### 4. NeuroAI Interface

#### 4.1 Neural States as Density Matrices
- Firing rates → quantum states \cite{schneidman2006weak}
- Correlations as off-diagonal coherences
- Von Neumann entropy of neural populations

#### 4.2 Qubit Resource Estimation
- Wolff et al. (2025) scaling analysis \cite{wolff2025quantum}
- Amplitude encoding: ⌈log₂ N⌉ qubits
- Organoid and culture systems accessible with near-term hardware

#### 4.3 Data Pipeline
- BL-1 cortical culture simulator integration \cite{kagan2022vitro}
- NWB/DANDI organoid datasets \cite{sharf2022functional, vandermolen2026protosequences}
- Spike trains → rates → density matrices → quantum observables

### 5. Active Inference Integration

#### 5.1 ALF Bridge
- `QuantumEFEAgent` as drop-in replacement for `AnalyticAgent`
- Polar decomposition: transition matrix → closest unitary
- Preferences → preferred density matrix

#### 5.2 Results
- Quantumness sweep on foraging task
- Optimal q is task-dependent
- Connection to bounded rationality

### 6. PennyLane Circuit Implementation

- Belief update circuits
- Interference circuits (conjunction fallacy)
- Multiscale neural dynamics circuits
- Error mitigation (ZNE)
- Optional Qiskit export and D-Wave annealing

### 7. Educational Curriculum

- Six notebooks covering foundations → applications
- Connection to spinning-up-alf
- Ising model scaffold for understanding multi-agent results

### 8. Discussion

- Quantum cognition is not a claim about quantum physics in the brain
- It is a mathematical framework that generalizes classical probability
- The framework is empirically testable: fit q to behavioral/neural data
- Connections: Dumas quantum neuroAI, Nayebi alignment bounds, BL-1 organoid validation

### References
