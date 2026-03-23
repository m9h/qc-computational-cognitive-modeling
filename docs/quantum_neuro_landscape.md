---
category: research
section: background
weight: 20
title: "Quantum Computing for Neuroscience: Landscape and Replication Targets"
status: draft
slide_summary: "Survey of quantum+neuroscience efforts — Dumas, QEEGNet, pyRiemann-qiskit, Quantum DeepONet — with replication targets and gaps qcccm can fill."
tags: [quantum-computing, neuroscience, EEG, connectomics, BCI, landscape, replication]
---

# Quantum Computing for Neuroscience: Landscape and Replication Targets

## Overview

As of early 2026, quantum computing for neuroscience is an emerging field with:
- **Rich theory** (Busemeyer, Pothos, Khrennikov) but **no reusable open-source library** — qcccm fills this gap
- **Several groups running neural data through quantum circuits** (EEG classification, fMRI analysis, BCI) with promising but early results
- **One major gap everyone identifies**: quantum methods for **connectomics** are completely unexplored
- **Quantum walks on brain networks** — the most natural application — has **zero published implementations**

---

## 1. Dumas Group (PPSP Lab, UdeM / Mila)

### People
- **Guillaume Dumas** — PI, computational psychiatry, hyperscanning, social neuro-AI
- **Annemarie Wolff** — Lead on quantum work, member of IBM Quantum Healthcare Working Group
- **Atsushi Iriki** (Teikyo U) — Co-author, bridge to Khrennikov's quantum-like cognition work
- **Georg Northoff** — Co-author, consciousness/brain dynamics theorist

### Key Output
- **Review paper**: Wolff, Choquette, Northoff, Iriki & Dumas, "Quantum Computing for Neuroscience: Theory, Methods and Opportunities." PsyArXiv, Sept 2025 (10.31234/osf.io/vw8n3). Not yet in a journal.
- **Two follow-up papers in pipeline**: Publication licenses signed July 2025 for "multiscale" and "path_integral" papers — likely submitted or in press.
- **Thesis**: Classical computing hits physical limits for analyzing large, multidimensional, non-stationary neural datasets. Quantum algorithms (QSVM, QFT, quantum walks) can reveal novel neural features computationally inaccessible to classical methods.

### Repo: ppsp-team/QuantumComp4Neuro
- GitHub: https://github.com/ppsp-team/QuantumComp4Neuro
- 0 stars, last push Aug 2025
- Contains: figure notebooks for the review paper (Qiskit 0.42.0), publication licenses
- Earlier versions had an EEG QSVM notebook and a QFT demo (deleted in Aug 2025 cleanup)
- **Replication value**: Low — figure generation only, no runnable experiments

### HyPyP (Hyperscanning)
- GitHub: https://github.com/ppsp-team/HyPyP (92 stars)
- Active maintenance (last push March 2026), but **no quantum extensions**
- Purely classical EEG/MEG/fNIRS inter-brain synchrony pipeline
- **Opportunity**: Add quantum walk-based inter-brain connectivity metrics using qcccm

### Connection to qcccm
- Dumas' group benchmarks quantum **speedups** for brain simulation; we build quantum **cognitive models**
- Iriki is the shared link to Khrennikov's quantum-like oscillatory cognition theory
- Their IBM QS1 Bromont access (UNIQUE $40K pilot) could run our circuits if we collaborate

---

## 2. QEEGNet (Chen et al., 2024-2025)

### Papers
- "QEEGNet: Quantum Machine Learning for Enhanced Electroencephalography Encoding." IEEE SiPS 2024. arXiv:2407.19214
- "Exploring the Potential of QEEGNet for Cross-Task and Cross-Dataset Electroencephalography Encoding." arXiv:2503.00080, March 2025

### What it does
- Hybrid quantum-classical architecture integrating quantum layers into EEGNet
- Outperforms classical EEGNet on **BCI Competition IV 2a** benchmark for most subjects
- Noise-robust under depolarizing and bit-flip noise models
- Cross-dataset generalization remains inconsistent

### Technical details
- Framework: likely PennyLane or TensorFlow Quantum (need to verify from repo)
- Dataset: BCI Competition IV 2a (motor imagery, 9 subjects, 4 classes)
- Quantum layers replace specific convolutional layers in EEGNet architecture

### Replication value: HIGH
- BCI IV 2a is a standard benchmark, publicly available
- We could replicate with PennyLane + JAX and compare against pyRiemann-qiskit pipeline
- Tests whether quantum layers genuinely help or if the improvement is from hyperparameter differences

---

## 3. pyRiemann-qiskit (Andreev, Cattan et al., 2023)

### Paper
Andreev, Cattan, Chevallier & Barthelemy. "pyRiemann-qiskit: A Sandbox for Quantum Classification Experiments with Riemannian Geometry." *Research Ideas and Outcomes* 9, March 2023. doi:10.3897/rio.9.e101006

### Repo
- GitHub: https://github.com/pyRiemann/pyRiemann-qiskit (31 stars, BSD 3-Clause)
- Qiskit ecosystem member, actively maintained (last commit March 2026)
- Lead maintainer: Gregoire Cattan (IBM Software, Krakow)

### What it does
- Bridges Riemannian geometry (SPD matrices from EEG covariance) with quantum classifiers
- **QuanticSVM** — quantum kernel via ZZFeatureMap
- **QuanticVQC** — variational quantum classifier with TwoLocal ansatz
- **QuantumStateDiscriminator** — models EEG mental states as density matrices, classifies via Pretty Good Measurement (PGM/POVM). Born-rule probabilities, no softmax.
- **ContinuousQIOCEClassifier** — QAOA batch classifier with angle encoding (new, v0.5.0)
- Full scikit-learn pipeline: `XdawnCovariances → TangentSpace → PCA → QuanticSVM`

### Key insight for qcccm
SPD matrices (EEG covariance) are density matrices after trace normalization. The `QuantumStateDiscriminator` makes this explicit:
```python
M = X @ X.T / X.shape[1]  # covariance
rho = M / np.trace(M)      # → density matrix
```
This is the same mathematical object as `qcccm.core.density_matrix`.

### Replication value: HIGH
- Multiple MOABB benchmarks available
- scikit-learn compatible pipeline is easy to compare against
- We could add a PennyLane-native version of QuantumStateDiscriminator to qcccm
- Could apply their pipeline to multi-agent behavioral covariances (not just EEG)

---

## 4. QSVM-QNN for BCI (Behera, Al-Kuwari, Farouk, 2025)

### Paper
arXiv:2505.14192, accepted at IEEE VTC Spring 2025 workshops

### What it does
- Hybrid QSVM + QNN model for EEG-based brain-computer interface
- Achieves **0.990 and 0.950 accuracy** on two benchmark EEG datasets
- Validated under six quantum noise models
- Outperforms both classical-only and standalone quantum models

### Replication value: MEDIUM
- Need to find the repo (if public)
- The noise model validation is interesting — parallels our `mitigation/zne.py` work
- Could test whether our ZNE improves their noisy circuit performance

---

## 5. Quantum Time-series Transformer for fMRI (2025)

### Paper
arXiv:2509.00711, August 2025

### What it does
- Quantum-enhanced transformer using Linear Combination of Unitaries (LCU) and Quantum Singular Value Transformation (QSVT) for resting-state fMRI
- **Polylogarithmic complexity** — potential advantage for neuroimaging's small-sample, high-dimensional data
- Works well with the small sample sizes typical of neuroimaging studies

### Replication value: HIGH (conceptual)
- Resting-state fMRI parcellations are essentially graphs → natural fit for quantum walks
- The small-sample advantage claim is directly testable
- Could combine with our `networks/topology.py` for brain graph construction

---

## 6. Quantum-Inspired EEG with Connectome Generation (2026)

### Paper
Scientific Reports, 2026. doi:10.1038/s41598-026-41821-8

### What it does
- Quantum entangled particle pattern-based feature extraction from EEG
- Over **90% accuracy** on six EEG datasets
- Generates **connectome diagrams** as interpretable outputs
- One of the few papers connecting quantum methods to connectomics visualization

### Replication value: MEDIUM
- "Quantum-inspired" (not actual quantum circuits) — classical algorithms mimicking quantum patterns
- The connectome generation is interesting for our brain network quantum walk work

---

## 7. Quantum DeepONet (Lu Lu Group, Yale, 2025)

### Paper
Xiao, Zheng, Jiao, Yang & Lu. "Quantum DeepONet: Neural operators accelerated by quantum computing." *Quantum* 9:1761, June 2025. arXiv:2409.15683

### Repo
- GitHub: https://github.com/lu-group/quantum-deeponet (20 stars, Apache-2.0)
- Dependencies: DeepXDE v1.10.1 + Qiskit
- Contains: data generation scripts (ODE, Poisson, advection, Burgers), source for data-driven and physics-informed quantum DeepONet

### What it does
- Replaces DeepONet branch/trunk networks with QOrthoNN (quantum orthogonal neural networks)
- Unary encoding + pyramidal RBS gate circuits → O(n/δ²) evaluation (vs O(n²) classical)
- **Train classically** (DeepXDE/JAX), **evaluate quantumly** (Qiskit) — avoids barren plateaus
- QPI-DeepONet: physics-informed variant with PDE loss terms
- Results: 0.15-2.25% relative L² errors on standard operator learning benchmarks
- Tested under depolarizing noise with post-selection error mitigation

### Connection to qcccm
- The "train classical, evaluate quantum" pattern matches our autoresearch architecture
- Could learn time-evolution operators of social Hamiltonians from Metropolis trajectories
- DeepXDE has a JAX backend — compatible with our stack
- The physics-informed variant could enforce conservation laws in social dynamics

### Replication value: HIGH
- Clean repo with runnable examples
- Apache-2.0 license
- Could extend to learn spin glass dynamics operators

---

## 8. Other Notable Efforts

### IBM + Inclusive Brains (June 2025)
- Joint study: IBM Granite models + quantum ML for brain-machine interfaces
- Classifying and interpreting brain activity using quantum techniques
- Automatic algorithm selection per individual

### Google Quantum Neuroscience Program
- Led by Hartmut Neven
- ~$100K grants to probe quantum effects in neural function
- Focus on quantum sensing + quantum computing for real-time neural monitoring
- More quantum biology than quantum computing for data analysis

### Hybrid Quantum-Classical Epilepsy EEG (Scientific Reports, Jan 2026)
- CWT-based time-frequency + quantum-inspired neural layers for seizure classification

### QSVM for Schizophrenia Detection (J. Medical Systems, 2024)
- QSVM on IBM Quantum Lab (QasmSimulator, 32 qubits)
- Pauli feature maps
- 100% accuracy (limited dataset caveat)

### Fusion Quantum VAE for Brain-Heart Signals (2025)
- Multimodal EEG+ECG quantum variational autoencoder
- 97.8% accuracy for mental health classification

### NeuroQ: Quantum Brain Emulation (Biomimetics, Aug 2025)
- Reformulates FitzHugh-Nagumo neuron via stochastic mechanics → Schrödinger-like equation
- Conceptual only, no implementation

---

## 9. The Gap: What No One Has Done

The systematic review of quantum deep learning in neuroinformatics (Artificial Intelligence Review, Feb 2025) explicitly states:

> **"QDL remains completely unexplored in connectomics."**

Specifically, no one has:
1. Applied **quantum walks to actual brain network data** (structural or functional connectomes)
2. Used **quantum PageRank** on brain graphs to identify hub regions
3. Applied **quantum community detection** (QAOA/D-Wave) to brain parcellations
4. Used **quantum state discrimination** (POVM/PGM) to classify brain states from connectome features
5. Compared **quantum vs classical centrality** measures on real connectome data

Our `qcccm` modules — `core/quantum_walk.py`, `networks/`, `spin_glass/` — are positioned to be first in all five areas.

---

## 10. Replication Priority List

| Priority | Target | Source | Framework Needed | Effort |
|----------|--------|--------|-----------------|--------|
| 1 | Quantum walk on brain connectome | Novel (gap) | qcccm quantum_walk + networks | Medium |
| 2 | pyRiemann-qiskit pipeline on BCI data | Andreev et al. 2023 | pyRiemann-qiskit + qcccm bridge | Low |
| 3 | QEEGNet replication on BCI IV 2a | Chen et al. 2024 | PennyLane + JAX | Medium |
| 4 | Quantum DeepONet for spin dynamics | Xiao et al. 2025 | DeepXDE + qcccm spin_glass | High |
| 5 | Quantum state discrimination for agent classification | Novel (extending pyRiemann) | qcccm density_matrix + circuits | Medium |
| 6 | QAOA community detection on connectome | Novel (gap) | qcccm spin_glass + QAOA | High |

---

## 11. Key References

### Dumas Group
- Wolff et al. (2025). "Quantum Computing for Neuroscience." PsyArXiv 10.31234/osf.io/vw8n3
- ppsp-team/QuantumComp4Neuro: https://github.com/ppsp-team/QuantumComp4Neuro
- ppsp-team/HyPyP: https://github.com/ppsp-team/HyPyP

### Quantum EEG/BCI
- Chen et al. (2024). "QEEGNet." arXiv:2407.19214
- Chen et al. (2025). "Cross-Task QEEGNet." arXiv:2503.00080
- Behera et al. (2025). "QSVM-QNN for BCI." arXiv:2505.14192
- Andreev, Cattan et al. (2023). "pyRiemann-qiskit." RIO Journal doi:10.3897/rio.9.e101006

### Quantum Neuroimaging
- Quantum fMRI Transformer. arXiv:2509.00711
- Quantum-inspired EEG with connectome. Scientific Reports 2026, doi:10.1038/s41598-026-41821-8
- Quantum DL in neuroinformatics review. AI Review, Feb 2025, doi:10.1007/s10462-025-11136-7

### Quantum Operator Learning
- Xiao et al. (2025). "Quantum DeepONet." Quantum 9:1761, arXiv:2409.15683
- lu-group/quantum-deeponet: https://github.com/lu-group/quantum-deeponet

### Quantum Cognition Theory
- Busemeyer & Bruza (2012). *Quantum Models of Cognition and Decision*
- Pothos & Busemeyer (2013). "Can quantum probability provide a new direction?" Psych. Review
- Khrennikov (2010). *Ubiquitous Quantum Structure*
- Khrennikov, Iriki & Basieva (2025). QL oscillatory cognition

### Sociophysics
- Mullick & Sen (2025). "Sociophysics models inspired by the Ising model." arXiv:2506.23837
- Challet, Marsili, Zecchina (2000). "Statistical Mechanics of Minority Games." PRL 84:1824
- Brock & Durlauf (2001). "Discrete Choice with Social Interactions." RES 68:235
