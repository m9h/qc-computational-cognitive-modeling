# Code & references

Living index of code (cloned + external) and the research lineages this project connects.

## 1. Ingber's own code — cloned in `refs/ingber/` (all BSD-3)
| Repo | Role in roadmap |
|---|---|
| `adaptive-simulated-annealing` | ASA global optimizer (C). The SMNI Lagrangian fitter. Ships `asa.c`, `asa_usr.c`, and `asa_test_asa`/`asa_test_usr` drivers. Maintained to Apr 2024. |
| `qPATHINT` | quantum path-integral propagator (+ historic `PATHINT_1995_STM`). The destination of the roadmap. |
| `qPATHTREE` | path-tree variant of qPATHINT. |
| `EEG_qCa` | EEG quantum-calcium code (closest to our dataset). |
| `qMAXIMA` | Maxima symbolic-math for the quantum derivations. |
| `spline-Laplacian` | EEG surface-Laplacian preprocessing (usable now). |

Contact / status: no obituary found; archive active to 2024. Emails on record:
`ingber@ingber.com`, `ingber@alumni.caltech.edu`, `ingber@caa.caltech.edu`.

## 2. The optimization lineage (ASA → modern)
Ingber used **ASA** because the SMNI path-integral likelihood is multimodal and he had no
gradients in the 1990s. ASA is *stochastic global, derivative-free*. Modern work splits two ways:

### 2a. Convex / operator-splitting attack — **Boyd lab & descendants**
Not annealing; the convex-relaxation alternative for the same non-convex problem class. This is
the GPU-ADMM frontier (Morgan worked with Boyd's lab on GPU ADMM).
- **ADMM** — Boyd, Parikh, Chu, Peleato, Eckstein (2011), *Distributed Optimization and
  Statistical Learning via ADMM*. The survey that revived operator splitting.
- **OSQP** — Stellato, Banjac, Goulart, Bemporad, Boyd (2020), *Math. Prog. Comp.* QP via ADMM;
  GPU variant **cuOSQP**.
- **SCS** (Splitting Conic Solver) — O'Donoghue, Chu, Parikh, Boyd (2016), *JOTA*; GPU-capable.
  (Brendan O'Donoghue: Boyd student → DeepMind — a bridge name between convex opt and RL/inference.)
- **Non-convex ADMM / DCCP** — Boyd (2011) §9 consensus & sharing; **Disciplined Convex-Concave
  Programming**, Shen, Diamond, Gu, Boyd (2016): convexify → split → iterate on non-convex problems.
- **Differentiable convex optimization** — `cvxpylayers` / `diffcp`, Agrawal, Amos, Barratt, Boyd,
  Busseti, Diamond (NeurIPS 2019). Embeds convex solves inside autodiff → the bridge to JAX/GPU.
- Ecosystem: **CVXPY** (`cvxgrp`), now with Disciplined Nonlinear Programming (DNLP) extension.

### 2b. Stochastic-global successors to ASA — other top groups
The direct spiritual descendants of simulated annealing for continuous non-convex search.
- **CMA-ES** — Hansen & Ostermeier (2001), INRIA. De-facto modern derivative-free optimizer;
  uses a *maximum-likelihood* update of a search distribution. JAX: **evosax** (R. Lange, DeepMind),
  CLINAMEN2.
- **Bayesian optimization** — **BoTorch / Ax** (Balandat et al., Meta, NeurIPS 2020), Google
  **Vizier**, Dragonfly (CMU). Replaces SA when evaluations are expensive.
- **Nevergrad** (Meta, Rapin & Teytaud) — derivative-free platform bundling CMA-ES/BO/ES.
- **Annealing hardware lineage** — parallel tempering, population annealing, quantum annealing
  (D-Wave), Fujitsu Digital Annealer.
- **MPPI** (Model Predictive Path Integral control) — Williams, Aldrich, Theodorou (Georgia Tech);
  recently shown ≈ preconditioned gradient descent (arXiv 2603.24489). Notable here because it is
  literally a *path-integral* sampling optimizer — resonant with SMNI/qPATHINT.

### 2c. Synthesis for our fit (the JAX rewrite)
The SMNI path-integral likelihood is now **autodifferentiable**. Plan:
gradient methods (`optax`/JAXopt) for the smooth part → a **global layer** (CMA-ES via `evosax`,
or BO via BoTorch) for multimodality → **ADMM/operator-splitting (OSQP/SCS, GPU)** where
subproblems are convex — i.e. replace ASA with a modern, differentiable, GPU-capable stack.

## 3. Friston / Active Inference ecosystem (FEP side of the bridge)
See `FRISTON_BRIDGE.md` / `PATH_INTEGRAL.md` for the theory link.
- **pymdp** — `github.com/infer-actively/pymdp` (Heins et al. 2022). Active inference for
  POMDPs / discrete state spaces.
- **Active Inference Institute** — `activeinference.org`, `github.com/ActiveInferenceInstitute`
  (e.g. ActiveBlockference). President: **Daniel Ari Friedman** (danielarifriedman.com).
- **RxInfer.jl** — BIASlab (Bert de Vries, TU Eindhoven): reactive message-passing variational
  inference (Julia) — relevant to the variational-density side of FEP.
- **SPM / DEM** — Friston's own MATLAB lineage (dynamic expectation maximization, variational
  Laplace) — the historical FEP fitting code.

## 4. How the pieces line up
| Layer | Ingber (classical) | Modern / this project |
|---|---|---|
| Model | SMNI Lagrangian, path integral | same, in JAX (autodiff) |
| Fit | ASA (stochastic global) | gradient + CMA-ES/BO global layer; ADMM where convex |
| Features | CMI (canonical momenta) | CMI ≡ FEP conjugate momenta (hypothesis) |
| Inference frame | statistical mechanics | active inference (pymdp / RxInfer / SPM) |
| Compute | XSEDE supercomputers | DGX Spark (qPATHINT); laptop for CMI |
