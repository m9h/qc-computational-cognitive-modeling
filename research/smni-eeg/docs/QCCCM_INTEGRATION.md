# Integrating smni-eeg into QCCCM

Morgan's **QCCCM** (`qc-computational-cognitive-modeling`, MIT, cloned at `refs/qcccm/`) is the
intended host framework — it already provides the JAX+PennyLane stack this project was going to
build. So the plan shifts from "standalone JAX rewrite" to **"add SMNI/CMI/qPATHINT as a model +
data domain inside QCCCM."** Below is the concrete module map.

## What QCCCM already has (no need to build)
| Need (from our docs) | QCCCM module / symbol | Replaces |
|---|---|---|
| Fit SMNI params without ASA | `qcccm/fitting/mle.py` → `fit_mle`, `MLEConfig`, `model_comparison` | Ingber's ASA + Nelder–Mead, 6-day single-core runs |
| Log-likelihoods (JAX, `grad`-ready) | `qcccm/fitting/likelihoods.py` | hand-derived analytic fits |
| Path-integral propagation | `qcccm/spin_glass/solvers_jax.py` → `_pimc_sweep`, `transverse_field_mc_jax` (PIMC + Trotter, `jit`/`vmap`/`scan`) | PATHINT/qPATHINT banded-kernel C folding |
| FEP / quantum bridge | `qcccm/models/bridge.py` → `beliefs_to_density_matrix`, `density_matrix_to_beliefs`, `quantum_efe`, `stochastic_to_unitary` (Szegedy) | our `FRISTON_BRIDGE.md` made executable |
| Quantum Active Inference agent | `qcccm/models/alf_bridge.py` → `QuantumEFEAgent` (ALF POMDP→density matrices, `quantumness` q∈[0,1]) | pymdp interop, the FEP side |
| Quantum primitives | `qcccm/core/` (density matrices, quantum walks, states) | — |
| Hardware / annealing | `qcccm/annealing/` (QUBO, D-Wave Ocean); PennyLane→hardware | Ingber's unfulfilled D-Wave/Rigetti port |
| Neural data IO | `qcccm/neuroai/` (DANDI/NWB) | (wrap our `.rd` loader) |
| Error mitigation, viz, benchmarks | `qcccm/mitigation` (ZNE), `viz`, `benchmarks` | — |

QCCCM ships **281 tests with physics-invariant checks** (classical limits, energy bounds,
thermodynamic consistency) — the reproducibility/test-harness gap from `CRITIQUE_AND_FUTURE.md` is
already solved on the host side.

## What we add to QCCCM
1. **`qcccm/models/smni.py`** — the SMNI Lagrangian `L` (eq. 4–5 of `smni21_hybrid`), drift `gᴳ`,
   diffusion `g_{GG'}`, and **CMI** `Π = ∂L/∂q̇` (port of our `src/cmi.py`, which already matches
   Ingber's `Π=Σ⁻¹(Ṁ−g)` definition). Pure JAX → differentiable, `vmap` over trials.
2. **SMNI likelihood** in `qcccm/fitting/likelihoods.py** — the short-time Gaussian path-integral
   log-likelihood whose stationary momenta are the CMI; fit with the existing `fit_mle`.
3. **SMNI/EEG data adapter** in `qcccm/neuroai/` — wrap `smni-eeg/src/load_rd.py` (gzip-aware
   `.rd` → `(trials, chan, time)`), exposing the alcoholism dataset to QCCCM pipelines.
4. **qPATHINT propagator** — build on `spin_glass/solvers_jax.py` PIMC/Trotter: complex-variable
   kernel folding for the Ca²⁺ wave-packet with serial shocks (tier A/B of `QPATHINT_REIMAGINED.md`).
5. **CMI ↔ EFE experiment** — the project's central hypothesis becomes a runnable QCCCM benchmark:
   compute CMI from EEG (our pipeline) and compare to the conjugate momenta / `quantum_efe` of the
   `bridge.py` density-matrix beliefs on the same trials.

## The bridge, now executable
Our `PATH_INTEGRAL.md` claim — **CMI ≡ conjugate momenta of the FEP free-energy action** — stops
being a hypothesis-on-paper: `bridge.py` already maps beliefs→density matrices and computes
quantum expected free energy, and `alf_bridge.py` runs an active-inference agent. So we can
literally fit SMNI to EEG (→ CMI) and, on the same data, evaluate the QCCCM EFE momenta, and test
whether they align. That is the experiment the whole project was pointing at.

## Teaching ladder ↔ QCCCM curriculum
QCCCM already has 6 progressive notebooks (`01_bits_to_qubits` … `06_quantum_neuroai_pipeline`) and
a `CURRICULUM.md`, with "no QM prerequisites." The MPPI→PATHINT→SMNI/CMI→qPATHINT ladder from
`CRITIQUE_AND_FUTURE.md` slots in as the **path-integral track** (e.g. `07_path_integrals_mppi`,
`08_pathint_smni_cmi`), reusing `solvers_jax.py` and the new `smni.py`. "Path integrals for general
computation" becomes a curriculum strand: MPPI (control) → PIMC (already in repo) → PATHINT (density)
→ qPATHINT (amplitude) → quantum active inference (`bridge.py`).

## Recommended first PR
Port `src/cmi.py` → `qcccm/models/smni.py` + an SMNI likelihood in `fitting/likelihoods.py`, fit the
FULL dataset with `fit_mle`, and reproduce our **t≈−11 group separation** through the QCCCM path —
validating the integration against the result we already have. Then wire the CMI↔EFE comparison.

## Open questions for Morgan
- Develop QCCCM as the home and keep `smni-eeg/` as the data/docs satellite, or vendor smni-eeg
  modules into QCCCM directly?
- Is "ALF" in `alf_bridge.py` an external active-inference lib to align with pymdp, or your own?
