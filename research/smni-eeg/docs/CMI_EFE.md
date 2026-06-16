# CMI ≡ free-energy momenta: the analytic core (Step 3)

The central hypothesis (`PATH_INTEGRAL.md`) is that Ingber's **CMI** are the conjugate
momenta of Friston's free-energy **action**. This note records what is *proven* vs. what
remains an *experiment*.

## Proven (Gaussian / Laplace limit) — in code, tested
For the SMNI short-time Lagrangian `L = ½ (q̇−g)ᵀ Σ⁻¹ (q̇−g)`, the conjugate momentum is
`Π = ∂L/∂q̇ = Σ⁻¹(q̇−g)` — the CMI. Under a Laplace-encoded variational free energy with a
Gaussian generative model, the free energy is the precision-weighted prediction error
`½ εᵀΣ⁻¹ε`, `ε = q̇−g` (generalized motion − flow), whose velocity-gradient is the same
`Σ⁻¹ε`. So **at this order CMI and the FEP conjugate momentum are literally identical**.

`qcccm/models/cmi_efe.py` computes `Π` by `jax.grad` of the Lagrangian and
`tests/test_cmi_efe.py` asserts it equals `smni.canonical_momenta` (closed form). Verified.

## The experiment (where it could break — "what quantum adds")
The identity holds only for a *diagonal* (decohered) belief state. The interesting question
is whether a **coherent** state explains EEG/behaviour better:

1. Represent each trial's state as a density matrix `ρ` (via `neuroai` /
   `bridge.beliefs_to_density_matrix`), with `quantumness q∈[0,1]` controlling off-diagonal
   coherences (`alf_bridge.beliefs_to_quantum_state`).
2. Compute the **quantum** free-energy momentum from `bridge.quantum_efe` and compare to the
   classical CMI as `q` sweeps 0→1.
3. At `q=0` the two coincide (proven above). Measure whether `q>0` improves fit /
   alcoholic-vs-control discrimination on the SMNI_CMI data.

If `q>0` adds nothing, that's a clean negative result on quantum cognition for this dataset;
if it does, it localizes *where* quantum matters. Either way the claim is falsifiable — the
whole point of grounding Ingber's program in runnable, tested code.

## Results — the quantum-coherence sweep (run on real EEG)
`qcccm/neuroai/coherence.py` + `scripts/quantum_coherence_sweep.py`. Each trial's channel
covariance → density matrix `ρ`; `q` blends `(1−q)·diag(ρ) + q·ρ`; feature = von Neumann
entropy; Welch t (alcoholic vs control) per `q`:

| set | q=0 | q=0.25 | q=0.5 | q=0.75 | q=1 |
|---|---|---|---|---|---|
| TRAIN | −5.37 | −5.16 | −4.53 | −3.03 | +0.21 |
| TEST (held-out) | −6.58 | −6.35 | −5.67 | −4.04 | −0.20 |
| FULL | −13.94 | −10.81 | −4.01 | +6.03 | **+18.23** |

**Verdict (held-out TEST is the arbiter):** the classical diagonal signal (q=0) **replicates**
held-out (|t|=6.6); the pure-coherence signal (q=1) **does not** (|t|=0.2 on both TRAIN and
TEST). FULL shows a large q=1 effect (+18) but it is **in-sample, sign-flipped, and cancels at
q≈0.5** — an independent axis that does not generalize.

**Conclusion:** for this dataset, off-diagonal coherence adds **no generalizing discriminative
power** beyond the classical variance structure — a clean negative result on "quantum adds
something" here. (Consistent with the earlier read that Ingber's quantum layer is a
speculative-but-falsifiable hypothesis, not a fit-improver.) The `bridge.quantum_efe`
policy-level test remains open for a task with explicit policy structure.

## Status
- [x] Gaussian-limit identity (CMI = ∂L/∂q̇ = FEP momentum) — proven & tested
- [x] EEG state → density matrix adapter (`q`-parameterized) — `coherence.py`, tested
- [x] quantum-vs-classical sweep over `q` on TRAIN/TEST/FULL — negative (held-out)
- [ ] policy-level `quantum_efe` test (needs a task/policy structure)
