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

## Status
- [x] Gaussian-limit identity (CMI = ∂L/∂q̇ = FEP momentum) — proven & tested
- [ ] EEG state → density matrix adapter (`q`-parameterized)
- [ ] quantum-vs-classical momentum sweep over `q` on TRAIN/TEST/FULL
