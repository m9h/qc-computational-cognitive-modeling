# Bridge: SMNI / calcium-from-EEG ↔ Friston's path-integral FEP & strange particles

**Goal.** Use the concrete, *data-anchored* Ingber program (SMNI + CMI + the EEG→Ca²⁺ coupling,
fit to the `SMNI_CMI` recordings) as an entry point for discussing:

> Friston, Da Costa, Sajid, Heins, Ueltzhöffer, Pavliotis & Parr (2023),
> **"Path integrals, particular kinds, and strange things,"** *Physics of Life Reviews*
> (arXiv:2210.12761 · PubMed 37703703).

The reason this is a good anchor and not a loose analogy: **both frameworks are least-action
formulations expressed as path integrals over densities.** They share the math; they differ in
interpretation. That gap is exactly what's worth talking about.

## What the Friston paper says (verbatim core)
> "This paper describes a path integral formulation of the free energy principle... a method or
> principle of least action... Particles are defined by a particular partition, in which internal
> states are individuated from external states by active and sensory blanket states... **Strange
> particles can be described as inferring their own actions, endowing them with apparent autonomy
> or agency.**"

Particle taxonomy: dissipative vs conservative; inert vs active; **ordinary vs strange**.
- **Ordinary** active particle: active states depend only on blanket states (reactive).
- **Strange** particle: active states depend on **internal/autonomous states** — the particle
  *infers its own actions*. This self-referential loop is what reads as sentience/agency.

## The shared spine: path integrals & least action
| | Ingber SMNI | Friston FEP (2023) |
|---|---|---|
| Object | path integral of multivariate conditional probabilities | path integral over particle trajectories |
| Extremized quantity | SMNI **Lagrangian / action** (fit by ASA) | **free-energy action** (principle of least action) |
| Fit/solve | Adaptive Simulated Annealing on real EEG | variational density dynamics |
| Output features | **Canonical Momenta Indicators (CMI)** | gradient flows / conjugate momenta on belief manifolds |

So Ingber's **CMI are literally canonical momenta** of the action; Friston's formulation also runs
on conjugate momenta of a least-action path integral. The bridge claim to test: *does minimizing
the SMNI action correspond (or map) to minimizing a variational free-energy action, and are CMI a
measurable read-out of the "momenta" Friston's particles carry?*

## Markov blankets ↔ Ingber's scales (the concrete part)
Ingber's multi-scale stack gives a **physical instantiation** of nested Markov blankets:

```
molecular Ca²⁺ waves  ──(blanket)──  synaptic / tripartite junction  ──  macrocolumn (SMNI)  ──  scalp EEG
```

The **EEG → Ca²⁺ coupling `Π = p + qA`** (EEG field as magnetic vector potential `A`, Ca²⁺ charge
`q = −2e`) is a concrete, quantified **top-down flow across a blanket**: a macroscopic blanket state
(EEG `A`) influencing an internal molecular momentum (`p` of Ca²⁺). That is precisely the kind of
active/sensory cross-boundary coupling the FEP needs, made physical and (per Ingber) computable.

## Where the "strange particle" lands
Candidate strange particle: a **neocortical macrocolumn / neuron** whose firing (active/blanket
output) is modulated by its **internal Ca²⁺ state**, which is in turn shaped by the column's own
prior firing via the `A`-field. Internal states (Ca²⁺ momenta) influence active states (firing) →
the column **"infers its own actions"** → *strange* in Friston's exact sense. SMNI supplies the
empirically-fit dynamics; the calcium coupling supplies the internal→active dependence.

## Concrete dataset hook
The `SMNI_CMI` EEG (see `DATASET.md`) lets us stop hand-waving:
- Compute **CMI** per Ingber (1997) on real trials → candidate "momenta" of cortical particles.
- Treat scalp channels as **blanket/sensory states**; ask what internal dynamics best explain them
  under (a) SMNI least-action and (b) a variational free-energy least-action — and compare.
- **Alcoholic vs control** and **S1 / S2-match / S2-nomatch** become different parameterizations /
  priors of the *same* particle dynamics — a place to test whether group/condition differences look
  like different inference (FEP) or different Lagrangian parameters (SMNI).

## Honest divergences (the discussion's live edges)
- **Interpretation:** Friston reads internal dynamics as *Bayesian inference*; Ingber reads them as
  *nonequilibrium statistical mechanics* with no commitment to inference. Same equations, different
  semantics — the core thing to argue about.
- **Speculative tiers:** Ingber's *classical* `p + qA` coupling and SMNI/CMI fits are well-grounded;
  the *quantum* qPATHINT/consciousness claims are explicitly speculative. Friston's strange-particle
  "agency" is a formal/interpretive construct, not a measured quantity. Keep these tiers separate.
- **Open question:** is "strange" (self-inferred action) *derivable* from the SMNI+Ca²⁺ dynamics, or
  an extra interpretive layer placed on top?

## References
- Friston et al. 2023, *Phys. Life Rev.* — arXiv:2210.12761.
- Friston 2019, "A free energy principle for a particular physics" — arXiv:1906.10184.
- Ingber 1997, *Phys. Rev. E* 55(4):4578 (CMI). Ingber et al. 2014, *J. Theor. Biol.* 343:138
  (EEG→Ca²⁺). See `HISTORY.md` for the full list.
