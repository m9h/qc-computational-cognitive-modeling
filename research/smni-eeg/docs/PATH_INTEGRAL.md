# The path-integral spine: SMNI action ↔ free-energy action

This is the technical center of the project. The bet is that Ingber's SMNI and Friston's
2023 path-integral FEP are **the same least-action object** evaluated with different
interpretations, and that the `SMNI_CMI` EEG lets us make the correspondence *computational*
rather than rhetorical.

## 1. Two actions, one form

**Ingber / SMNI.** Mesocolumnar firing fields `M^G(t)` (G = excitatory/inhibitory) evolve by a
short-time conditional probability with a Lagrangian `L_SMNI`, and the long-time density is a
path integral:

```
P[M(t)] ∝ ∫ DM  exp( − ∫ dt  L_SMNI(M, Ṁ) )
```
with action `A_SMNI = ∫ dt L_SMNI`. Fits to EEG are done by **ASA** maximizing the likelihood of
observed paths. **Canonical momenta** `Π_G = ∂L_SMNI / ∂Ṁ^G` are the **CMI** (Ingber 1997).

**Friston / FEP (2023).** A particle's trajectory `x(t)` (partitioned into internal `μ`, blanket =
sensory `s` + active `a`, external `η`) extremizes a free-energy **action** via least action:

```
x*(t) = argmin  ∫ dt  𝓛_FEP(x, ẋ)          (path integral / principle of least action)
```
where internal flows can be read as gradient descent on variational free energy `F`, with
conjugate momenta on the belief manifold.

**The claim to test:** there is a map `L_SMNI ↔ 𝓛_FEP` under which **CMI ≡ the conjugate momenta**
of the FEP action, i.e. the CMI are an *observable read-out of the particle's momenta*.

## 2. Correspondence table (what maps to what)
| SMNI | FEP path-integral | Status |
|---|---|---|
| firing fields `M^G(t)` | internal + blanket states `x(t)` | structural, plausible |
| `L_SMNI`, action `A_SMNI` | `𝓛_FEP`, free-energy action | **central hypothesis** |
| CMI `Π_G = ∂L/∂Ṁ` | conjugate momenta of beliefs | **central hypothesis** |
| ASA likelihood fit of paths | least-action / variational density dynamics | methodological analog |
| EEG channels | blanket (sensory) states | modeling choice |
| Ca²⁺ `p` coupled via `A` (`Π=p+qA`) | internal-state → active-state dependence | makes column **strange** |

## 3. Why this makes the column a *strange* particle
Friston: a **strange particle infers its own actions** (active states depend on internal states).
In SMNI+calcium, the macrocolumn's **internal** Ca²⁺ momentum `p` is driven by the EEG field `A`
that the column's *own* firing helped produce, and Ca²⁺ in turn modulates firing (the **active**
output). Internal → active dependence is present by construction ⇒ the column satisfies Friston's
*strange* criterion. The `p + qA` coupling is thus not just calcium physics — it's the concrete
mechanism that would make a cortical particle "strange."

## 4. Computational program (on the dataset, DGX Spark)
1. **Estimate CMI** from `SMNI_CMI` trials (Ingber 1997 recipe): fit `L_SMNI` per
   subject/condition, extract `Π_G(t)` = CMI time series. Deliver as the empirical "momenta."
2. **Cast as a particle**: define internal/blanket partition (EEG = blanket), build the FEP action
   for the same trials; check whether the FEP conjugate momenta align (up to a transform) with CMI.
3. **Test the map**: compare `A_SMNI`-optimal paths vs FEP least-action paths on held-out trials
   (we have a clean TRAIN/TEST split, 10/10 each). Quantify agreement of momenta and of
   alcoholic-vs-control / S1–S2 discrimination.
4. **Strange-particle probe**: add the `p + qA` internal→active loop and test whether the
   resulting dynamics are better described as self-inferred action (FEP-strange) than as a passive
   Lagrangian — i.e. is "strangeness" *derived* or *imposed*?
5. **Quantum extension (speculative, later)**: qPATHINT propagation of the Ca²⁺ packet as a
   separate, clearly-flagged track.

## 5. Open theoretical questions (the discussion)
- Is `𝓛_FEP` a *reparameterization* of `L_SMNI`, or genuinely different dynamics that merely share
  a least-action skeleton?
- Are CMI = momenta of *beliefs* (FEP) or momenta of *physical fields* (SMNI) — and is that a
  distinction with an empirical difference on this data?
- Does minimizing SMNI action entail minimizing variational free energy, or only coincide at
  extrema? (Conservative vs dissipative particle question in Friston 2023.)
- At which blanket (Ca²⁺ / synaptic / columnar / scalp) does "inference" first become a useful
  description vs. just statistical mechanics?

## References
- Friston, Da Costa, Sajid, Heins, Ueltzhöffer, Pavliotis, Parr (2023), *Phys. Life Rev.*,
  arXiv:2210.12761 — path-integral FEP, particle taxonomy, strange particles.
- Friston (2019), arXiv:1906.10184 — "A free energy principle for a particular physics."
- Ingber (1997), *Phys. Rev. E* 55(4):4578 — SMNI canonical momenta indicators.
- Ingber, Pappalepore, Stesiak (2014), *J. Theor. Biol.* 343:138 — EEG→Ca²⁺ `p+qA`.
- See `HISTORY.md` (full bibliography) and `FRISTON_BRIDGE.md` (conceptual bridge).
