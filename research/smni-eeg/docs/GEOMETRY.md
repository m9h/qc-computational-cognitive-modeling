# Coordinate-free / covariant view of SMNI–CMI (and why it dictates the features)

Generalized coordinates (Friston's jet `(M, Ṁ, M̈, …)`; Lagrangian `q,q̇`) are a **chart**.
The invariant content lives in coordinate-free objects, and *only invariant quantities are
legitimate features*. This note states the geometry and the feature program it forces.

## 1. The intrinsic objects
- Configuration manifold `Q`; states on the **tangent bundle** `TQ`; Lagrangian `L: TQ → ℝ`.
- **Momenta = fiber derivative (Legendre transform)** `𝔽L: TQ → T*Q`. A canonical momentum — the
  **CMI** — is intrinsically a **covector** `p = 𝔽L(v) ∈ T*Q`; this is chart-free *as an object*.
- Friston's generalized coordinates are a trivialization of the **jet bundle** `Jᵏ(ℝ,Q)`; the shift
  operator `D` is the coordinate form of the intrinsic total-derivative (Cartan) structure.

## 2. Ingber's path integral is already covariant
The full SMNI Lagrangian (smni21_hybrid p.10) is
`L = ½ (q̇ⁱ − gⁱ) g_{ii'} (q̇^{i'} − g^{i'}) + R/6`, with metric `g_{ii'} = (g^{ii'})⁻¹`,
measure `√g` (`g = det g_{ii'}`), and **scalar curvature `R/6`**. The `R/6` + `√g` are the
signature of the Graham/Langouche **covariant** path integral — built to transform correctly under
*nonlinear* changes of variable. So SMNI is intrinsically a Riemannian (Fisher–Rao) construction;
the per-channel Cartesian chart we used (and the dropped `R/6`) is one flat trivialization.

By **Čencov's theorem** the Fisher–Rao metric is the unique reparameterization-invariant metric on
the model manifold — so the geometry is canonical and the privileged features are its invariants.

## 3. The consequence: which features are legitimate
A feature is meaningful iff it is **coordinate-free** (invariant to relabeling/remixing the state,
e.g. EEG re-referencing or linear channel mixing `M → PM`).
- The momentum 1-form is intrinsic, but its **components** `Σ⁻¹(Ṁ−g)` and the scalar **`mean|CMI|`
  are chart-dependent** — not invariant to `P`. (A second reason that feature was fragile.)
- Under remixing the drift Jacobian transforms by **similarity** `∂g/∂M → P(∂g/∂M)P⁻¹`, so its
  **spectrum is invariant**. Legitimate coordinate-free features:
  - **divergence** `∇·g = tr(∂g/∂M)` (contraction rate),
  - **spectral abscissa / radius**, eigenvalue real parts (decay rates) and imaginary parts
    (oscillation rates),
  - spectral invariants of the CMI covariance; the metric determinant `√g`.

## 4. The FEP tie-in (and a live critique)
Generalized coordinates buy tractability (the flow → linear shift `D`, local Laplace inference);
coordinate-freedom buys *meaning*. Whether FEP quantities (free energy, Markov blankets) are
reparameterization-invariant is an active critique (Biehl, Aguilera, …) — "generalized vs
coordinate-free" is precisely that axis. Our discipline: report only invariants.

## 5. Empirical test — do invariants recover what `mean|CMI|` missed?
Per-subject drift fit → flow-invariant features (§3), leave-one-subject-out CV, vs chart-dependent
baselines. (`scripts/invariant_features_cv.py`.)

| feature (subject-level LOO, 122 subjects) | AUC |
|---|---|
| coordinate-free flow invariants (divergence, spectrum) | **0.343** |
| mean\|CMI\| per channel (chart-dependent) | 0.515 |
| variance per channel (chart-dependent) | **0.616** |

**The prediction was wrong.** Coordinate-free flow invariants did *not* recover group signal —
they are not above chance (numerically below, 0.343). mean|CMI| is at chance (0.515). Only raw
amplitude/variance generalizes (0.616). So the invariance framing is the *right methodology* — it
is the legitimate way to pose the question — but the answer is still negative: the fitted SMNI
dynamics, in **every** form tested (raw CMI, chart-dependent magnitude, *and* reparameterization-
invariant flow spectrum), do not encode the cross-subject group difference. That difference lives
in signal amplitude, which the dynamical/whitened features discard. Recording the failed
prediction is the point of doing it this way.
