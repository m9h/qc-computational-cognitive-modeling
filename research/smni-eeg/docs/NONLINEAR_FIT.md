# Nonlinear SMNI drift via fit_mle (Step A) — result

Tests whether Ingber's SMNI `tanh` drift nonlinearity helps on the SMNI_CMI EEG, and
demonstrates the modern fitter. Per-channel drift `g = a·M + b + β·tanh(γ·M)` fit by
**autodiff MLE** (`qcccm.fitting.mle.fit_mle`); linear per-channel baseline (β=0) is the
nested comparison. Fit on TRAIN, evaluated held-out on TEST.

| model | params | train NLL | AIC | TRAIN t | TEST (held-out) t |
|---|---|---|---|---|---|
| linear (per-channel) | 128 | 59,142,904 | **118,286,064** | −6.04 | **−7.40** |
| nonlinear (tanh) | 256 | 59,142,848 | 118,286,208 | −6.04 | −7.40 |

## Findings
1. **The fitter works — `fit_mle` replaces ASA.** A 64-channel nonlinear MLE fits in
   ~minutes via autodiff gradients, vs Ingber's single-core ASA "6-day runs". The
   90s-constraint→modern-tooling thesis (`CRITIQUE_AND_FUTURE.md`) is demonstrated end-to-end.
2. **The nonlinearity adds nothing here.** NLL improves by 56 out of 59M (negligible); the
   fitted gains `β≈1.9` are nonzero but inconsequential; held-out group separation is
   **identical** (−7.40); and **AIC prefers the linear model** (parsimony). Another clean
   negative — the robust signal is the linear/classical structure.
3. **Side-finding: per-channel beats cross-channel for discrimination.** The per-channel
   (diagonal) linear drift gives a *stronger* held-out separation (t=−7.40) than the
   full cross-channel `A` model from `smni.py` (t=−4.04). Cross-channel coupling in the
   drift diluted the per-channel CMI group signal — worth following up.

## Takeaway
Across Steps A and 3, the story is consistent and falsifiable: the **classical, linear,
per-channel CMI signal is what replicates held-out**; neither quantum coherence nor drift
nonlinearity adds generalizing power on this dataset. The contribution is the modern,
tested, GPU/autodiff reimplementation that makes these claims *checkable* — not new physics.
