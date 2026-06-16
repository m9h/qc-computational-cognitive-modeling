# Re-evaluating Ingber's CMI on the UCI EEG alcoholism dataset: a cautionary, subject-level analysis

*Working write-up. Code: `qcccm` branch `smni-cmi`; data: UCI / Ingber SMNI_CMI.*

## Abstract
We re-express Ingber's SMNI / Canonical Momenta Indicators (CMI) in a modern, differentiable,
tested form (JAX) and evaluate, as a **cross-subject biomarker**, whether mean|CMI| separates
alcoholic from control subjects. Trial-level statistics look strong (Welch t up to −11) — **but
they are pseudoreplication**: the dataset's TRAIN and TEST sets share all 20 subjects, and the
full set has 11,057 trials from only 122 subjects. Under proper **subject-level cross-validation**,
the CMI classifier is **at chance (AUC ≈ 0.45)**, while a trivial raw-variance baseline reaches
**AUC ≈ 0.63**. The CMI transform `Σ⁻¹(Ṁ−g)` whitens away the per-channel amplitude/variance scale
— which is where the across-subject group signal actually resides — so it discards the
discriminative information. Quantum-coherence and nonlinear-drift extensions add nothing even by
the weaker (subject-overlapping) tests. The contribution is a tested reimplementation that
**exposes the apparent CMI effect as an evaluation artifact** — a result only visible with proper
subject-level discipline.

## 1. Background
SMNI (Ingber 1981–2021) is a path-integral statistical mechanics of EEG; CMI are its canonical
momenta `Π = ∂L/∂q̇ = Σ⁻¹(Ṁ − g)` (proven here, by autodiff, to coincide with the Gaussian-limit
FEP conjugate momentum — `CMI_EFE.md`). Note Ingber fit CMI **per subject** to describe dynamics;
the *cross-subject classification* tested here is our question, not his claim.

## 2. Methods
Loader (`qcccm.neuroai.smni_eeg`), differentiable CMI (`qcccm.models.smni`), nonlinear drift via
`fit_mle` (`qcccm.models.smni_nonlinear`), coherence sweep (`qcccm.neuroai.coherence`), PATHINT
(`qcccm.neuroai.pathint`). Classifier: per-channel mean|CMI| → logistic regression, **subject-level
5-fold CV** (`scripts/cmi_classify.py`), AUC dependency-free.

## 3. Results
### 3.1 The trial-level effect is pseudoreplication
| set | trials (alc/ctrl) | subjects | Welch t on mean|CMI| |
|---|---|---|---|
| TRAIN | 600 (300/300) | 20 | −4.32 |
| TEST | 600 (300/300) | **same 20** | −4.04 |
| FULL | 11,057 | 122 | −11.04 |

TRAIN and TEST share **all 20 subjects** (verified), so the "held-out" t was a within-subject trial
split. The full-set t treats 11,057 correlated trials as independent — the n is subjects (122), not
trials. These t-values are **not** evidence of a cross-subject effect.

### 3.2 Subject-level CV: CMI is at chance; a variance baseline is not
Subject-level 5-fold CV on FULL (122 subjects, 11,057 trials), held-out AUC:

| feature | held-out AUC |
|---|---|
| **mean\|CMI\| (logistic regression)** | **0.454** (≈ chance) |
| raw per-channel variance (baseline) | **0.627** |
| per-condition CMI — S1 / S2match / S2nomatch | 0.444 / 0.464 / 0.464 |

CMI does not discriminate held-out subjects; a trivial variance feature does. Mechanism: CMI is the
precision-whitened innovation `(Ṁ−g)/σ²`, which removes the amplitude/variance scale that carries
the across-subject group difference.

### 3.3 Quantum coherence and nonlinear drift add nothing
Coherence sweep: pure-coherence (q=1) entropy feature shows no separation even on the
subject-overlapping TEST (|t|=0.2) (`CMI_EFE.md`). Nonlinear `tanh` drift: negligible NLL gain,
higher AIC than linear (`NONLINEAR_FIT.md`). Both negative.

## 4. Path integrals as a teaching/computation through-line
The path-integral machinery is sound and reusable independently of the biomarker question: a
runnable ladder — MPPI (control) → PATHINT (density) → SMNI/CMI → qPATHINT — is provided
(`notebooks/07_path_integrals.ipynb`), validated against analytic diffusion / Ornstein–Uhlenbeck.

## 5. Discussion
The headline lesson is methodological: an apparently strong CMI group effect (t up to −11)
**evaporates under subject-level evaluation**, because the dataset's splits share subjects and
trials are pseudoreplicated. As a cross-subject biomarker on this data, mean|CMI| is at chance and
underperforms raw variance. This does **not** refute SMNI as a per-subject descriptive model (its
intended use), but it does caution against trial-level CMI classification claims. Combined with the
null coherence and nonlinear results, the consistent message is that the modern, tested
reimplementation's chief value here is **making these claims checkable** — and the rigorous checks
are negative.

## 6. Reproducibility
`uv` + jax 0.10; 26 tests; subject-level CV with documented subject overlap. Branch `smni-cmi`.
