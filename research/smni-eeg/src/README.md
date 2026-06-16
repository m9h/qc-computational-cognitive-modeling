# src/ — CMI estimation pipeline

Step 1 of the path-integral program (see `../docs/PATH_INTEGRAL.md`).

## Modules
- **`load_rd.py`** — parser for the `.rd` ASCII trial files. Handles uncompressed
  (TRAIN) and gzipped (TEST/FULL) trials transparently, skips header/stray lines,
  parses subject/group/condition. `load_dataset()` + `stack_trials()` → `(N, C, T)`.
- **`cmi.py`** — `CMIModel`: canonical-momenta estimator. Fits a linear (OU) drift
  `g(M)=A·M+b` and diffusion `Σ`, then `Π = Σ⁻¹(Ṁ − g(M))` are the CMI. Fit on one
  set, `transform` another (held-out). `save`/`load` for the fitted model.
- **`run_cmi.py`** — CLI: fit on a set, compute CMI, save arrays + model, and run a
  group-separation sanity check (alcoholic vs control).

## Usage
```bash
python3 src/run_cmi.py --set TRAIN                 # fit + CMI on TRAIN, save to out/
python3 src/run_cmi.py --set TRAIN --condition S2match
python3 src/run_cmi.py --set FULL --full-cov       # full diffusion covariance
```
Outputs: `out/cmi.npz` (cmi, groups, channels, conditions, subjects) and
`out/cmi_model.npz` (fitted A, b, Σ).

## Validated result (2026-06-15)
- TRAIN: 600 trials (300/300). CMI |Π| separates groups, Welch **t = −4.32**.
- Held-out TEST (TRAIN-fit model applied): Welch **t = −4.04**, same direction
  (control > alcoholic in momentum magnitude). The signal generalizes out-of-sample.

## Notes / next
- The linear drift is a **tractable surrogate** for the full nonlinear SMNI
  Lagrangian. The `CMIModel` interface is kept so an ASA-fit nonlinear drift can
  replace `fit()`/`drift()` without changing the CMI math.
- Next: per-condition CMI; map CMI ↔ FEP conjugate momenta (program step 2–3);
  scale to FULL on the DGX Spark.
