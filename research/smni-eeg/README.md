# SMNI-EEG

Working project around the **UCI / Ingber EEG alcoholism dataset** (`SMNI_CMI`) — its
history, its physics/modeling lineage (Lester Ingber's *Statistical Mechanics of
Neocortical Interactions*), and pipelines to process it for modeling and reproduction work.

## Central idea: a shared path-integral / least-action spine
The organizing thread is that **both Ingber's SMNI and Friston's free energy principle are
least-action formulations expressed as path integrals over densities** — same machinery,
different semantics. We use the data-anchored Ingber program as a concrete way into Friston,
Da Costa, Sajid, Heins, Ueltzhöffer, Pavliotis & Parr (2023), *"Path integrals, particular
kinds, and strange things"* (*Phys. Life Rev.*, arXiv:2210.12761). See
[`docs/FRISTON_BRIDGE.md`](docs/FRISTON_BRIDGE.md) and [`docs/PATH_INTEGRAL.md`](docs/PATH_INTEGRAL.md).

## Threads converging on the same dataset
1. **Path-integral correspondence (the spine)** — map Ingber's SMNI **action/Lagrangian** and
   **canonical momenta (CMI)** onto Friston's **free-energy action** and conjugate momenta;
   locate the cortical macrocolumn as a candidate **strange particle** (one that "infers its own
   actions"). [`docs/PATH_INTEGRAL.md`](docs/PATH_INTEGRAL.md).
2. **Estimating molecular Ca²⁺ activity from EEG** — Ingber's `Π = p + qA` coupling, propagated
   with the **qPATHINT** quantum path-integral algorithm; a concrete cross-scale (blanket)
   flow. Developed under **XSEDE** allocations. [`docs/HISTORY.md`](docs/HISTORY.md) §0–1.
3. **SMNI / CMI foundation** — the path-integral statistical mechanics of EEG; this exact
   dataset was distributed with Ingber's 1997 CMI paper.
4. **Trilinear / tensor modeling** — reproduction of **Wang, Begleiter & Porjesz (2000)** TCM,
   already re-run on the **DGX Spark**.

## Layout
```
smni-eeg/
├── README.md            you are here
├── docs/
│   ├── HISTORY.md        project + author history, dataset provenance, bibliography
│   └── DATASET.md        data catalog, file-format spec, processing plan
├── data/                 symlinks into ~/Data (no duplication)
│   ├── TRAIN  → SMNI_CMI_TRAIN   (20 subj: 10 alcoholic / 10 control)
│   ├── TEST   → SMNI_CMI_TEST    (20 subj: 10 / 10)
│   ├── FULL   → SMNI_CMI_FULL    (122 subj: 77 / 45)
│   └── prior_work → alcoholism_data (existing notebooks + EDF export)
├── src/                  loaders / pipelines (to build)
└── notebooks/            exploratory analysis
```

## Status
- [x] Datasets located, cataloged, symlinked
- [x] History + provenance written (`docs/`)
- [x] Path-integral bridge to Friston 2023 (`docs/FRISTON_BRIDGE.md`, `docs/PATH_INTEGRAL.md`)
- [x] `.rd` → tensor loader in `src/` (handles gzipped TEST/FULL)
- [x] **CMI estimation** (`src/cmi.py`, `src/run_cmi.py`) — validated: group
      separation generalizes TRAIN→TEST (Welch t ≈ −4.3 / −4.0)
- [ ] Map CMI ↔ FEP conjugate momenta (program step 2–3)
- [ ] Nonlinear SMNI Lagrangian + ASA drift (replace linear surrogate)
- [ ] Trilinear / PARAFAC reproduction (Wang 2000) on DGX Spark
- [ ] qPATHINT calcium track (speculative, later)

### Doc index
- `docs/HISTORY.md` · `docs/DATASET.md` · `docs/PATH_INTEGRAL.md` · `docs/FRISTON_BRIDGE.md`
- `docs/CODE_AND_REFERENCES.md` — Ingber repos, ASA↔Boyd/optimization, Active Inference
- `docs/CRITIQUE_AND_FUTURE.md` — Ingber's future needs, external critiques, 90s→modern map, MPPI teaching ladder
- `docs/QPATHINT_REIMAGINED.md` — modern reimplementation tiers (JAX → Dynamiqs/QuTiP → CUDA-Q/PennyLane→QCCD)
- `docs/QCCCM_INTEGRATION.md` — **host framework**: map SMNI/CMI/qPATHINT onto Morgan's QCCCM repo (`refs/qcccm`)

## Compute
Heavy processing targets the **DGX Spark** on the LAN (the trilinear reproduction already
ran there). Add it as a remote location in NVIDIA AI Workbench for managed runs.
