# Dataset catalog & processing plan — SMNI_CMI EEG

## Inventory (in `~/Data`, symlinked under `data/`)

| Set | Path | Subjects | Alcoholic (`co*a*`) | Control (`co*c*`) | On-disk |
|---|---|---|---|---|---|
| TRAIN | `data/TRAIN` → `SMNI_CMI_TRAIN` | 20 | 10 | 10 | 158 MB |
| TEST  | `data/TEST`  → `SMNI_CMI_TEST`  | 20 | 10 | 10 | 39 MB |
| FULL  | `data/FULL`  → `SMNI_CMI_FULL`  | 122 | 77 | 45 | 1.4 GB |
| prior_work | `data/prior_work` → `alcoholism_data` | — | — | — | notebooks + EDF |

Redundant tarballs in `~/Data` (already extracted into TRAIN/TEST): `SMNI_CMI_TRAIN.tar.gz`
(37M), `SMNI_CMI_TEST.tar.gz` (38M) — safe to delete once extraction is verified.

## Acquisition parameters
- **64 channels**, 10-20 + extended montage, scalp EEG.
- **256 Hz** sampling → **3.906 ms / sample** (`1/256 s`), amplitude in **µV**.
- Trial length: small set ≈ **256 samples (1 s)**; FULL trials longer (header reports
  **416 samples**, of which **368 post-stimulus**).
- Conditions per trial: **`S1 obj`**, **`S2 match`**, **`S2 nomatch`**.

## File format (`*.rd.NNN`, ASCII)
One file per trial inside each subject directory (e.g. `co2a0000364/co2a0000364.rd.101`):

```
# co2a0000364.rd                                  <- subject / run id
# 120 trials, 64 chans, 416 samples 368 post_stim samples
# 3.906000 msecs uV                               <- sample interval, units
# S2 match , trial 101                            <- condition, trial index
# FP1 chan 0                                       <- (repeated per channel)
101 FP1 0 -0.844                                   <- trial chan sample value(µV)
101 FP1 1 -0.356
...
```
Data rows are 4 whitespace-separated columns: `trial_idx  channel_name  sample_idx  value_µV`.
Naming: `co` + cohort digit + `a|c` (alcoholic|control) + subject id; trial suffix `.rd.NNN`.

⚠️ Some subject dirs contain stray non-trial files (e.g. `.DS_Store`, a notebook) — the loader
must skip files whose lines don't match the 4-column schema.

## Existing prior work (`data/prior_work`)
- `alcholism_eeg.ipynb`, `process_single_file_to_df.ipynb` — parse `.rd` → pandas.
- `edf_file.edf` — an already-converted EDF (path into MNE / standard EEG tooling).

## Processing plan
1. **Loader** (`src/`): `.rd` ASCII → xarray/numpy tensor
   `(subject × condition × trial × channel[64] × time)`; emit per-subject EDF for MNE.
2. **Preprocess**: bandpass/notch, baseline (pre-stim window), artifact reject, epoch by
   condition (S1 / S2-match / S2-nomatch); group label alcoholic/control.
3. **Trilinear / PARAFAC reproduction** (Wang 2000 TCM): tensor decomposition
   (space × time × subjects) via `tensorly`; **run on DGX Spark** (GPU). Compare components &
   alcoholic-vs-control discrimination to the published maps/waveforms.
4. **SMNI / CMI track**: compute canonical momenta indicators per Ingber (1997); ASA fits of
   the SMNI Lagrangian; optionally explore the `p + qA` / qPATHINT calcium coupling as a
   downstream multi-scale extension.

## Compute
DGX Spark on the LAN is the GPU target (trilinear reproduction already ran there). Register it
as a remote location in NVIDIA AI Workbench for managed/reproducible runs.
