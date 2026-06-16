"""Validate the SMNI/CMI integration: reproduce the smni-eeg group separation
through the QCCCM `qcccm.models.smni` path.

Reuses the cached SMNI_CMI tensors and loader from the smni-eeg satellite repo,
computes CMI with the JAX module here, and checks that the alcoholic-vs-control
canonical-momentum effect matches the reference result (TRAIN t≈-4.3, held-out
TEST t≈-4.0, FULL t≈-11).

Run:  python scripts/reproduce_smni_cmi.py
"""
from __future__ import annotations

import os

import jax
jax.config.update("jax_enable_x64", True)   # parity with the numpy reference
import jax.numpy as jnp
import numpy as np

from qcccm.models import smni
from qcccm.neuroai.smni_eeg import build_set

# Data + caches are NOT in git (large). Default to the local satellite checkout;
# override with SMNI_EEG / SMNI_EEG_DATA / SMNI_EEG_OUT on another system.
_BASE = os.environ.get("SMNI_EEG", os.path.expanduser("~/Workspace/smni-eeg"))
DATA = os.environ.get("SMNI_EEG_DATA", os.path.join(_BASE, "data"))
OUT = os.environ.get("SMNI_EEG_OUT", os.path.join(_BASE, "out"))


def welch_t(a: np.ndarray, c: np.ndarray) -> float:
    return float((a.mean() - c.mean()) /
                 np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c)))


def load(set_name: str):
    s = build_set(os.path.join(DATA, set_name),
                  cache=os.path.join(OUT, f"cache_{set_name}.npz"))
    M = jnp.asarray(np.asarray(s.M, dtype=np.float64))
    return M, np.asarray(s.groups)


def group_sep(cmi: jnp.ndarray, groups: np.ndarray, label: str) -> None:
    mag = np.asarray(smni.momentum_magnitude(cmi)).mean(axis=1)   # (N,) per trial
    a, c = mag[groups == "alcoholic"], mag[groups == "control"]
    print(f"  [{label:18s}] alcoholic n={len(a):5d} {a.mean():.4g} | "
          f"control n={len(c):5d} {c.mean():.4g} | Welch t = {welch_t(a, c):+.3f}")


def main() -> None:
    print("== fit-on-TRAIN, self + held-out TEST ==")
    Mtr, gtr = load("TRAIN")
    drift = smni.fit_linear_drift(Mtr)
    group_sep(smni.canonical_momenta(Mtr, drift), gtr, "TRAIN (fit)")
    Mte, gte = load("TEST")
    group_sep(smni.canonical_momenta(Mte, drift), gte, "TEST (held-out)")

    print("== fit-on-FULL ==")
    Mf, gf = load("FULL")
    group_sep(smni.canonical_momenta(Mf, smni.fit_linear_drift(Mf)), gf, "FULL")


if __name__ == "__main__":
    main()
