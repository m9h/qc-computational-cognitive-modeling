"""Nonlinear vs linear SMNI drift (Step A): does the tanh nonlinearity help?

Fits per-channel linear (closed form) and nonlinear (g=a·M+b+β·tanh(γ·M), via
fit_mle = autodiff MLE, the ASA replacement) drifts on TRAIN, applies them
held-out to TEST, and compares (a) alcoholic-vs-control CMI separation and
(b) model fit (total NLL / AIC). See research/smni-eeg/docs.

Run:  python scripts/nonlinear_smni_compare.py
"""
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from qcccm.neuroai.smni_eeg import build_set
from qcccm.models import smni, smni_nonlinear as nl

_BASE = os.environ.get("SMNI_EEG", os.path.expanduser("~/Workspace/smni-eeg"))
DATA = os.environ.get("SMNI_EEG_DATA", os.path.join(_BASE, "data"))
OUT = os.environ.get("SMNI_EEG_OUT", os.path.join(_BASE, "out"))


def _load(name):
    s = build_set(os.path.join(DATA, name),
                  cache=os.path.join(OUT, f"cache_{name}.npz"))
    return jnp.asarray(np.asarray(s.M, np.float64)), np.asarray(s.groups)


def _welch_t(mag, groups):
    a, c = mag[groups == "alcoholic"], mag[groups == "control"]
    return float((a.mean() - c.mean()) /
                 np.sqrt(a.var(ddof=1)/len(a) + c.var(ddof=1)/len(c)))


def _sep(M, p, groups):
    mag = np.asarray(nl.canonical_momenta(M, p))
    mag = np.sqrt((mag ** 2).sum(1)).mean(1)
    return _welch_t(mag, groups)


def _total_nll(M, p):
    V = smni.velocity(M)
    r = np.asarray(V - nl.drift(M, p))
    # profile NLL per channel summed
    nll = 0.0
    for c in range(r.shape[1]):
        rc = r[:, c, :]
        nll += 0.5 * rc.size * np.log(rc.var() + 1e-12)
    return nll


def main():
    Mtr, gtr = _load("TRAIN")
    Mte, gte = _load("TEST")
    n_obs = int(Mtr.shape[0] * Mtr.shape[2])

    print("fitting per-channel LINEAR baseline (closed form)...")
    lin = nl.fit_linear_drift_perchannel(Mtr)
    print("fitting per-channel NONLINEAR drift via fit_mle (64 channels)...")
    non = nl.fit_nonlinear_drift(Mtr)

    C = Mtr.shape[1]
    for name, p, k_per in [("linear", lin, 2), ("nonlinear", non, 4)]:
        k = k_per * C
        nll = _total_nll(Mtr, p)
        aic = 2 * k + 2 * nll      # NLL already; AIC = 2k - 2logL = 2k + 2*NLL
        print(f"\n[{name:9s}] params={k:4d}  trainNLL={nll:,.0f}  AIC={aic:,.0f}")
        print(f"           group t: TRAIN={_sep(Mtr, p, gtr):+.3f}  "
              f"TEST(held-out)={_sep(Mte, p, gte):+.3f}")
    print(f"\nmean |β| (nonlinear gain) = {float(jnp.mean(jnp.abs(non.beta))):.4f}")


if __name__ == "__main__":
    main()
