"""Coordinate-free flow invariants vs chart-dependent features (subject-level CV).

Fits a drift PER SUBJECT and extracts reparameterization-invariant features of the
flow — eigenvalues of the drift Jacobian A (invariant under channel remixing
A→PAP⁻¹: divergence, spectral abscissa, oscillation rate, …). Compares, with
leave-one-subject-out CV, against the chart-dependent baselines (per-channel
mean|CMI|, per-channel variance). Tests whether coordinate-free invariants recover
group signal that mean|CMI| missed. Dependency-free.

Run:  python scripts/invariant_features_cv.py
"""
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from qcccm.neuroai.smni_eeg import build_set
from qcccm.models import smni

_BASE = os.environ.get("SMNI_EEG", os.path.expanduser("~/Workspace/smni-eeg"))
DATA = os.environ.get("SMNI_EEG_DATA", os.path.join(_BASE, "data"))
OUT = os.environ.get("SMNI_EEG_OUT", os.path.join(_BASE, "out"))


def auc(y, s):
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def fit_lr(X, y, l2=1.0, lr=0.3, iters=1500):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w / len(y))
        b -= lr * float(np.mean(p - y))
    return w, b


def subject_features(M_s):
    d = smni.fit_linear_drift(jnp.asarray(M_s))
    A = np.asarray(d.A)
    lam = np.linalg.eigvals(A)
    re, im = lam.real, np.abs(lam.imag)
    inv = np.array([np.trace(A).real,          # divergence Σ Re λ
                    re.max(),                  # spectral abscissa
                    re.mean(),                 # mean decay rate
                    float(np.abs(lam).max()),  # spectral radius
                    im.mean(),                 # typical oscillation
                    float((re > 0).mean())])   # fraction unstable modes
    cmi = np.asarray(smni.canonical_momenta(jnp.asarray(M_s), d))
    return inv, np.abs(cmi).mean(axis=(0, 2)), M_s.var(axis=2).mean(axis=0)


def loo_auc(X, y):
    X = np.asarray(X, float)
    scores = np.empty(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xs = (X - mu) / sd
        w, b = fit_lr(Xs[tr], y[tr])
        scores[i] = Xs[i] @ w + b
    return auc(y, scores)


def main():
    s = build_set(os.path.join(DATA, "FULL"), cache=os.path.join(OUT, "cache_FULL.npz"))
    M = np.asarray(s.M, np.float64)
    subj = np.asarray(s.subjects); grp = np.asarray(s.groups)
    subjects = np.unique(subj)
    inv_X, cmi_X, var_X, y = [], [], [], []
    for su in subjects:
        m = subj == su
        inv, cmi, var = subject_features(M[m])
        inv_X.append(inv); cmi_X.append(cmi); var_X.append(var)
        y.append(1.0 if grp[m][0] == "alcoholic" else 0.0)
    y = np.array(y)
    print(f"subjects: {len(y)} ({int(y.sum())} alcoholic / {int(len(y)-y.sum())} control)")
    print("leave-one-subject-out AUC:")
    print(f"  coordinate-free flow invariants (6-d) : {loo_auc(inv_X, y):.3f}")
    print(f"  mean|CMI| per channel (64-d, chart)   : {loo_auc(cmi_X, y):.3f}")
    print(f"  variance per channel  (64-d, chart)   : {loo_auc(var_X, y):.3f}")


if __name__ == "__main__":
    main()
