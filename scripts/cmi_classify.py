"""Step C: per-condition CMI + subject-level cross-validated classifier.

Turns the CMI group effect into a proper classification result with NO subject
leakage: subjects (not trials) are split into folds. Per fold we fit the SMNI drift
on training subjects, take per-channel mean|CMI| as the trial feature, fit logistic
regression, and predict held-out subjects. Reports pooled held-out AUC, per-condition
AUC, and a raw-EEG-variance baseline. Dependency-free (no sklearn).

Run:  python scripts/cmi_classify.py
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


def auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s)
    ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_lr(X, y, l2=1e-2, lr=0.5, iters=800):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / n + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def cmi_features(M, drift):
    """Per-trial per-channel mean |CMI| -> (n_trials, n_chan)."""
    cmi = np.asarray(smni.canonical_momenta(M, drift))
    return np.abs(cmi).mean(axis=2)


def var_features(M):
    """Baseline: per-trial per-channel signal variance."""
    return np.asarray(jnp.var(M, axis=2))


def subject_cv(M, groups, subjects, conditions, k=5, seed=0):
    y = (groups == "alcoholic").astype(float)
    subs = np.unique(subjects)
    rng = np.random.RandomState(seed); rng.shuffle(subs)
    folds = np.array_split(subs, k)

    cmi_scores = np.full(len(y), np.nan)
    var_scores = np.full(len(y), np.nan)
    for fold in folds:
        te = np.isin(subjects, fold); tr = ~te
        drift = smni.fit_linear_drift(jnp.asarray(M[tr]))
        # CMI classifier
        Xc = cmi_features(jnp.asarray(M), drift)
        mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-8
        Xc = (Xc - mu) / sd
        w, b = fit_lr(Xc[tr], y[tr])
        cmi_scores[te] = Xc[te] @ w + b
        # variance baseline
        Xv = var_features(jnp.asarray(M))
        muv, sdv = Xv[tr].mean(0), Xv[tr].std(0) + 1e-8
        Xv = (Xv - muv) / sdv
        wv, bv = fit_lr(Xv[tr], y[tr])
        var_scores[te] = Xv[te] @ wv + bv

    print(f"subject-level {k}-fold CV (n={len(y)} trials, "
          f"{len(subs)} subjects):")
    print(f"  CMI classifier   held-out AUC = {auc(y, cmi_scores):.3f}")
    print(f"  variance baseline held-out AUC = {auc(y, var_scores):.3f}")
    print("  per-condition (CMI classifier) held-out AUC:")
    for cond in ("S1", "S2match", "S2nomatch"):
        m = conditions == cond
        print(f"    {cond:10s} n={int(m.sum()):5d}  AUC = {auc(y[m], cmi_scores[m]):.3f}")


def main():
    s = build_set(os.path.join(DATA, "FULL"),
                  cache=os.path.join(OUT, "cache_FULL.npz"))
    M = np.asarray(s.M, np.float64)
    subject_cv(M, np.asarray(s.groups), np.asarray(s.subjects),
               np.asarray(s.conditions))


if __name__ == "__main__":
    main()
