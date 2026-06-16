"""Step 1 of the program (see ../docs/PATH_INTEGRAL.md): estimate CMI on SMNI_CMI.

Fits the linear-drift SMNI surrogate on TRAIN, computes canonical momenta (CMI),
saves the model + CMI arrays, and runs a quick sanity check: does the per-trial CMI
momentum magnitude differ between alcoholic and control groups?

Usage:
    python3 src/run_cmi.py [--set TRAIN] [--condition S1] [--out out/cmi_train.npz]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from load_rd import build_set
from cmi import CMIModel, cmi_summary

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="TRAIN", help="TRAIN | TEST | FULL")
    ap.add_argument("--condition", default=None,
                    help="S1 | S2match | S2nomatch (default: all)")
    ap.add_argument("--limit-subjects", type=int, default=None)
    ap.add_argument("--full-cov", action="store_true",
                    help="use full diffusion covariance instead of diagonal")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "cmi.npz"))
    args = ap.parse_args()

    set_dir = os.path.join(DATA, args.set)
    cache = os.path.join(ROOT, "out",
                         f"cache_{args.set}"
                         f"{'_'+args.condition if args.condition else ''}.npz")
    print(f"[load] {set_dir} condition={args.condition}")
    d = build_set(set_dir, cache=cache, condition=args.condition)
    M, channels, groups = d["M"], d["channels"], d["groups"]
    print(f"[load] trials={M.shape[0]} chans={M.shape[1]} samples={M.shape[2]} "
          f"| alcoholic={int((groups=='alcoholic').sum())} "
          f"control={int((groups=='control').sum())}")

    print("[fit] linear-drift SMNI surrogate "
          f"({'full' if args.full_cov else 'diagonal'} diffusion)")
    model = CMIModel(diagonal=not args.full_cov)
    cmi = model.fit_transform(M, channels=channels)

    summ = cmi_summary(cmi)
    print(f"[cmi] {summ}")

    # sanity check: mean momentum magnitude per group
    mag = np.sqrt((cmi ** 2).sum(axis=1)).mean(axis=1)   # (N,) per-trial mean |Π|
    for g in ("alcoholic", "control"):
        sel = groups == g
        if sel.any():
            print(f"  mean |CMI| [{g:9s}] n={sel.sum():3d}  "
                  f"{mag[sel].mean():.4g} ± {mag[sel].std():.4g}")
    if (groups == "alcoholic").any() and (groups == "control").any():
        a, c = mag[groups == "alcoholic"], mag[groups == "control"]
        # Welch t statistic (numpy-only)
        t = (a.mean() - c.mean()) / np.sqrt(a.var(ddof=1)/len(a) + c.var(ddof=1)/len(c))
        print(f"  group separation (Welch t on mean |CMI|): t = {t:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, cmi=cmi, groups=groups,
             channels=np.array(channels, dtype=object),
             conditions=np.asarray(d["conditions"], dtype=object),
             subjects=np.asarray(d["subjects"], dtype=object))
    model.save(args.out.replace(".npz", "_model.npz"))
    print(f"[save] CMI -> {args.out}")
    print(f"[save] model -> {args.out.replace('.npz', '_model.npz')}")


if __name__ == "__main__":
    main()
