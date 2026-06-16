"""Run the quantum-coherence sweep on the real SMNI_CMI EEG.

Asks the honest question: does including off-diagonal (cross-channel) coherence in
the density-matrix representation improve alcoholic-vs-control separation? At q=0
only per-channel variances enter (classical); q sweeps in coherence. Reports the
Welch t of the von-Neumann-entropy feature at each q. See
research/smni-eeg/docs/CMI_EFE.md.

Run:  python scripts/quantum_coherence_sweep.py
"""
from __future__ import annotations

import os

import jax.numpy as jnp
import numpy as np

from qcccm.neuroai.smni_eeg import build_set
from qcccm.neuroai import coherence as co

_BASE = os.environ.get("SMNI_EEG", os.path.expanduser("~/Workspace/smni-eeg"))
DATA = os.environ.get("SMNI_EEG_DATA", os.path.join(_BASE, "data"))
OUT = os.environ.get("SMNI_EEG_OUT", os.path.join(_BASE, "out"))

QS = (0.0, 0.25, 0.5, 0.75, 1.0)


def run(set_name: str) -> dict[float, float]:
    s = build_set(os.path.join(DATA, set_name),
                  cache=os.path.join(OUT, f"cache_{set_name}.npz"))
    M = jnp.asarray(np.asarray(s.M, dtype=np.float32))
    sweep = co.coherence_sweep(M, s.groups, qs=QS)
    cells = "  ".join(f"q={q}:{t:+.2f}" for q, t in sweep.items())
    print(f"[{set_name:5s}] {cells}")
    return sweep


def main() -> None:
    print("Welch t (alcoholic vs control) of entropy feature vs quantumness q:")
    sweeps = {name: run(name) for name in ("TRAIN", "TEST", "FULL")}

    # Held-out TEST is the arbiter (it validated the classical CMI result).
    # Coherence "adds something" only if the q=1 effect REPLICATES held-out.
    te = sweeps["TEST"]
    classical_robust = abs(te[0.0]) > 3.0
    coherence_robust = abs(te[1.0]) > 3.0
    print("\nVerdict (held-out TEST as arbiter):")
    print(f"  classical (q=0)  |t|={abs(te[0.0]):.2f}  -> "
          f"{'replicates' if classical_robust else 'absent'}")
    print(f"  coherence (q=1)  |t|={abs(te[1.0]):.2f}  -> "
          f"{'replicates' if coherence_robust else 'does NOT replicate'}")
    if classical_robust and not coherence_robust:
        print("  => Off-diagonal coherence adds no GENERALIZING discriminability "
              "for this dataset.\n     (FULL's large in-sample q=1 effect is "
              "sign-flipped and does not hold out.)")


if __name__ == "__main__":
    main()
