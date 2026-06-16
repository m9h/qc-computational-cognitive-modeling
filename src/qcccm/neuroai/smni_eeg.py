"""SMNI_CMI EEG dataset loader — the UCI / Ingber EEG alcoholism database.

First-class QCCCM access to the dataset Ingber distributed with his 1997
Canonical Momenta Indicators paper. Parses the ``.rd`` ASCII trial files
(uncompressed TRAIN, gzipped TEST/FULL), yielding ``(trials, channels, time)``
tensors that feed :mod:`qcccm.models.smni` (CMI) and, via
:func:`set_to_density_matrices`, the quantum bridge in :mod:`qcccm.models.bridge`.

File format (one file per trial inside each subject directory)::

    # co2a0000364.rd
    # 120 trials, 64 chans, 416 samples 368 post_stim samples
    # 3.906000 msecs uV
    # S2 match , trial 101
    # FP1 chan 0
    101 FP1 0 -0.844            <- trial chan sample value(uV)
    ...

Subjects are coded ``co<digit>a*`` (alcoholic) / ``co<digit>c*`` (control);
conditions are ``S1 obj``, ``S2 match``, ``S2 nomatch``. 64 channels @ 256 Hz.
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

FS_HZ = 256.0  # sampling rate (interval 3.906 ms = 1/256 s)

_SUBJ_RE = re.compile(r"co\d([ac])", re.IGNORECASE)
_COND_MAP = {"S1 obj": "S1", "S2 match": "S2match", "S2 nomatch": "S2nomatch"}


def _open_text(path: str):
    """Open a trial file, transparently decompressing ``.gz`` (TEST/FULL)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def _parse_group(subject: str) -> str:
    m = _SUBJ_RE.match(subject)
    if not m:
        return "unknown"
    return "alcoholic" if m.group(1).lower() == "a" else "control"


@dataclass
class Trial:
    subject: str
    group: str               # 'alcoholic' | 'control' | 'unknown'
    condition: str           # 'S1' | 'S2match' | 'S2nomatch' | 'unknown'
    trial_idx: int
    channels: list[str]
    data: np.ndarray         # (n_chan, n_samples), microvolts
    path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_chan(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


class SMNICMISet(NamedTuple):
    """A stacked SMNI_CMI subset with aligned per-trial metadata."""
    M: np.ndarray            # (n_trials, n_chan, n_samples)
    channels: list[str]
    groups: np.ndarray       # (n_trials,) 'alcoholic'/'control'
    conditions: np.ndarray   # (n_trials,) 'S1'/'S2match'/'S2nomatch'
    subjects: np.ndarray     # (n_trials,)
    trial_idx: np.ndarray    # (n_trials,)


def load_trial(path: str) -> Trial | None:
    """Parse one ``.rd`` trial file; returns None if it isn't a valid trial."""
    subject = condition = None
    trial_idx = -1
    chan_order: list[str] = []
    chan_to_samples: dict[str, dict[int, float]] = {}

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if subject is None and line.endswith(".rd"):
                    subject = line.lstrip("#").strip()[:-3].strip()
                elif " trial " in line and ("S1" in line or "S2" in line):
                    body = line.lstrip("#").strip()
                    condition = _COND_MAP.get(body.split(",")[0].strip(), "unknown")
                    try:
                        trial_idx = int(line.rsplit("trial", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                continue
            parts = line.split()
            if len(parts) != 4 or not parts[0].lstrip("-").isdigit():
                continue
            _, chan, tok_sample, tok_val = parts
            try:
                sample, val = int(tok_sample), float(tok_val)
            except ValueError:
                continue
            if chan not in chan_to_samples:
                chan_to_samples[chan] = {}
                chan_order.append(chan)
            chan_to_samples[chan][sample] = val

    if not chan_order:
        return None
    n_samples = max(len(v) for v in chan_to_samples.values())
    data = np.full((len(chan_order), n_samples), np.nan, dtype=np.float64)
    for ci, chan in enumerate(chan_order):
        for s, v in chan_to_samples[chan].items():
            if 0 <= s < n_samples:
                data[ci, s] = v
    subject = subject or os.path.basename(path).split(".")[0]
    return Trial(subject=subject, group=_parse_group(subject),
                 condition=condition or "unknown", trial_idx=trial_idx,
                 channels=chan_order, data=data, path=path,
                 meta={"fs_hz": FS_HZ})


def load_subject(subject_dir: str) -> list[Trial]:
    out = []
    for name in sorted(os.listdir(subject_dir)):
        p = os.path.join(subject_dir, name)
        if os.path.isfile(p) and ".rd" in name:
            t = load_trial(p)
            if t is not None:
                out.append(t)
    return out


def load_dataset(set_dir: str, condition: str | None = None,
                 group: str | None = None, limit_subjects: int | None = None
                 ) -> list[Trial]:
    """Load all trials under a set dir (TRAIN/TEST/FULL), with optional filters."""
    subdirs = sorted(d for d in os.listdir(set_dir)
                     if os.path.isdir(os.path.join(set_dir, d))
                     and d.lower().startswith("co"))
    if limit_subjects:
        subdirs = subdirs[:limit_subjects]
    out: list[Trial] = []
    for d in subdirs:
        for t in load_subject(os.path.join(set_dir, d)):
            if condition and t.condition != condition:
                continue
            if group and t.group != group:
                continue
            out.append(t)
    return out


def stack_trials(trials: list[Trial], channels: list[str] | None = None,
                 n_samples: int | None = None
                 ) -> tuple[np.ndarray, list[str], list[int]]:
    """Stack trials to (n_trials, n_chan, n_samples) on a common channel set and
    the *modal* sample length (so a stray short file can't shrink everything).
    Returns (array, channels, kept_indices)."""
    if not trials:
        return np.empty((0, 0, 0)), channels or [], []
    if channels is None:
        channels = trials[0].channels
    chan_set = set(channels)
    eligible = [(i, t) for i, t in enumerate(trials)
                if chan_set.issubset(t.channels)]
    if not eligible:
        return np.empty((0, len(channels), 0)), channels, []
    if n_samples is None:
        lengths, counts = np.unique([t.n_samples for _, t in eligible],
                                    return_counts=True)
        n_samples = int(lengths[counts.argmax()])
    rows, kept = [], []
    for i, t in eligible:
        if t.n_samples < n_samples:
            continue
        idx = [t.channels.index(c) for c in channels]
        rows.append(t.data[idx, :n_samples])
        kept.append(i)
    arr = np.stack(rows, 0) if rows else np.empty((0, len(channels), n_samples))
    return arr, channels, kept


def build_set(set_dir: str, channels: list[str] | None = None,
              cache: str | None = None, condition: str | None = None,
              n_samples: int | None = None) -> SMNICMISet:
    """Load + stack a set into :class:`SMNICMISet`, caching to ``.npz``.

    Parsing the ~11k gzipped FULL files is the slow step; the cache makes
    subsequent loads instant.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return SMNICMISet(M=z["M"], channels=list(z["channels"]),
                          groups=z["groups"], conditions=z["conditions"],
                          subjects=z["subjects"], trial_idx=z["trial_idx"])
    trials = load_dataset(set_dir, condition=condition)
    M, channels, kept = stack_trials(trials, channels, n_samples)
    s = SMNICMISet(
        M=M, channels=channels,
        groups=np.array([trials[i].group for i in kept], dtype=object),
        conditions=np.array([trials[i].condition for i in kept], dtype=object),
        subjects=np.array([trials[i].subject for i in kept], dtype=object),
        trial_idx=np.array([trials[i].trial_idx for i in kept]),
    )
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, M=s.M, channels=np.array(s.channels, dtype=object),
                 groups=s.groups, conditions=s.conditions,
                 subjects=s.subjects, trial_idx=s.trial_idx)
    return s
