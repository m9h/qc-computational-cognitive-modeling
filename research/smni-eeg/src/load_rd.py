"""Loader for the SMNI_CMI EEG `.rd` ASCII trial files.

File format (one file per trial, inside each subject directory)::

    # co2a0000364.rd                                 <- subject/run id
    # 120 trials, 64 chans, 416 samples 368 post_stim samples
    # 3.906000 msecs uV                              <- sample interval, units
    # S2 match , trial 101                           <- condition, trial index
    # FP1 chan 0                                      <- per-channel header (skipped)
    101 FP1 0 -0.844                                 <- trial chan sample value(uV)
    ...

Data rows are 4 whitespace columns: ``trial_idx channel sample value``.
Comment/header lines start with ``#`` (note ``# FP1 chan 0`` also has 4 tokens,
so we additionally require the first token to be an integer).

This module is numpy-only (no pandas/scipy dependency).
"""
from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass, field

import numpy as np


def _open_text(path: str):
    """Open a trial file, transparently decompressing .gz (TEST/FULL are gzipped)."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")

# subject id like co2a0000364 / co3c0000457 -> group letter is the char after `co<digit>`
_SUBJ_RE = re.compile(r"co\d([ac])", re.IGNORECASE)

# normalize the condition header to a compact tag
_COND_MAP = {
    "S1 obj": "S1",
    "S2 match": "S2match",
    "S2 nomatch": "S2nomatch",
}

FS_HZ = 256.0  # sampling rate (sample interval 3.906 ms = 1/256 s)


@dataclass
class Trial:
    subject: str
    group: str               # 'alcoholic' | 'control'
    condition: str           # 'S1' | 'S2match' | 'S2nomatch' | 'unknown'
    trial_idx: int
    channels: list[str]      # length n_chan, in file order
    data: np.ndarray         # shape (n_chan, n_samples), float64, microvolts
    path: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_chan(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]


def _parse_group(subject: str) -> str:
    m = _SUBJ_RE.match(subject)
    if not m:
        return "unknown"
    return "alcoholic" if m.group(1).lower() == "a" else "control"


def _parse_condition(line: str) -> str:
    # line e.g. "# S2 match , trial 101"  -> "S2 match"
    body = line.lstrip("#").strip()
    cond = body.split(",")[0].strip()
    return _COND_MAP.get(cond, "unknown")


def load_trial(path: str) -> Trial | None:
    """Parse one `.rd` trial file. Returns None if it isn't a valid trial file."""
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
                    condition = _parse_condition(line)
                    try:
                        trial_idx = int(line.rsplit("trial", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            tok_trial, chan, tok_sample, tok_val = parts
            if not tok_trial.lstrip("-").isdigit():
                continue  # skips stray non-trial lines (e.g. notebook code)
            try:
                sample = int(tok_sample)
                val = float(tok_val)
            except ValueError:
                continue
            if chan not in chan_to_samples:
                chan_to_samples[chan] = {}
                chan_order.append(chan)
            chan_to_samples[chan][sample] = val

    if not chan_order:
        return None  # not a trial file

    n_samples = max(len(v) for v in chan_to_samples.values())
    data = np.full((len(chan_order), n_samples), np.nan, dtype=np.float64)
    for ci, chan in enumerate(chan_order):
        for s, v in chan_to_samples[chan].items():
            if 0 <= s < n_samples:
                data[ci, s] = v

    subject = subject or os.path.basename(path).split(".")[0]
    return Trial(
        subject=subject,
        group=_parse_group(subject),
        condition=condition or "unknown",
        trial_idx=trial_idx,
        channels=chan_order,
        data=data,
        path=path,
        meta={"fs_hz": FS_HZ},
    )


def load_subject(subject_dir: str) -> list[Trial]:
    trials = []
    for name in sorted(os.listdir(subject_dir)):
        p = os.path.join(subject_dir, name)
        if not os.path.isfile(p) or ".rd" not in name:
            continue
        t = load_trial(p)
        if t is not None:
            trials.append(t)
    return trials


def load_dataset(set_dir: str, condition: str | None = None,
                 group: str | None = None, limit_subjects: int | None = None
                 ) -> list[Trial]:
    """Load all trials under a set dir (TRAIN/TEST/FULL), with optional filters."""
    subdirs = sorted(
        d for d in os.listdir(set_dir)
        if os.path.isdir(os.path.join(set_dir, d)) and d.lower().startswith("co")
    )
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
                 n_samples: int | None = None, return_index: bool = False):
    """Stack trials into array (n_trials, n_chan, n_samples) on a common channel
    set and a common sample length.

    The target length defaults to the *modal* (most common) sample count, so a
    stray short/truncated trial file can't drag everything down. Trials missing a
    channel, or shorter than the target length, are dropped (and counted).

    Returns (arr, channels); if return_index, also a list of the kept trial
    indices (positions in `trials`) so caller metadata can be aligned.
    """
    empty = np.empty((0, len(channels or []), 0))
    if not trials:
        return (empty, [], []) if return_index else (empty, [])
    if channels is None:
        channels = trials[0].channels
    chan_set = set(channels)
    eligible = [(i, t) for i, t in enumerate(trials)
                if chan_set.issubset(t.channels)]
    if not eligible:
        return (empty, channels, []) if return_index else (empty, channels)
    if n_samples is None:
        lengths, counts = np.unique([t.n_samples for _, t in eligible],
                                    return_counts=True)
        n_samples = int(lengths[counts.argmax()])
    rows, kept, dropped = [], [], 0
    for i, t in eligible:
        if t.n_samples < n_samples:
            dropped += 1
            continue
        idx = [t.channels.index(c) for c in channels]
        rows.append(t.data[idx, :n_samples])
        kept.append(i)
    if dropped:
        print(f"[stack_trials] dropped {dropped} trial(s) shorter than "
              f"{n_samples} samples")
    arr = np.stack(rows, axis=0) if rows else np.empty((0, len(channels), n_samples))
    return (arr, channels, kept) if return_index else (arr, channels)


def build_set(set_dir: str, channels: list[str] | None = None,
              cache: str | None = None, condition: str | None = None,
              n_samples: int | None = None):
    """Load + stack a set into arrays with aligned metadata, caching to .npz.

    Returns dict with: M (N,C,T), channels, groups, conditions, subjects, trial_idx.
    Parsing ~11k gzipped files (FULL) is the slow part; the cache makes re-runs
    instant.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] loaded {cache}  M={z['M'].shape}")
        return {"M": z["M"], "channels": list(z["channels"]),
                "groups": z["groups"], "conditions": z["conditions"],
                "subjects": z["subjects"], "trial_idx": z["trial_idx"]}
    trials = load_dataset(set_dir, condition=condition)
    M, channels, kept = stack_trials(trials, channels=channels,
                                     n_samples=n_samples, return_index=True)
    meta = {
        "M": M,
        "channels": np.array(channels, dtype=object),
        "groups": np.array([trials[i].group for i in kept], dtype=object),
        "conditions": np.array([trials[i].condition for i in kept], dtype=object),
        "subjects": np.array([trials[i].subject for i in kept], dtype=object),
        "trial_idx": np.array([trials[i].trial_idx for i in kept]),
    }
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez(cache, **meta)
        print(f"[cache] wrote {cache}  M={M.shape}")
    meta["channels"] = list(channels)
    return meta


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/Workspace/smni-eeg/data/TRAIN")
    ts = load_dataset(d, limit_subjects=2)
    print(f"loaded {len(ts)} trials from {d}")
    if ts:
        t = ts[0]
        print(f"  e.g. subject={t.subject} group={t.group} cond={t.condition} "
              f"chans={t.n_chan} samples={t.n_samples}")
        arr, chans = stack_trials(ts)
        print(f"  stacked: {arr.shape} over {len(chans)} channels")
