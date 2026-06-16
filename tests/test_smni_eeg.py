"""Tests for qcccm.neuroai.smni_eeg — the SMNI_CMI .rd loader.

Uses synthetic .rd files (no real dataset needed) to exercise parsing of the
ASCII format, gzip handling, group/condition tagging, and stacking.
"""
from __future__ import annotations

import gzip
import os

import numpy as np

from qcccm.neuroai import smni_eeg as se


def _write_rd(path: str, subject: str, cond_header: str, trial: int,
              channels: list[str], n_samples: int, gz: bool = False) -> None:
    lines = [f"# {subject}.rd",
             f"# 10 trials, {len(channels)} chans, {n_samples} samples",
             "# 3.906000 msecs uV",
             f"# {cond_header} , trial {trial}"]
    for ci, ch in enumerate(channels):
        lines.append(f"# {ch} chan {ci}")
        for s in range(n_samples):
            lines.append(f"{trial} {ch} {s} {float(ci + 0.1 * s):.3f}")
    text = "\n".join(lines) + "\n"
    opener = gzip.open if gz else open
    with opener(path, "wt") as fh:
        fh.write(text)


def _make_set(root: str) -> str:
    chans = ["FP1", "FP2", "F7"]
    # alcoholic subject, uncompressed; control subject, gzipped
    a = os.path.join(root, "co2a0000001")
    c = os.path.join(root, "co2c0000002")
    os.makedirs(a); os.makedirs(c)
    _write_rd(os.path.join(a, "co2a0000001.rd.000"), "co2a0000001",
              "S1 obj", 0, chans, 16)
    _write_rd(os.path.join(a, "co2a0000001.rd.001"), "co2a0000001",
              "S2 match", 1, chans, 16)
    _write_rd(os.path.join(c, "co2c0000002.rd.000.gz"), "co2c0000002",
              "S2 nomatch", 0, chans, 16, gz=True)
    return root


def test_parse_single_trial(tmp_path):
    p = os.path.join(tmp_path, "co2a0000001.rd.000")
    _write_rd(p, "co2a0000001", "S2 match", 5, ["FP1", "FP2"], 8)
    t = se.load_trial(p)
    assert t is not None
    assert t.subject == "co2a0000001"
    assert t.group == "alcoholic"
    assert t.condition == "S2match"
    assert t.trial_idx == 5
    assert t.data.shape == (2, 8)
    assert t.channels == ["FP1", "FP2"]


def test_gzip_and_group_condition(tmp_path):
    _make_set(str(tmp_path))
    trials = se.load_dataset(str(tmp_path))
    assert len(trials) == 3
    groups = sorted({t.group for t in trials})
    conds = sorted({t.condition for t in trials})
    assert groups == ["alcoholic", "control"]
    assert conds == ["S1", "S2match", "S2nomatch"]
    # the control trial was gzipped and must still parse
    assert any(t.group == "control" and t.condition == "S2nomatch" for t in trials)


def test_build_set_and_alignment(tmp_path):
    _make_set(str(tmp_path))
    s = se.build_set(str(tmp_path))
    assert s.M.shape == (3, 3, 16)
    assert set(s.groups) == {"alcoholic", "control"}
    assert len(s.groups) == s.M.shape[0] == len(s.subjects) == len(s.conditions)


def test_build_set_cache_roundtrip(tmp_path):
    _make_set(str(tmp_path))
    cache = os.path.join(tmp_path, "cache.npz")
    s1 = se.build_set(str(tmp_path), cache=cache)
    assert os.path.exists(cache)
    s2 = se.build_set(str(tmp_path), cache=cache)        # second load hits cache
    assert np.allclose(s1.M, s2.M)
    assert list(s1.channels) == list(s2.channels)


def test_stack_drops_short_trials(tmp_path):
    chans = ["FP1", "FP2"]
    d = os.path.join(tmp_path, "co2a0000001"); os.makedirs(d)
    _write_rd(os.path.join(d, "co2a0000001.rd.000"), "co2a0000001",
              "S1 obj", 0, chans, 16)
    _write_rd(os.path.join(d, "co2a0000001.rd.001"), "co2a0000001",
              "S1 obj", 1, chans, 16)
    _write_rd(os.path.join(d, "co2a0000001.rd.002"), "co2a0000001",
              "S1 obj", 2, chans, 4)          # short outlier
    trials = se.load_subject(d)
    M, ch, kept = se.stack_trials(trials)
    assert M.shape == (2, 2, 16)              # modal length 16; short one dropped
    assert len(kept) == 2
