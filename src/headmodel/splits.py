"""Frozen case lists for WP3 / WP3b (the manifest keeps growing while charm queues run).

- WP3 test cases: the 57 held-out cases of WP3's evaluation (keys of runs/wp3/eval.json, commit
  115b491), frozen in runs/wp3b/test_cases_wp3.json. New manifest test cases are not added to it.
- WP3b validation cases: the 30 train cases with the smallest sha1(case id), sorted, computed on the
  134-case train snapshot of runs/wp3/template/template.json, frozen in runs/wp3b/val_cases.json. The
  file is authoritative: on today's (larger) train split the rule picks other cases, so it is never
  recomputed once the file exists.

Both files are derived from the data manifest and live under runs/ (gitignored).
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WP3_EVAL = REPO / "runs/wp3/eval.json"
WP3_TEST_FILE = REPO / "runs/wp3b/test_cases_wp3.json"
VAL_FILE = REPO / "runs/wp3b/val_cases.json"
VAL_PROVENANCE = REPO / "runs/wp3b/val_cases_provenance.json"
WP3_TRAIN_SNAPSHOT = REPO / "runs/wp3/template/template.json"
N_WP3_TEST = 57
N_VAL = 30


def _atomic_text(f: Path, text: str) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, f)


def sha1_rule(cases, n: int | None = None) -> list[str]:
    """The n (default N_VAL) cases with the smallest sha1(case id) hex, sorted."""
    n = N_VAL if n is None else n
    return sorted(sorted(cases, key=lambda c: hashlib.sha1(c.encode()).hexdigest())[:n])


def wp3_test_cases() -> list[str]:
    """The 57 WP3 test cases (frozen file; created once from runs/wp3/eval.json)."""
    if WP3_TEST_FILE.is_file():
        test = json.loads(WP3_TEST_FILE.read_text())["cases"]
    else:
        test = sorted(r["case"] for r in json.loads(WP3_EVAL.read_text())["cases"])
        _atomic_text(WP3_TEST_FILE, json.dumps({
            "what": "the WP3 held-out test cases (keys of runs/wp3/eval.json, commit 115b491), frozen",
            "n": len(test), "cases": test}, indent=1))
    if len(test) != N_WP3_TEST:
        raise RuntimeError(f"{WP3_TEST_FILE}: {len(test)} cases, expected {N_WP3_TEST}")
    return test


def val_cases(man: dict | None = None) -> list[str]:
    """The 30 WP3b validation cases. The fixed file wins; if it is absent it is created by the sha1
    rule on the WP3 train snapshot (runs/wp3/template/template.json), else on man's train split.
    Warns (never fails) when the rule on man's current train split gives another list."""
    if VAL_FILE.is_file():
        val = json.loads(VAL_FILE.read_text())
    else:
        if WP3_TRAIN_SNAPSHOT.is_file():
            snap = sorted(c["case"] for c in json.loads(WP3_TRAIN_SNAPSHOT.read_text())["cases"])
            src = str(WP3_TRAIN_SNAPSHOT.relative_to(REPO))
        else:
            snap = sorted(c for c, i in man["cases"].items() if i["split"] == "train")
            src = "manifest train split"
        val = sha1_rule(snap)
        _atomic_text(VAL_FILE, json.dumps(val, indent=1))
        _atomic_text(VAL_PROVENANCE, json.dumps({"rule": f"{N_VAL} smallest sha1(case id) hex, sorted",
                                                 "snapshot": src, "train_cases": snap}, indent=1))
    if man is not None:
        train = [c for c, i in man["cases"].items() if i["split"] == "train"]
        if not set(val) <= set(train):
            raise RuntimeError(f"{VAL_FILE}: validation cases outside the train split: {sorted(set(val) - set(train))}")
        if sha1_rule(train) != val:
            warnings.warn(f"{VAL_FILE} is the sha1 rule on the train snapshot of {VAL_PROVENANCE.name}, not on "
                          f"the current {len(train)} train cases; the fixed file is used", stacklevel=2)
    return val
