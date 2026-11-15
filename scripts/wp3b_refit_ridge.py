"""WP3b Fix: refit the ridge frames on the extended lambda grid (headmodel.frame.RIDGE_LAMS).

The first fits used lambda in (0.01 .. 3.0) and picked 0.01, the end of the grid, in both templates. The
features are not recomputed: they are the stored per-case affines the first fits used, so only lambda
(chosen by leave-one-out on the same train cases) changes. For each template dir the old model is kept
as frame_ridge_lam0.01.json (checked to be reproduced by the old grid) and frame_ridge.json is replaced;
both now store the fitted case list.

  runs/wp3b/template_trainminusval   train minus the 30 validation cases; features from
                                     runs/wp3b/frame/<case>.json (raw, labels_brainonly, W); also writes
                                     affines.json (the rows the model was fitted from)
  runs/wp3b/template                 all 134 WP3 train cases; features from its affines.json

No validation or test case is read. Usage:  PYTHONPATH=src python scripts/wp3b_refit_ridge.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import frame, splits  # noqa: E402

W3B = REPO / "runs/wp3b"
OLD_NAME = "frame_ridge_lam0.01.json"
OLD_LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def _write(f: Path, obj) -> None:
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, f)


def rows_trainminusval() -> dict:
    tdir = W3B / "template_trainminusval"
    val = set(splits.val_cases())
    test = set(splits.wp3_test_cases())
    fit = sorted({c["case"] for c in json.loads((tdir / "template.json").read_text())["cases"]} - val)
    rows = {}
    for c in fit:   # the rows wp3b_frame.py --ridge raw+labels_brainonly --write-ridge used
        f = W3B / "frame" / f"{c}.json"
        if not f.is_file():
            continue
        r = json.loads(f.read_text())
        if all(n in r["A"] for n in ("raw", "labels_brainonly")):
            rows[c] = {"raw": r["A"]["raw"], "brain": r["A"]["labels_brainonly"], "W": r["W"]}
    assert not (set(rows) & (val | test))
    return rows


def rows_template() -> dict:
    rows = json.loads((W3B / "template/affines.json").read_text())
    assert not (set(rows) & set(splits.wp3_test_cases()))
    return rows


def refit(tdir: Path, rows: dict, write_affines: bool) -> dict:
    cur = tdir / frame.RIDGE_FILE
    old_f = tdir / OLD_NAME
    old = json.loads((old_f if old_f.is_file() else cur).read_text())
    cases = sorted(rows)
    feats = [[np.asarray(rows[c]["raw"]), np.asarray(rows[c]["brain"])] for c in cases]
    W = [np.asarray(rows[c]["W"]) for c in cases]
    chk = frame.fit_ridge(feats, W, lams=OLD_LAMS)
    if not (chk["lam"] == old["lam"] and np.allclose(chk["B"], old["B"]) and chk["n_cases"] == old["n_cases"]):
        raise SystemExit(f"{tdir}: the stored rows do not reproduce the first ridge fit")
    if not old_f.is_file():
        _write(old_f, {**old, "cases": cases, "lams": list(OLD_LAMS),
                       "note": "first WP3b ridge fit (lambda grid 0.01 .. 3.0); case list added by wp3b_refit_ridge.py"})
    meta = {k: old[k] for k in ("features", "what", "split", "licence") if k in old}
    new = frame.fit_ridge(feats, W, meta={**meta, "built": time.strftime("%Y-%m-%d %H:%M"),
                                          "refit": "lambda grid extended (frame.RIDGE_LAMS), same features and "
                                                   f"cases as {OLD_NAME}"}, cases=cases)
    _write(cur, new)
    if write_affines:
        _write(tdir / "affines.json", rows)
    return {"dir": str(tdir.relative_to(REPO)), "n": len(cases), "old_lam": old["lam"], "old_loo": old["loo_box_mm"],
            "new_lam": new["lam"], "new_loo": new["loo_box_mm"], "edge": new["lam_at_grid_edge"]}


def main():
    for tdir, rows, wa in ((W3B / "template_trainminusval", rows_trainminusval(), True),
                           (W3B / "template", rows_template(), False)):
        print(refit(tdir, rows, wa), flush=True)


if __name__ == "__main__":
    main()
