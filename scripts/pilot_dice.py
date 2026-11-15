"""Dice per label of nnU-Net predictions against the reference labels (epoch pilots, user rule 2026-10-28).

    python scripts/pilot_dice.py <pred_dir> <labels_dir> <dataset.json> <out.json>

Every prediction <case>.nii.gz in pred_dir is scored against labels_dir/<case>.nii.gz for each foreground
label of dataset.json. A label absent from both volumes counts as Dice 1. Writes per-case and per-label
Dice, and "mean": the mean over cases of the per-case mean over labels; prints a short summary.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int(np.logical_and(a, b).sum()) / s


def main(pred_dir: str, labels_dir: str, dataset_json: str, out: str) -> None:
    labels = {name: int(v) for name, v in json.loads(Path(dataset_json).read_text())["labels"].items() if int(v) > 0}
    cases = {}
    for p in sorted(Path(pred_dir).glob("*.nii.gz")):
        ref = Path(labels_dir) / p.name
        a = np.asarray(nib.load(p).dataobj)
        b = np.asarray(nib.load(ref).dataobj)
        if a.shape != b.shape:
            raise SystemExit(f"{p.name}: prediction {a.shape} vs reference {b.shape}")
        cases[p.name[:-7]] = {name: dice(a == v, b == v) for name, v in labels.items()}
    if not cases:
        raise SystemExit(f"no predictions in {pred_dir}")
    per_label = {name: float(np.mean([c[name] for c in cases.values()])) for name in labels}
    mean = float(np.mean([np.mean(list(c.values())) for c in cases.values()]))
    Path(out).write_text(json.dumps({"mean": mean, "per_label": per_label, "cases": cases}, indent=1))
    print(f"{len(cases)} cases, mean Dice {mean:.4f}; " + ", ".join(f"{k} {v:.3f}" for k, v in per_label.items()))


if __name__ == "__main__":
    main(*sys.argv[1:5])
