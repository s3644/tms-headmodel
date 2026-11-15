"""Simulated clinical thick-slice MRI series from a 1 mm T1, for the multi-series arms of the R5 scoring rule
(PLAN 10.5): each series is written as NIfTI + a dcm2niix-style sidecar (SliceThickness), as navprep reads them.

  python scripts/sim_series.py T1.nii.gz OUTDIR --series ax:5:1 sag:5:1 [cor:5:1] [--cover sag:120]
                               [--seed 0] [--noise 0.02]

A series is <orientation>:<thickness mm>:<gap mm>; orientation ax (slices along S-I), sag (L-R) or cor (A-P).
In-plane voxels are 1 mm (clinical ones are finer; 1 mm is what the segmenter sees). Every series after the first
is acquired with the head moved: rotation U(-2, 2) deg and translation U(-2, 2) mm per axis, about the image
centre (seeded). --cover keeps only the central N mm across that series' slices (a sagittal series often
misses the ears). Rician noise relative to the 99th percentile. The true motions go in OUTDIR/sim.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from headmodel import multistack as ms  # noqa: E402

AXIS_CODES = {"ax": "SI", "sag": "LR", "cor": "AP"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("t1", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--series", nargs="+", required=True)
    ap.add_argument("--cover", nargs="*", default=[], help="orientation:mm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.02)
    a = ap.parse_args(argv)
    cover = {k: float(v) for k, v in (c.split(":") for c in a.cover)}
    img = nib.load(str(a.t1))
    v = np.asarray(img.dataobj, dtype=np.float32)
    scale = float(np.percentile(v[v > 0], 99))
    vol = torch.from_numpy(v / scale).cuda()
    aff = torch.from_numpy(img.affine).double()
    codes = nib.aff2axcodes(img.affine)
    c = aff[:3, :3] @ (torch.tensor(v.shape, dtype=torch.float64) - 1) / 2 + aff[:3, 3]
    C, Ci = torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    C[:3, 3], Ci[:3, 3] = c, -c
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    log = {"source": a.t1.name, "seed": a.seed, "noise": a.noise, "series": []}
    for k, spec in enumerate(a.series):
        name, thick, gap = spec.split(":")
        thick, gap = float(thick), float(gap)
        axis = [i for i, code in enumerate(codes) if code in AXIS_CODES[name]][0]
        p = np.zeros(6) if k == 0 else np.r_[np.deg2rad(rng.uniform(-2, 2, 3)), rng.uniform(-2, 2, 3)]
        motion = C @ ms.rigid(torch.tensor(p, dtype=torch.float64)) @ Ci
        s = ms.simulate(vol, aff, axis, thick, gap, motion=motion, noise=a.noise, seed=a.seed * 100 + k)
        data, A = s.data, s.affine.clone()
        if name in cover:
            n = data.shape[axis]
            keep = min(n, int(round(cover[name] / (thick + gap))))
            lo = (n - keep) // 2
            sl = [slice(None)] * 3
            sl[axis] = slice(lo, lo + keep)
            data = data[tuple(sl)]
            A[:3, 3] = A[:3, 3] + A[:3, axis] * lo
        f = a.out / f"{k}_{name}.nii.gz"
        nib.save(nib.Nifti1Image((data * scale).cpu().numpy().astype(np.float32), A.numpy()), str(f))
        f.with_name(f"{k}_{name}.json").write_text(json.dumps({"SliceThickness": thick,
                                                                 "SpacingBetweenSlices": thick + gap}))
        log["series"].append({"file": f.name, "orientation": name, "thickness_mm": thick, "gap_mm": gap,
                              "cover_mm": cover.get(name), "motion_params": [round(float(x), 5) for x in p],
                              "motion": motion.tolist()})
    (a.out / "sim.json").write_text(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
