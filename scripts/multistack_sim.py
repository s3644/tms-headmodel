"""Exploratory (PLAN 10.5): how much does a second 5 mm stack (sagittal + axial T1) recover over one?

On a round-1 *training* head (never a test case): simulate 5 mm stacks (with a 1 mm gap, Rician noise, the head
moved between the series), then compare with the 1 mm original, inside the head:
- axial only / sagittal only, plain interpolation;
- axial only, reconstructed (deconvolution + TV, no second stack);
- axial + sagittal, registered and reconstructed;
- the same with a typical sagittal coverage (120 mm across, so the ears and lateral scalp are outside it).
Metrics: intensity RMSE in the head (relative to the 99th percentile), skin-surface distance (threshold head
mask), and the pose error of the registration. No navigation points are scored here: that needs a sealed rule.

Usage: python scripts/multistack_sim.py [CASE ...]   -> runs/multistack_sim/<case>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from headmodel import multistack as ms  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs/multistack_sim"


def head_mask(v: np.ndarray, thr: float) -> np.ndarray:
    m = v > thr
    lab, n = ndimage.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))


def surface(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m)


def skin_error(pred: np.ndarray, truth_mask: np.ndarray, thr: float, region=None) -> dict:
    d = ndimage.distance_transform_edt(~surface(truth_mask))
    s = surface(head_mask(pred, thr))
    if region is not None:
        s &= region
    e = d[s]
    return {"mean_mm": round(float(e.mean()), 3), "p95_mm": round(float(np.percentile(e, 95)), 3)}


def main(case: str) -> dict:
    m2m = Path(json.loads((REPO / "data/manifests/round1.json").read_text())["cases"][case]["m2m"])
    assert json.loads((REPO / "data/manifests/round1.json").read_text())["cases"][case]["split"] == "train"
    img = nib.load(m2m / "T1.nii.gz")
    v = np.asarray(img.dataobj, dtype=np.float32)
    v /= np.percentile(v[v > 0], 99)
    dev = "cuda"
    truth = torch.from_numpy(v).to(dev)
    aff = torch.from_numpy(img.affine).double()
    axcodes = nib.aff2axcodes(img.affine)
    ax_axis = [i for i, c in enumerate(axcodes) if c in "SI"][0]
    sag_axis = [i for i, c in enumerate(axcodes) if c in "LR"][0]

    c = aff[:3, :3] @ (torch.tensor(v.shape, dtype=torch.float64) - 1) / 2 + aff[:3, 3]
    C, Ci = torch.eye(4, dtype=torch.float64), torch.eye(4, dtype=torch.float64)
    C[:3, 3], Ci[:3, 3] = c, -c
    true_motion = C @ ms.rigid(torch.tensor([0.03, -0.02, 0.025, 1.5, -2.0, 1.0], dtype=torch.float64)) @ Ci

    t0 = time.time()
    ax = ms.simulate(truth, aff, ax_axis, 5.0, 1.0, noise=0.02, seed=1)
    sag = ms.simulate(truth, aff, sag_axis, 5.0, 1.0, motion=true_motion, noise=0.02, seed=2)
    thr = 0.15
    tmask = head_mask(v, thr)
    region_in = ndimage.binary_dilation(tmask, iterations=3)

    def rmse(x):
        x = ms.to_numpy(x)
        return round(float(np.sqrt(((x - v)[region_in] ** 2).mean())), 4)

    res = {"case": case, "split": "train", "stacks": "5 mm slices, 1 mm gap, Rician noise 0.02; sagittal series "
           "moved by 1.7-1.4 deg, (1.5, -2.0, 1.0) mm"}
    I = torch.eye(4, dtype=torch.float64)
    rows = {}
    for name, x in [("axial only, interpolated", ms.interp_to(ax, v.shape, aff)),
                    ("sagittal only, interpolated", ms.interp_to(sag, v.shape, aff, true_motion)),
                    ("axial only, reconstructed", ms.reconstruct([ax], [I.to(dev)], v.shape, aff))]:
        rows[name] = {"rmse": rmse(x), "skin": skin_error(ms.to_numpy(x), tmask, thr)}

    t1 = time.time()
    x, mot = ms.register_and_reconstruct([ax, sag], v.shape, aff)
    reg_s = time.time() - t1
    # pose error at the head surface: the estimated vs true motion, applied to the true skin points
    sk = np.argwhere(surface(tmask))
    pts = torch.from_numpy(sk @ img.affine[:3, :3].T + img.affine[:3, 3])
    err = (pts @ mot[1].cpu()[:3, :3].T + mot[1].cpu()[:3, 3]) - (pts @ true_motion[:3, :3].T + true_motion[:3, 3])
    res["registration"] = {"seconds_with_reconstruction": round(reg_s, 1),
                           "skin_pose_error_mm_mean": round(float(err.norm(dim=1).mean()), 3),
                           "skin_pose_error_mm_max": round(float(err.norm(dim=1).max()), 3)}
    rows["axial + sagittal, registered + reconstructed"] = {"rmse": rmse(x), "skin": skin_error(ms.to_numpy(x), tmask, thr)}

    # typical clinical sagittal coverage: 20 slices x 6 mm = 120 mm across, centred
    n = sag.data.shape[sag.axis]
    lo = max(0, n // 2 - 10)
    sl = [slice(None)] * 3
    sl[sag.axis] = slice(lo, lo + 20)
    A = sag.affine.clone()
    A[:3, 3] = A[:3, 3] + A[:3, sag.axis] * lo                  # a cropped series: its first slice moves
    sag_c = ms.Stack(sag.data[tuple(sl)].contiguous(), A, sag.axis, sag.thick)
    x2 = ms.reconstruct([ax, sag_c], mot, v.shape, aff)
    rows["axial + sagittal (120 mm coverage)"] = {"rmse": rmse(x2), "skin": skin_error(ms.to_numpy(x2), tmask, thr)}
    res["rows"] = rows
    res["total_s"] = round(time.time() - t0, 1)
    res["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{case}.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    for c in sys.argv[1:] or ["1BA076"]:
        print(json.dumps(main(c), indent=1))
