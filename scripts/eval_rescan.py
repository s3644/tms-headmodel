"""TASK F: charm's own scan-rescan floor on the WP3 metrics (OASIS-1 reliability subset).

Pairs per subject (ids from runs/rescan/cases.json, charm folders /mnt/24tb/headmodel/rescan/<id>/m2m_<id>):
  same_session   A = MR1 mpr-1 vs B = MR1 mpr-2 (repeat scan minutes apart)
  different_day  A = MR1 mpr-1 vs C = MR2 mpr-1 (rescan on a later day)
Per pair: B's T1 is rigidly registered to A's T1 (headmodel.rescan.register_rigid: 6 dof, NCC inside
A's charm brain mask = WM + GM + CSF of label_prep/tissue_labeling_upsampled, dilated 3 voxels; started
from the rotation of charm's two MNI affines and the brain centroids). B's charm outputs are mapped
into A's space with that rigid transform and compared with A's. Metrics (labels as in eval.md):
  targets            the 10 MNI targets of scripts/eval_register.py through simnibs.mni2subject_coords
  {eeg,fid}_unprojected  the EEG10-10_UI_Jurak_2007 cap / Fiducials warped by each scan's MNI2Conform,
                     no skin projection (warp_coordinates Generic rows): the eval_register gate metric
  {eeg,fid}_same_skin    B's warped cap projected on A's charm mesh skin (A's skin moved into B's space by
                     the rigid fit, projected there, mapped back) vs A's eeg_positions csv: skin
                     variability removed, as WP3's proj_charm_skin (our warp on charm's skin vs charm's csv)
  {eeg,fid}_own_skin     eeg_positions/*.csv as delivered by each scan (each projected on its own skin):
                     includes the scan-to-scan variability of the segmented skin; no WP3 analogue
Per pair mean / max over points, then mean of pair means / SD / worst pair mean / worst point / p95 /
pairs passing (mean <= 2 mm and max <= 5 mm), per pair kind; also with the points within FOV_EDGE_MM of
the T1 field-of-view edge in A or B left out (the OASIS T1s are 160 mm left-right and face-masked, so in
about half of them the scalp is cut at the lateral edges), and per image the head voxels (charm labels > 0)
on each face of the T1 box (truncated = more than TRUNC_VOXELS on a face other than inferior).

Residual of the rigid step (recorded per pair): inverse consistency (B->A registered separately, compared
with the inverse of A->B at the evaluated points), and the A-vs-A check (A registered to itself from a
~7 deg / 7 mm perturbed start; identical images, so it checks convergence only). The fit itself is
checked by the synthetic pair below, started from the identity (not from the true answer).

--ours (GPU, optional): our registration (headmodel.register defaults, charm's labels block-majority'd to
the T1 grid as in scripts/eval_register.py) on the A images vs charm on the same images: targets and
EEG / fiducials unprojected, and projected on charm's mesh skin (same_skin) and on the label skin
(eval_register.eval_projected). No choice is made on these cases (neither WP3 test nor validation cases).

--synthetic M2M: smoke test without OASIS runs: B = a copy of the charm folder M2M (use a training case,
e.g. ~/Projects/ct-headmodel/runs/1BA076/m2m_ref, never a WP3 test case) seen through a known rigid transform
(T1 resampled on a shifted grid, labels / warps / csv positions transformed exactly), written to
runs/rescan/synthetic/; the rigid fits start from the identity, so the pair checks that the fit recovers
an 8.5 deg / 11.5 mm motion, and should score ~0 mm.

--ours-r4: the same with the adopted WP3b round-4 setting (register.adopted_setting), in runs/rescan/ours_r4.
Every comparison also gets the rTMS-set EEG rows (eval_register.RTMS_EEG, the WP3 gate scope from round 4 on).

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/eval_rescan.py [--ours] [--ours-r4] [--redo] [--device cuda]
                       PYTHONPATH=src python scripts/eval_rescan.py --synthetic ~/Projects/ct-headmodel/runs/1BA076/m2m_ref
Writes runs/rescan/pairs/<pair>.json (resumable; unfinished charm runs are listed as pending, failed ones
as failed), runs/rescan/eval.json and runs/rescan/eval.md. OASIS DUA: everything stays on this machine
and out of git.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io
import scipy.ndimage as ndi
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import dataset, register, rescan  # noqa: E402
import eval_register as ev  # noqa: E402

ROOT = Path("/mnt/24tb/headmodel/rescan")
OUT = REPO / "runs/rescan"
QUEUE_LOG = REPO / "runs/logs/queue_rescan.log"
BRAIN = (1, 2, 3)   # charm WM, GM, CSF
CAPS = {"eeg": "EEG10-10_UI_Jurak_2007", "fid": "Fiducials"}
VARIANTS = ("unprojected", "same_skin", "own_skin")
GROUPS = ("targets",) + tuple(f"{c}_{v}" for c in CAPS for v in VARIANTS)
LABELS = {"targets": "targets (mni2subject, unprojected)",
          "eeg_unprojected": "EEG warped, unprojected (WP3 gate metric)",
          "eeg_same_skin": "EEG projected on the same skin (A's charm skin; ours: charm's skin)",
          "eeg_own_skin": "EEG csv as delivered, each scan on its own skin",
          "fid_unprojected": "fiducials warped, unprojected",
          "fid_same_skin": "fiducials projected on the same skin",
          "fid_own_skin": "fiducials csv as delivered, own skin",
          "eeg_label_skin": "EEG projected on the label skin (ours only)",
          "eeg_rtms_unprojected": "EEG rTMS set warped, unprojected (WP3 gate scope from round 4)",
          "eeg_rtms_same_skin": "EEG rTMS set projected on the same skin"}
# rows beyond GROUPS: ours only, and the rTMS-set EEG positions (eval_register.RTMS_EEG) taken from the whole cap
EXTRA = ("eeg_label_skin", "eeg_rtms_unprojected", "eeg_rtms_same_skin")
KINDS = {"same_session": "B", "different_day": "C"}
TARGETS = np.array(list(ev.TARGETS.values()), float)
FOV_EDGE_MM = 10.0     # points closer than this to the T1 box edge (in A or B) are left out of the 'inside' rows
TRUNC_VOXELS = 200     # head voxels on a face of the T1 box above which the image counts as truncated there
INIT_PERTURB = ([0.06, -0.05, 0.07], [5.0, -4.0, 3.0])   # A-vs-A start: ~6.9 deg about the brain centroid, 7.1 mm


def finished(m2m: Path) -> bool:
    log = m2m / "charm_log.html"
    return any(m2m.glob("*.msh")) and log.is_file() and "charm run finished" in log.read_text(errors="ignore")


def queue_events(log: Path = QUEUE_LOG) -> dict:
    """Last queue event per case id in scripts/queue_rescan.sh's log: start / done / FAILED."""
    ev_ = {}
    if log.is_file():
        for line in log.read_text(errors="ignore").splitlines():
            m = re.match(r"^\S+ \S+ (OAS1_\S+) (start|done|FAILED)\b", line)
            if m:
                ev_[m.group(1)] = m.group(2)
    return ev_


def status(case: str, events: dict | None = None) -> str:
    """done / failed / pending for one charm run of the rescan queue. failed: the queue logged FAILED as
    its last event, or charm.out ends in a Python traceback, and the run did not finish."""
    m2m = ROOT / case / f"m2m_{case}"
    if finished(m2m):
        return "done"
    events = queue_events() if events is None else events
    if events.get(case) == "FAILED":
        return "failed"
    out = ROOT / case / "charm.out"
    if out.is_file() and events.get(case) != "start":
        tail = out.read_text(errors="ignore")[-20000:]
        if "Traceback (most recent call last)" in tail:
            return "failed"
    return "pending"


def read_csv(f: Path) -> tuple[list[str], list[str], np.ndarray]:
    types, names, pts = [], [], []
    with open(f) as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                types.append(row[0])
                pts.append([float(v) for v in row[1:4]])
                names.append(row[4] if len(row) > 4 else "")
    return types, names, np.array(pts)


def targets(m2m: Path) -> np.ndarray:
    import simnibs

    return np.asarray(simnibs.mni2subject_coords(TARGETS, str(m2m)), float)


def head_faces(lab: np.ndarray, aff: np.ndarray) -> dict:
    """Head voxels (labels > 0) on each face of the voxel box, keyed by the anatomical side the face is
    on (R/L/A/P/S/I from the affine's dominant world axis)."""
    out = {}
    for ax in range(3):
        w = int(np.argmax(np.abs(aff[:3, ax])))
        pos = "RAS"[w]
        neg = "LPI"[w]
        sign = np.sign(aff[w, ax])
        lo = int((np.take(lab, 0, axis=ax) > 0).sum())
        hi = int((np.take(lab, lab.shape[ax] - 1, axis=ax) > 0).sum())
        out[neg if sign > 0 else pos] = lo
        out[pos if sign > 0 else neg] = hi
    return out


def nifti_faces(cases: list[str]) -> dict:
    """Per input T1 (ROOT/nifti/<id>.nii.gz, before charm): head voxels on each face of the box from the
    intensity foreground (prep_oasis_rescan.foreground) and the truncated faces. Cached in
    runs/rescan/fov_faces.json."""
    from prep_oasis_rescan import foreground

    f = OUT / "fov_faces.json"
    out = json.loads(f.read_text()) if f.is_file() else {}
    for c in cases:
        nii = ROOT / "nifti" / f"{c}.nii.gz"
        if c in out or not nii.is_file():
            continue
        img = nib.load(nii)
        fc = head_faces(foreground(np.asarray(img.dataobj, np.float32)).astype(np.uint8), img.affine)
        out[c] = {"faces": fc, "truncated": [k for k, n in fc.items() if k != "I" and n > TRUNC_VOXELS]}
    f.write_text(json.dumps(out, indent=1))
    return out


class Head:
    """T1, brain mask (on the T1 grid), brain centroid, charm's MNI -> subject affine, the head voxels on
    the T1 box faces and (lazily) charm's mesh skin of one m2m."""

    def __init__(self, m2m: Path):
        self.m2m = Path(m2m)
        self.img = nib.load(self.m2m / "T1.nii.gz")
        self.t1 = dataset._vol(self.img).astype(np.float32)
        self.aff = self.img.affine.copy()
        up = nib.load(self.m2m / dataset.SEG_LABELS)
        lab = rescan.labels_on_grid(dataset._vol(up), up.affine, self.t1.shape, self.aff)
        self.mask = ndi.binary_dilation(np.isin(lab, BRAIN), iterations=3)
        self.centroid = rescan.apply(self.aff, np.argwhere(np.isin(lab, BRAIN))).mean(0)
        self.W = scipy.io.loadmat(self.m2m / "segmentation/coregistrationMatrices.mat")["worldToWorldTransformMatrix"]
        self.faces = head_faces(lab, self.aff)
        self._skin = None

    @property
    def truncated(self) -> list[str]:
        return [k for k, n in self.faces.items() if k != "I" and n > TRUNC_VOXELS]

    def skin(self):
        if self._skin is None:
            self._skin = ev.charm_skin(self.m2m)
        return self._skin

    def margin(self, pts: np.ndarray) -> np.ndarray:
        return ev._fov_margin_mm(self.img, np.asarray(pts, float))


def align(a: Head, b: Head, device: str, init: str = "affines") -> dict:
    """M_ab (A world -> B world), the reverse fit and the A-vs-A check. init: 'affines' (charm's MNI
    affines + brain centroids, the real-data start) or 'identity'."""
    t0 = time.perf_counter()
    if init == "identity":
        i_ab = i_ba = np.eye(4)
    else:
        i_ab = rescan.init_from_affines(a.W, b.W, a.centroid, b.centroid)
        i_ba = rescan.init_from_affines(b.W, a.W, b.centroid, a.centroid)
    fit = rescan.register_rigid(a.t1, a.aff, b.t1, b.aff, a.mask, init=i_ab, device=device)
    rev = rescan.register_rigid(b.t1, b.aff, a.t1, a.aff, b.mask, init=i_ba, device=device)
    pert = rescan.rigid(*INIT_PERTURB, a.centroid)
    self_fit = rescan.register_rigid(a.t1, a.aff, a.t1, a.aff, a.mask, init=pert, device=device)
    return {"M_ab": fit["M"], "M_ba": rev["M"], "M_aa": self_fit["M"], "M_init": i_ab, "ncc": fit["ncc"],
            "ncc_init": fit["ncc_init"], "ncc_rev": rev["ncc"], "seconds": time.perf_counter() - t0, "init": init}


def cap_points(a: Head, b: Head, M_ab: np.ndarray) -> tuple[dict, dict, dict]:
    """Per cap: A's and B's points for each variant (A's in A world, B's in B world) and the names.
    unprojected: Generic rows of warp_coordinates; same_skin: B's warp projected on A's skin moved into B's
    space by M_ab (A's side: A's csv); own_skin: each scan's eeg_positions csv."""
    pa, pb, names = {}, {}, {}
    skin_b = copy.deepcopy(a.skin())
    skin_b.nodes.node_coord = rescan.apply(M_ab, skin_b.nodes.node_coord)
    for c, cap in CAPS.items():
        rows = ev._cap_rows(cap)
        n = rows[3][: len(rows[3]) // 2]
        na, csv_a = ev._charm_csv(a.m2m, cap)
        nb, csv_b = ev._charm_csv(b.m2m, cap)
        if na != n or nb != n:
            raise RuntimeError(f"{cap}.csv names differ from the MNI cap ({a.m2m} / {b.m2m})")
        _, raw_a = ev._warp(rows, a.m2m, a.skin())
        proj_b, raw_b = ev._warp(rows, b.m2m, skin_b)
        pa.update({f"{c}_unprojected": raw_a, f"{c}_same_skin": csv_a, f"{c}_own_skin": csv_a})
        pb.update({f"{c}_unprojected": raw_b, f"{c}_same_skin": proj_b, f"{c}_own_skin": csv_b})
        names[c] = n
    return pa, pb, names


def eval_pair(a_m2m: Path, b_m2m: Path, device: str, init: str = "affines") -> dict:
    a, b = Head(a_m2m), Head(b_m2m)
    al = align(a, b, device, init)
    pa, pb, names = cap_points(a, b, al["M_ab"])
    pa["targets"], pb["targets"] = targets(a_m2m), targets(b_m2m)
    names["targets"] = list(ev.TARGETS)
    err = rescan.pair_errors(pa, pb, al["M_ab"])
    Minv = np.linalg.inv(al["M_ab"])
    # FOV margin of each point: A's point in A's T1, B's point in B's T1; the pair's margin is the smaller
    margin = {g: np.minimum(a.margin(pa[g]), b.margin(pb[g])) for g in GROUPS}
    # residual: inverse consistency at B's points, and A-vs-A at A's points
    inv_cons = {g: np.linalg.norm(rescan.apply(Minv, pb[g]) - rescan.apply(al["M_ba"], pb[g]), axis=1) for g in GROUPS}
    self_err = {g: np.linalg.norm(rescan.apply(al["M_aa"], pa[g]) - pa[g], axis=1) for g in GROUPS}
    R = al["M_ab"][:3, :3]
    return {"a": a_m2m.name, "b": b_m2m.name, "M_ab": al["M_ab"].tolist(), "M_ba": al["M_ba"].tolist(),
            "M_init": np.asarray(al["M_init"]).tolist(), "init": init,
            "motion_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
            "motion_mm": float(np.linalg.norm(rescan.apply(al["M_ab"], [a.centroid]) - a.centroid)),
            "ncc_init": al["ncc_init"], "ncc": al["ncc"], "ncc_rev": al["ncc_rev"], "seconds": al["seconds"],
            "errors": {g: v.tolist() for g, v in err.items()},
            "fov_margin_mm": {g: v.tolist() for g, v in margin.items()},
            "faces": {"a": a.faces, "b": b.faces}, "truncated": {"a": a.truncated, "b": b.truncated},
            "inverse_consistency": {g: v.tolist() for g, v in inv_cons.items()},
            "self_check": {g: v.tolist() for g, v in self_err.items()},
            "names": names}


# ---------------------------------------------------------------------------------------------------
# ours vs charm on the A images (optional third row)

def labels_t1_grid(m2m: Path, t1) -> tuple[np.ndarray, str]:
    up = nib.load(m2m / dataset.SEG_LABELS)
    try:
        return dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine), "block_majority"
    except ValueError:   # charm's 0.5 mm grid is not 2x the T1 grid (OASIS: 1.25 mm slices): general majority
        return rescan.labels_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine), "majority"


def eval_ours(case: str, charm_m2m: Path, device: str, sub: str = "ours", adopted: bool = False) -> dict:
    out_dir = OUT / sub / case
    out_m2m = out_dir / f"m2m_{case}"
    out_m2m.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(charm_m2m / "T1.nii.gz")
    lab, how = labels_t1_grid(charm_m2m, t1)
    params, tdir = register.adopted_setting() if adopted else (None, register.TEMPLATE_DIR)
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), out_m2m, device=device, params=params,
                            template_dir=tdir)
    shutil.copyfile(charm_m2m / "T1.nii.gz", out_m2m / "T1.nii.gz")
    ours, charm = targets(out_m2m), targets(charm_m2m)
    proj = ev.eval_projected(out_dir, charm_m2m, OUT / "ours_label_skin" / f"{case}.npz")
    g = proj["groups"]
    err = {"targets": np.linalg.norm(ours - charm, axis=1).tolist()}
    margin = {"targets": ev._fov_margin_mm(t1, charm).tolist()}
    for c, pg in (("eeg", "eeg"), ("fid", "fiducials")):
        err[f"{c}_unprojected"] = g[pg]["unprojected"]
        err[f"{c}_same_skin"] = g[pg]["proj_charm_skin"]
        margin[f"{c}_unprojected"] = g[pg]["fov_margin_unprojected_mm"]
        margin[f"{c}_same_skin"] = g[pg]["fov_margin_mm"]
    err["eeg_label_skin"], margin["eeg_label_skin"] = g["eeg"]["proj_label_skin"], g["eeg"]["fov_margin_mm"]
    return {"case": case, "labels": how, "gpu_s": res["timings"]["gpu_total"], "peak_mem_gb": res["peak_mem_gb"],
            "errors": err, "fov_margin_mm": margin, "errors_fid_label_skin": g["fiducials"]["proj_label_skin"]}


# ---------------------------------------------------------------------------------------------------
# synthetic pair (smoke test)

def make_synthetic(src: Path, dst: Path, T: np.ndarray, shift_vox=(0.4, -0.3, 0.5)) -> Path:
    """dst = src seen through world_B = T world_A: T1 resampled (cubic) on a grid shifted by shift_vox,
    labels and Conform2MNI with T applied to their affines, MNI2Conform's values and the csv positions
    mapped by T, worldToWorld = T W. charm_log.html and the mesh are copied so the pair counts as done."""
    if dst.exists():
        shutil.rmtree(dst)
    for d in ("label_prep", "toMNI", "segmentation", "eeg_positions"):
        (dst / d).mkdir(parents=True, exist_ok=True)
    t1 = nib.load(src / "T1.nii.gz")
    S = np.eye(4)
    S[:3, 3] = shift_vox
    aff_b = T @ t1.affine @ S
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in t1.shape[:3]], indexing="ij"), -1).reshape(-1, 3)
    vox_a = rescan.apply(np.linalg.inv(t1.affine) @ np.linalg.inv(T) @ aff_b, ijk)
    data = ndi.map_coordinates(dataset._vol(t1).astype(np.float32), vox_a.T, order=3).reshape(t1.shape[:3])
    nib.save(nib.Nifti1Image(data.astype(np.float32), aff_b), dst / "T1.nii.gz")
    up = nib.load(src / dataset.SEG_LABELS)
    nib.save(nib.Nifti1Image(np.asarray(up.dataobj), T @ up.affine, up.header), dst / dataset.SEG_LABELS)
    c2m = nib.load(src / "toMNI/Conform2MNI_nonl.nii.gz")
    nib.save(nib.Nifti1Image(np.asarray(c2m.dataobj), T @ c2m.affine, c2m.header), dst / "toMNI/Conform2MNI_nonl.nii.gz")
    m2c = nib.load(src / "toMNI/MNI2Conform_nonl.nii.gz")
    v = np.asarray(m2c.dataobj, np.float32)
    v = (v.reshape(-1, 3) @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)).reshape(v.shape)
    nib.save(nib.Nifti1Image(v, m2c.affine, m2c.header), dst / "toMNI/MNI2Conform_nonl.nii.gz")
    mat = scipy.io.loadmat(src / "segmentation/coregistrationMatrices.mat")
    scipy.io.savemat(dst / "segmentation/coregistrationMatrices.mat",
                     {"worldToWorldTransformMatrix": T @ mat["worldToWorldTransformMatrix"],
                      "imageToImageTransformMatrix": mat["imageToImageTransformMatrix"]})
    for cap in CAPS.values():
        types, names, p = read_csv(src / "eeg_positions" / f"{cap}.csv")
        q = rescan.apply(T, p)
        with open(dst / "eeg_positions" / f"{cap}.csv", "w") as fh:
            for t, n, x in zip(types, names, q):
                fh.write(f"{t},{float(x[0])!r},{float(x[1])!r},{float(x[2])!r},{n}\n")
    shutil.copyfile(src / "charm_log.html", dst / "charm_log.html")
    (dst / "synthetic.msh").write_text("placeholder: a synthetic pair has no mesh of its own\n")
    (dst / "synthetic.json").write_text(json.dumps({"source": str(src), "T": T.tolist(), "shift_vox": list(shift_vox)}))
    return dst


def main_synthetic(src: Path, device: str) -> None:
    from headmodel import splits

    held = set(splits.wp3_test_cases()) | set(splits.val_cases())
    hit = [p for p in (src.name, src.parent.name) if p in held]
    if hit:
        raise SystemExit(f"{src}: {hit[0]} is a WP3 test or validation case; use a training case (e.g. 1BA076)")
    T = rescan.rigid([0.08, -0.05, 0.12], [6.0, -9.0, 4.0], centre=[0, 0, 0])   # ~8.5 deg, 11.5 mm
    dst = make_synthetic(src, OUT / "synthetic" / "m2m_synthetic_B", T)
    r = eval_pair(src, dst, device, init="identity")   # the fit has to find T itself
    Mtrue = np.asarray(T)
    pa = {"targets": targets(src)}
    tre = {g: np.linalg.norm(rescan.apply(np.asarray(r["M_ab"]), pa[g]) - rescan.apply(Mtrue, pa[g]), axis=1)
           for g in pa}
    r["registration_error_vs_truth"] = {g: v.tolist() for g, v in tre.items()}
    r["source_case"] = src.parent.name if src.name == "m2m_ref" else src.name
    r["init_error_vs_truth_targets_max"] = float(np.linalg.norm(
        rescan.apply(np.asarray(r["M_init"]), pa["targets"]) - rescan.apply(Mtrue, pa["targets"]), axis=1).max())
    (OUT / "synthetic" / "pair.json").write_text(json.dumps(r, indent=1))
    print(f"synthetic pair from {src} (T: 8.5 deg, 11.5 mm; T1 resampled cubic on a shifted grid; fits start at "
          f"the identity, {r['init_error_vs_truth_targets_max']:.1f} mm from the truth at the targets):")
    print(f"  rigid fit vs truth at the targets: max {tre['targets'].max():.3f} mm")
    for g in GROUPS:
        e = np.asarray(r["errors"][g])
        print(f"  {g}: pair error mean {e.mean():.3f} max {e.max():.3f} mm | inverse consistency max "
              f"{np.max(r['inverse_consistency'][g]):.3f} | A-vs-A max {np.max(r['self_check'][g]):.3f}")
    print(f"  NCC {r['ncc_init']:.4f} -> {r['ncc']:.4f}, {r['seconds']:.0f} s for the 3 rigid fits")


# ---------------------------------------------------------------------------------------------------

def with_rtms(r: dict) -> dict:
    """A copy of a pair / ours row with the rTMS-set EEG errors (and FOV margins) added from the whole cap."""
    names = ev.eeg_cap()[0]
    idx = [names.index(n) for n in ev.RTMS_EEG]
    err, margin = dict(r["errors"]), dict(r.get("fov_margin_mm", {}))
    for v in ("unprojected", "same_skin"):
        if f"eeg_{v}" in err:
            e = np.asarray(err[f"eeg_{v}"], float)
            if len(e) != len(names):   # not the whole cap (a test row): no rTMS rows
                continue
            err[f"eeg_rtms_{v}"] = e[idx].tolist()
            if f"eeg_{v}" in margin:
                margin[f"eeg_rtms_{v}"] = np.asarray(margin[f"eeg_{v}"], float)[idx].tolist()
    return {**r, "errors": err, "fov_margin_mm": margin}


def summary(rows: list[dict], min_margin: float | None = None) -> dict:
    """rescan.summarise per metric over the rows that have it; with min_margin, only the points whose
    FOV margin (smaller of A and B) is >= min_margin mm (pairs left without points drop out of that metric)."""
    rows = [with_rtms(r) for r in rows]
    s = {"n_pairs": len(rows)}
    for g in [g for g in list(GROUPS) + list(EXTRA) if any(g in r["errors"] for r in rows)]:
        sub = []
        for r in rows:
            if g not in r["errors"]:
                continue
            e = np.asarray(r["errors"][g], float)
            if min_margin is not None:
                e = e[np.asarray(r["fov_margin_mm"][g], float) >= min_margin]
            if len(e):
                sub.append({"errors": {g: e}})
        if sub:
            s[g] = rescan.summarise(sub, [g])[g] | {"n_pairs": len(sub)}
    return s


COMPARISONS = (("same_session", "charm vs charm, same session (MR1 mpr-1 vs mpr-2)"),
               ("different_day", "charm vs charm, different day (MR1 vs MR2 mpr-1)"),
               ("ours_vs_charm", "ours (headmodel.register defaults) vs charm, A images"),
               ("ours_r4_vs_charm", "ours (WP3b round 4, adopted) vs charm, A images"))
# --ours / --ours-r4: (summary key, output folder under runs/rescan, register setting)
OURS = {"defaults": ("ours_vs_charm", "ours", False), "r4": ("ours_r4_vs_charm", "ours_r4", True)}


def _table(s: dict, key: str) -> list[str]:
    L = ["| comparison | metric | pairs | mean | SD | worst pair mean | worst point | p95 | points > 5 mm | pairs passing |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for kind, label in COMPARISONS:
        sk = s.get(kind, {}).get(key, {"n_pairs": 0})
        if not sk.get("n_pairs"):
            L.append(f"| {label} | - | 0 | | | | | | | |")
            continue
        for g in [g for g in list(GROUPS) + list(EXTRA) if g in sk]:
            a = sk[g]
            L.append(f"| {label} | {LABELS[g]} | {a['n_pairs']} | {a['mean']:.2f} | {a['sd']:.2f} | "
                     f"{a['max_pair_mean']:.2f} | {a['max']:.2f} | {a['p95']:.2f} | {a['points_over_max']}/{a['n_points']} | "
                     f"{a['pairs_pass']}/{a['n_pairs']} |")
    return L


def markdown(s: dict) -> str:
    L = ["# charm scan-rescan floor on the WP3 metrics (OASIS-1 reliability subset)", "",
         "Distances (mm) between charm's outputs for two scans of the same head, B mapped into A's space by a "
         "6-dof NCC fit of the T1s inside A's charm brain mask. Metrics: targets = 10 MNI targets via "
         "mni2subject_coords; EEG / fiducials *unprojected* = the MNI cap warped by each scan's MNI2Conform "
         "(the eval_register gate metric); *same skin* = B's warped cap projected on A's charm skin vs A's csv "
         "(as WP3's proj_charm_skin, where ours is projected on charm's skin of the same image); *own skin* = "
         "each scan's eeg_positions csv as delivered (adds skin-segmentation variability; no WP3 analogue). "
         "Per pair mean and max; then mean of pair means / SD / worst pair mean / worst point / p95 of points / "
         "pairs passing (mean <= 2, max <= 5 mm).", "", "## All points", ""]
    L += _table(s, "all")
    L += ["", f"## Points at least {FOV_EDGE_MM:.0f} mm inside the T1 field of view in both A and B", "",
          "The OASIS-1 T1s are 160 mm left-right and face-masked: in about half of them the scalp is cut at the "
          "lateral faces, so charm's skin (and the projected T7/T8/TP9/TP10, sometimes Cz) depends on where the "
          "cut falls. These rows drop points within the margin (pairs without points left drop out).", ""]
    L += _table(s, "inside")
    L += ["", "Rigid-step residual (max over pairs of the per pair max, mm): "]
    for kind in KINDS:
        r = s.get(f"{kind}_residual")
        if r:
            L.append(f"- {kind}: inverse consistency " + ", ".join(f"{g} {r['inverse_consistency'][g]:.3f}" for g in GROUPS)
                     + "; A-vs-A from a perturbed start (identical images, convergence only) "
                     + ", ".join(f"{g} {r['self_check'][g]:.3f}" for g in GROUPS)
                     + f"; head motion between scans median {r['motion_deg_median']:.1f} deg / {r['motion_mm_median']:.1f} mm")
    if s.get("synthetic"):
        y = s["synthetic"]
        L += ["", f"Synthetic check ({y['source']}, a training case; known 8.5 deg / 11.5 mm motion, resampled T1, "
              f"rigid fits started at the identity, {y['init_error']:.1f} mm from the truth): rigid fit vs truth max "
              f"{y['vs_truth']:.3f} mm; pair error max " + ", ".join(f"{g} {y['max'][g]:.3f}" for g in y["max"]) + " mm."]
    imgs = s.get("images", {})
    if imgs:
        t1 = [k for k, v in imgs.items() if v.get("truncated_t1")]
        ch = [k for k, v in imgs.items() if v.get("truncated")]
        L += ["", f"## T1 field of view per image", "",
              f"Truncated = more than {TRUNC_VOXELS} head voxels on a face of the T1 box other than inferior: "
              f"{len(t1)}/{len(imgs)} images by the T1 intensity foreground, {len(ch)}/"
              f"{sum('faces' in v for v in imgs.values())} scored images by charm's labels (> 0).", "",
              "| image | truncated (T1) | T1 head voxels R / L / A / P / S / I | truncated (charm) | charm head voxels R / L / A / P / S / I |",
              "|---|---|---|---|---|"]
        fmt = lambda fc: " / ".join(str(fc.get(x, "")) for x in "RLAPSI") if fc else "-"   # noqa: E731
        for k, v in sorted(imgs.items()):
            L.append(f"| {k} | {', '.join(v.get('truncated_t1', [])) or '-'} | {fmt(v.get('faces_t1'))} | "
                     f"{', '.join(v.get('truncated', [])) or '-' if 'faces' in v else 'n/a'} | {fmt(v.get('faces'))} |")
    if s.get("pending"):
        L += ["", f"Pairs not yet scored (charm unfinished): {len(s['pending'])}."]
    if s.get("failed"):
        L += ["", f"Pairs blocked by a FAILED charm run ({len(s['failed'])}): "
              + "; ".join(f"{p} ({', '.join(c)})" for p, c in s["failed"].items()) + "."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--ours", action="store_true", help="also our registration vs charm on the A images (GPU)")
    ap.add_argument("--ours-r4", action="store_true", help="the same with the adopted WP3b round-4 setting "
                                                          "(register.adopted_setting; rule wp3b_r4_adopt's one rescan use)")
    ap.add_argument("--synthetic", type=Path, default=None, help="smoke test on a synthetic pair from this m2m")
    args = ap.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        return main_synthetic(args.synthetic.expanduser(), args.device)
    cases = json.loads((OUT / "cases.json").read_text())
    events = queue_events()
    rows = {k: [] for k in KINDS}
    pending, failed = [], {}
    images = {c: {"faces_t1": v["faces"], "truncated_t1": v["truncated"]}
              for c, v in nifti_faces(cases["A"] + cases["B"] + cases["C"]).items()}
    for i, a_id in enumerate(cases["A"]):
        for kind, role in KINDS.items():
            b_id = cases[role][i]
            if a_id.split("_MR")[0] != b_id.split("_MR")[0]:
                raise RuntimeError(f"{a_id} vs {b_id}: not the same subject")
            f = OUT / "pairs" / f"{kind}_{a_id}__{b_id}.json"
            if f.is_file() and not args.redo:
                r = json.loads(f.read_text())
                if "fov_margin_mm" in r:   # pairs scored before the metric variants: re-scored below
                    rows[kind].append(r)
                    for c, k in ((a_id, "a"), (b_id, "b")):
                        images.setdefault(c, {}).update(faces=r["faces"][k], truncated=r["truncated"][k])
                    continue
            st = {c: status(c, events) for c in (a_id, b_id)}
            if any(v == "failed" for v in st.values()):
                failed[f.stem] = [c for c, v in st.items() if v == "failed"]
                continue
            if any(v != "done" for v in st.values()):
                pending.append(f.stem)
                continue
            a_m2m, b_m2m = ROOT / a_id / f"m2m_{a_id}", ROOT / b_id / f"m2m_{b_id}"
            r = eval_pair(a_m2m, b_m2m, args.device)
            r["kind"] = kind
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(r))
            rows[kind].append(r)
            for c, k in ((a_id, "a"), (b_id, "b")):
                images.setdefault(c, {}).update(faces=r["faces"][k], truncated=r["truncated"][k])
            print(f"{kind} {a_id} vs {b_id}: " + "  ".join(
                f"{g} {np.mean(r['errors'][g]):.2f}/{np.max(r['errors'][g]):.2f}" for g in GROUPS)
                + f"  motion {r['motion_deg']:.1f} deg  {r['seconds']:.0f} s", flush=True)
    s = {"pending": pending, "failed": failed, "images": images, "groups": list(GROUPS), "fov_edge_mm": FOV_EDGE_MM,
         "gate": {"mean": rescan.GATE_MEAN, "max": rescan.GATE_MAX}}
    for kind in KINDS:
        s[kind] = {"all": summary(rows[kind]), "inside": summary(rows[kind], FOV_EDGE_MM)}
        if rows[kind]:
            s[f"{kind}_residual"] = {
                "inverse_consistency": {g: float(max(np.max(r["inverse_consistency"][g]) for r in rows[kind])) for g in GROUPS},
                "self_check": {g: float(max(np.max(r["self_check"][g]) for r in rows[kind])) for g in GROUPS},
                "motion_deg_median": float(np.median([r["motion_deg"] for r in rows[kind]])),
                "motion_mm_median": float(np.median([r["motion_mm"] for r in rows[kind]]))}
    all_ours = {}
    for which, (key, sub, adopted) in OURS.items():
        ours = []
        if (args.ours and which == "defaults") or (args.ours_r4 and which == "r4"):
            for a_id in cases["A"]:
                f = OUT / sub / f"{a_id}.json"
                if f.is_file() and not args.redo:
                    r = json.loads(f.read_text())
                    if "fov_margin_mm" in r:
                        ours.append(r)
                        continue
                m2m = ROOT / a_id / f"m2m_{a_id}"
                if status(a_id, events) != "done":
                    continue
                r = eval_ours(a_id, m2m, args.device, sub, adopted)
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(r))
                ours.append(r)
                print(f"{sub} {a_id}: " + "  ".join(f"{g} {np.mean(e):.2f}/{np.max(e):.2f}" for g, e in r["errors"].items()),
                      flush=True)
        else:
            ours = [r for r in (json.loads(f.read_text()) for f in sorted((OUT / sub).glob("OAS1_*.json")))
                    if "fov_margin_mm" in r]
        if ours:
            s[key] = {"all": summary(ours), "inside": summary(ours, FOV_EDGE_MM)}
        all_ours[which] = ours
    ours = all_ours["defaults"]
    syn = OUT / "synthetic" / "pair.json"
    if syn.is_file():
        y = json.loads(syn.read_text())
        if "registration_error_vs_truth" in y and y.get("init") == "identity":
            s["synthetic"] = {"source": y.get("source_case", y["a"]), "max": {g: float(np.max(y["errors"][g])) for g in GROUPS},
                              "vs_truth": float(np.max(y["registration_error_vs_truth"]["targets"])),
                              "init_error": y["init_error_vs_truth_targets_max"]}
    (OUT / "eval.json").write_text(json.dumps({"summary": s, "pairs": rows, "ours": ours, "ours_r4": all_ours["r4"]}))
    md = markdown(s)
    (OUT / "eval.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
