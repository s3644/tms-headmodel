"""WP3 evaluation: our GPU registration vs charm's warps on the held-out (test) cases.

Input per case: charm's own labels (label_prep/tissue_labeling_upsampled, block-majority to the T1
grid), so that the registration is judged alone. Our toMNI and coregistrationMatrices.mat are written
to runs/wp3/<case>/m2m_<case>/ (with a copy of T1.nii.gz). Both folders are then read through
simnibs.mni2subject_coords for the 10 MNI targets (ct-headmodel validate.py) and all positions of the
EEG10-10_UI_Jurak_2007 cap (warped coordinates, before any skin projection). Affine-only numbers use
the affine part of our map (the 12-dof frame the SVF starts from), applied directly.

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/eval_register.py [--n N] [--cases A B ...]
Writes runs/wp3/<case>/eval.json (resumable), runs/wp3/eval.json and runs/wp3/eval.md, always with the WP3
parameters (WP3_PARAMS, WP3_TEMPLATE_DIR) on the frozen 57 WP3 test cases (headmodel.splits.wp3_test_cases).

--projected (CPU, no registration): what the contract delivers. eeg_positions/*.csv are the MNI caps
warped by mni2subject AND projected on the skin (warp_coordinates, skin tag 1005). Re-scores the
existing runs/wp3/<case>/m2m_<case> warps that way (eval_projected) and writes
runs/wp3b/eval_projected.json and .md (per case rows resumable in runs/wp3b/projected/<case>.json).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import dataset, register, splits  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3"
GATE_MEAN, GATE_MAX = 2.0, 5.0
# runs/wp3 holds the WP3 result: its parameters and template, not the WP3b defaults (scripts/wp3b_eval_test.py)
WP3_PARAMS = {"frame": "labels", "svf_class_weights": None}
WP3_TEMPLATE_DIR = register.WP3_TEMPLATE_DIR
# affine_raw: our 12-dof fit alone; affine: its calibrated version (the frame the SVF starts from,
# written as worldToWorldTransformMatrix); nonlinear: the full map
KINDS = ("affine_raw", "affine", "nonlinear")

# ct-headmodel validate.py PRESETS (MNI152 mm): left targets, right = x-mirror
_LEFT = {"dlpfc_sgacc": (-38, 44, 26), "dlpfc_5cm": (-41, 16, 54), "dlpfc_ba46": (-44, 40, 29),
         "dlpfc_ba9": (-36, 39, 43), "m1_hand": (-37, -21, 58)}
TARGETS = {f"{k}_{s}": (x if s == "l" else -x, y, z) for k, (x, y, z) in _LEFT.items() for s in "lr"}


def eeg_cap() -> tuple[list[str], np.ndarray]:
    names, pts = [], []
    with open(register.simnibs_dir() / "resources/ElectrodeCaps_MNI/EEG10-10_UI_Jurak_2007.csv") as fh:
        for row in csv.reader(fh):
            if len(row) >= 5:
                names.append(row[4])
                pts.append([float(v) for v in row[1:4]])
    return names, np.array(pts)


def load_manifest() -> dict:
    f = REPO / "data/manifests/round1.json"
    return json.loads(f.read_text()) if f.is_file() else dataset.manifest(CT_ROOT)


def stats(err: np.ndarray) -> dict:
    return {"mean": float(err.mean()), "max": float(err.max())}


def eval_case(case: str, info: dict, targets: np.ndarray, eeg: np.ndarray, device: str) -> dict:
    import simnibs

    ref = Path(info["m2m"])
    out_m2m = OUT / case / f"m2m_{case}"
    out_m2m.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(ref / "T1.nii.gz")
    up = nib.load(ref / dataset.SEG_LABELS)
    t0 = time.perf_counter()
    lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    t_lab = time.perf_counter() - t0
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), out_m2m, device=device,
                            params=register.Params(**WP3_PARAMS), template_dir=WP3_TEMPLATE_DIR)
    shutil.copyfile(ref / "T1.nii.gz", out_m2m / "T1.nii.gz")
    pts = np.vstack([targets, eeg])
    ours = simnibs.mni2subject_coords(pts, str(out_m2m))
    charm = simnibs.mni2subject_coords(pts, str(ref))
    A = np.asarray(res["affine"])
    aff = nib.affines.apply_affine(A, pts)
    ear = np.linalg.norm(nib.affines.apply_affine(np.asarray(res["affine_raw"]), pts) - charm, axis=1)
    back = simnibs.subject2mni_coords(ours, str(out_m2m))
    # Conform2MNI lives on the T1 grid (as charm's), so the round trip is defined inside its field of view
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), ours)
    inside = ((ijk >= 0) & (ijk <= np.array(t1.shape[:3]) - 1)).all(1)
    e, ea = np.linalg.norm(ours - charm, axis=1), np.linalg.norm(aff - charm, axis=1)
    nt = len(targets)
    return {
        "case": case, "dataset": info.get("dataset"),
        "nonlinear": {"targets": stats(e[:nt]), "eeg": stats(e[nt:])},
        "affine": {"targets": stats(ea[:nt]), "eeg": stats(ea[nt:])},
        "affine_raw": {"targets": stats(ear[:nt]), "eeg": stats(ear[nt:])},
        "A": res["affine"], "A_raw": res["affine_raw"],
        "round_trip_max_mm": float(np.linalg.norm(back - pts, axis=1)[inside].max()),
        "points_outside_t1_fov": int((~inside).sum()),
        "err_targets": e[:nt].tolist(), "err_eeg": e[nt:].tolist(),
        "gpu_s": res["timings"]["gpu_total"], "write_s": res["timings"].get("write"), "labels_s": t_lab,
        "peak_mem_gb": res["peak_mem_gb"], "timings": res["timings"], "losses": res["losses"],
    }


WARP_FILES = ("MNI2Conform_nonl", "Conform2MNI_nonl")


def map_jacobian(img) -> np.ndarray:
    """Jacobian determinant of a written coordinate map (X, Y, Z, 3 world coordinates per voxel) with
    respect to the world coordinates of its own grid: central finite differences over voxels
    (np.gradient, one-sided at the border) times the inverse of the grid's 3x3 affine. > 0 everywhere:
    no folding."""
    data = np.asarray(img.dataobj, np.float32)
    Minv = np.linalg.inv(img.affine[:3, :3]).astype(np.float32)
    G = np.stack([np.stack(np.gradient(data[..., i]), -1) for i in range(3)], -2)   # d value_i / d ijk_j
    del data
    J = G @ Minv
    del G
    return np.linalg.det(J)


def written_jacobian(m2m: Path) -> dict:
    """min Jacobian and the number of voxels <= 0 of each written toMNI map (not the SVF's velocity grid)."""
    out = {}
    for name in WARP_FILES:
        d = map_jacobian(nib.load(Path(m2m) / "toMNI" / f"{name}.nii.gz"))
        out[name] = {"min": float(d.min()), "n_le0": int((d <= 0).sum())}
    out["min"] = min(out[n]["min"] for n in WARP_FILES)
    return out


def backfill_affine_raw(r: dict, info: dict, targets: np.ndarray, eeg: np.ndarray, device: str) -> dict:
    """Affine-only numbers for a case evaluated before they were recorded: the same affine stage
    (deterministic up to GPU float order), uncalibrated, no SVF."""
    import simnibs

    ref = Path(info["m2m"])
    t1, up = nib.load(ref / "T1.nii.gz"), nib.load(ref / dataset.SEG_LABELS)
    lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), None, device=device, write=False,
                            params=register.Params(**{**WP3_PARAMS, "svf_stages": ()}), template_dir=WP3_TEMPLATE_DIR)
    pts = np.vstack([targets, eeg])
    charm = simnibs.mni2subject_coords(pts, str(ref))
    ear = np.linalg.norm(nib.affines.apply_affine(np.asarray(res["affine_raw"]), pts) - charm, axis=1)
    nt = len(targets)
    r["affine_raw"] = {"targets": stats(ear[:nt]), "eeg": stats(ear[nt:])}
    r["A_raw"], r["A_calibrated_check"] = res["affine_raw"], res["affine"]
    print(f"{r['case']}: affine_raw targets {ear[:nt].mean():.2f} EEG {ear[nt:].mean():.2f}", flush=True)
    return r


def summarise(rows: list[dict]) -> dict:
    def agg(kind, what):
        means = np.array([r[kind][what]["mean"] for r in rows])
        maxs = np.array([r[kind][what]["max"] for r in rows])
        return {"mean": float(means.mean()), "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "max_of_case_means": float(means.max()), "max": float(maxs.max()),
                "cases_pass": int(((means <= GATE_MEAN) & (maxs <= GATE_MAX)).sum())}
    s = {k: {w: agg(k, w) for w in ("targets", "eeg")} for k in KINDS}
    s["n_cases"] = len(rows)
    s["gpu_s"] = {"mean": float(np.mean([r["gpu_s"] for r in rows])), "max": float(np.max([r["gpu_s"] for r in rows]))}
    s["peak_mem_gb"] = float(np.max([r["peak_mem_gb"] for r in rows]))
    s["round_trip_max_mm"] = float(np.max([r["round_trip_max_mm"] for r in rows]))
    s["points_outside_t1_fov"] = int(np.sum([r["points_outside_t1_fov"] for r in rows]))
    nl = s["nonlinear"]
    s["gate"] = {w: bool(nl[w]["mean"] <= GATE_MEAN and nl[w]["max"] <= GATE_MAX) for w in ("targets", "eeg")}
    return s


def markdown(s: dict, rows: list[dict], n_eeg: int) -> str:
    lines = [f"# WP3 registration vs charm ({s['n_cases']} held-out cases)", "",
             f"Distances (mm) between our and charm's `mni2subject_coords`: 10 MNI targets and {n_eeg} "
             "EEG10-10_UI_Jurak_2007 positions (no skin projection). Per case: mean and max over points; "
             "then mean / SD / max over cases. Gate: <= 2 mm mean, <= 5 mm worst.", "",
             "| warp | points | mean of case means | SD | worst case mean | worst point | cases passing |",
             "|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for w in ("targets", "eeg"):
            a = s[k][w]
            lines.append(f"| {k} | {w} | {a['mean']:.2f} | {a['sd']:.2f} | {a['max_of_case_means']:.2f} | "
                         f"{a['max']:.2f} | {a['cases_pass']}/{s['n_cases']} |")
    lines += ["", f"GPU time per case: mean {s['gpu_s']['mean']:.1f} s, max {s['gpu_s']['max']:.1f} s; "
              f"peak GPU memory {s['peak_mem_gb']:.2f} GB; round trip subject2mni(mni2subject(x)) max "
              f"{s['round_trip_max_mm']:.2f} mm (points inside the T1 field of view; "
              f"{s['points_outside_t1_fov']} case-points fall outside it, e.g. Cz above a cropped T1).",
              f"Gate (nonlinear): targets {'PASS' if s['gate']['targets'] else 'FAIL'}, "
              f"EEG {'PASS' if s['gate']['eeg'] else 'FAIL'}.", "",
              "| case | dataset | targets mean / max | EEG mean / max | affine targets / EEG mean | GPU s |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        nl, af = r["nonlinear"], r["affine"]
        lines.append(f"| {r['case']} | {(r['dataset'] or '').split(' (')[0]} | {nl['targets']['mean']:.2f} / "
                     f"{nl['targets']['max']:.2f} | {nl['eeg']['mean']:.2f} / {nl['eeg']['max']:.2f} | "
                     f"{af['targets']['mean']:.2f} / {af['eeg']['mean']:.2f} | {r['gpu_s']:.1f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------
# Projected EEG positions (the contract's eeg_positions/*.csv)

OUT_B = REPO / "runs/wp3b"
# The "on our skin" rows use the closed skin surface of our navigation app (not part of this repository): set
# HM_LOCATOR_BACKEND to a folder with an app/locator_ready.py providing skin_surface() and skin_msh().
LOCATOR_BACKEND = Path(os.environ.get("HM_LOCATOR_BACKEND", "/nonexistent/locator_backend"))
SKIN_TAG = 1005
CAPS = {"eeg": "EEG10-10_UI_Jurak_2007", "fiducials": "Fiducials"}
# the EEG positions clinical rTMS uses (WP3 gate scope from round 4 on, sealed rule wp3_gate_rtms_set; locator
# docs/TMS_PROTOCOLS.md); the whole cap is still reported
RTMS_EEG = ("F3", "F4", "Fz", "Fpz", "Fp1", "Fp2", "Cz", "C3", "C4", "FCz", "T7", "T8", "Oz")
# per point errors (mm) recorded by eval_projected
CONDITIONS = {
    "unprojected": "our vs charm's warp, no projection (as runs/wp3/eval.md)",
    "unprojected_tangential": "unprojected error minus its component along charm's skin normal",
    "proj_charm_skin": "our warp projected on charm's mesh skin vs charm's csv (registration alone)",
    "proj_label_skin": "our warp projected on the closed label skin vs charm's csv (locator-ready path)",
    "floor_label_skin": "charm's warp projected on the label skin vs charm's csv (floor of the above)",
    "check_charm_skin": "charm's warp re-projected on charm's mesh skin vs charm's csv (sanity, ~0)",
}


def _cap_rows(cap: str) -> tuple:
    """warp_coordinates' list input for one MNI cap: its electrode rows, then the same points as
    Generic rows (warped, never projected), so one call gives both."""
    from simnibs.utils.csv_reader import read_csv_positions

    type_, coords, extra, name, extra_cols, header = read_csv_positions(
        str(register.simnibs_dir() / f"resources/ElectrodeCaps_MNI/{cap}.csv"))
    n = len(type_)
    coords = np.asarray(coords, float)
    return (list(type_) + ["Generic"] * n, np.vstack([coords, coords]), list(extra) + [None] * n,
            list(name) + list(name), list(extra_cols) + [None] * n, header)


def _warp(rows: tuple, m2m: Path, mesh) -> tuple[np.ndarray, np.ndarray]:
    """(projected, unprojected) subject coordinates: charm's mesh-step call (warp_coordinates,
    mni2subject, skin tag 1005) with `mesh` as mesh_in."""
    import warnings

    from simnibs.utils import transformations

    with warnings.catch_warnings():   # our m2m-like folder has no head mesh: mesh_in replaces it
        warnings.filterwarnings("ignore", message="Could not find")
        _, xyz, _, _ = transformations.warp_coordinates(rows, str(m2m), transformation_direction="mni2subject",
                                                        mesh_in=mesh, skin_tag=SKIN_TAG)
    n = len(xyz) // 2
    return np.asarray(xyz[:n]), np.asarray(xyz[n:])


def _charm_csv(m2m: Path, cap: str) -> tuple[list[str], np.ndarray]:
    from simnibs.utils.csv_reader import read_csv_positions

    _, coords, _, name, _, _ = read_csv_positions(str(m2m / "eeg_positions" / f"{cap}.csv"))
    return list(name), np.asarray(coords, float)


def charm_skin(charm_m2m: Path):
    """charm's skin triangles (tag 1005) cropped from <id>.msh, as charm's mesh step passes them."""
    from simnibs.mesh_tools import mesh_io

    msh = next(f for f in sorted(charm_m2m.glob("*.msh")))
    m = mesh_io.read_msh(str(msh))
    idx = m.elm.get_triangles(SKIN_TAG)
    return m.crop_mesh(elements=m.elm.elm_number[idx])


def label_skin(charm_m2m: Path, cache: Path | None = None):
    """The locator-ready skin: closed outer skin of charm's label_prep/tissue_labeling_upsampled
    (rTMS locator's locator_ready.skin_surface), as a tag-1005 Msh (locator_ready.skin_msh)."""
    if str(LOCATOR_BACKEND) not in sys.path:
        sys.path.insert(0, str(LOCATOR_BACKEND))
    from app.locator_ready import skin_msh, skin_surface

    if cache is not None and cache.is_file():
        z = np.load(cache)
        v, f = z["v"], z["f"]
    else:
        img = nib.load(charm_m2m / dataset.SEG_LABELS)
        v, f = skin_surface(np.asarray(img.dataobj), img.affine)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, v=v, f=f)
    return skin_msh(v, f)


def _skin_normals(skin, pts: np.ndarray) -> np.ndarray:
    """Unit normal of the skin triangle closest to each point."""
    import trimesh

    tri = skin.elm.node_number_list[skin.elm.elm_type == 2][:, :3] - 1
    tm = trimesh.Trimesh(skin.nodes.node_coord, tri, process=False)
    _, _, fid = trimesh.proximity.closest_point(tm, pts)
    return np.asarray(tm.face_normals[fid])


def _fov_margin_mm(t1, pts: np.ndarray) -> np.ndarray:
    """Signed distance (mm) of each point to the T1 voxel-centre box, > 0 inside, < 0 outside."""
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), pts)
    hi = np.array(t1.shape[:3]) - 1
    vox = np.sqrt((t1.affine[:3, :3] ** 2).sum(0))
    return (np.minimum(ijk, hi - ijk) * vox).min(1)


def eval_projected(case_out_dir: Path, charm_m2m: Path, label_cache: Path | None = None) -> dict:
    """Score our warps (case_out_dir/m2m_<case>/toMNI) as the contract delivers them: EEG10-10 and
    fiducials warped and projected on the skin, against charm's eeg_positions/*.csv. Returns per
    point errors (mm) for every CONDITIONS entry, per cap group."""
    case_out_dir, charm_m2m = Path(case_out_dir), Path(charm_m2m)
    ours = case_out_dir / f"m2m_{case_out_dir.name}"
    if not (ours / "toMNI/MNI2Conform_nonl.nii.gz").is_file():
        raise FileNotFoundError(ours / "toMNI/MNI2Conform_nonl.nii.gz")
    t0 = time.perf_counter()
    mskin, lskin = charm_skin(charm_m2m), label_skin(charm_m2m, label_cache)
    t1 = nib.load(charm_m2m / "T1.nii.gz")
    out = {"case": case_out_dir.name, "groups": {}}
    for group, cap in CAPS.items():
        rows = _cap_rows(cap)
        names = rows[3][: len(rows[3]) // 2]
        cn, ref = _charm_csv(charm_m2m, cap)
        if cn != names:
            raise RuntimeError(f"{charm_m2m}: {cap}.csv names differ from the MNI cap")
        o_m, o_raw = _warp(rows, ours, mskin)
        o_l, _ = _warp(rows, ours, lskin)
        c_m, c_raw = _warp(rows, charm_m2m, mskin)
        c_l, _ = _warp(rows, charm_m2m, lskin)
        e = o_raw - c_raw
        n = _skin_normals(mskin, ref)
        e_t = e - (e * n).sum(1, keepdims=True) * n
        d = lambda a, b: np.linalg.norm(a - b, axis=1).tolist()   # noqa: E731
        out["groups"][group] = {
            "names": names,
            "unprojected": d(o_raw, c_raw), "unprojected_tangential": np.linalg.norm(e_t, axis=1).tolist(),
            "unprojected_normal": np.abs((e * n).sum(1)).tolist(),
            "proj_charm_skin": d(o_m, ref), "proj_label_skin": d(o_l, ref),
            "floor_label_skin": d(c_l, ref), "check_charm_skin": d(c_m, ref),
            "fov_margin_mm": _fov_margin_mm(t1, ref).tolist(),
            # charm's warped point before projection: can lie outside a cropped T1 (e.g. Cz, Iz)
            "fov_margin_unprojected_mm": _fov_margin_mm(t1, c_raw).tolist(),
            "charm_xyz": ref.tolist(), "ours_xyz_label_skin": o_l.tolist(), "ours_xyz_charm_skin": o_m.tolist(),
        }
    out["seconds"] = time.perf_counter() - t0
    return out


def summarise_projected(rows: list[dict]) -> dict:
    s = {"n_cases": len(rows), "gate": {"mean": GATE_MEAN, "max": GATE_MAX}}
    for group in CAPS:
        s[group] = {}
        for cond in CONDITIONS:
            per = [np.asarray(r["groups"][group][cond]) for r in rows]
            means = np.array([p.mean() for p in per])
            maxs = np.array([p.max() for p in per])
            pts = np.concatenate(per)
            s[group][cond] = {
                "mean": float(means.mean()), "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "max_of_case_means": float(means.max()), "max": float(pts.max()),
                "p95": float(np.percentile(pts, 95)),
                "cases_pass": int(((means <= GATE_MEAN) & (maxs <= GATE_MAX)).sum()),
                "points_over_max": int((pts > GATE_MAX).sum()), "n_points": int(len(pts)),
            }
            s[group][cond]["gate"] = bool(s[group][cond]["mean"] <= GATE_MEAN and s[group][cond]["max"] <= GATE_MAX)
    return s


def worst_points(rows: list[dict], cond: str, n: int = 10) -> list[dict]:
    pts = []
    for r in rows:
        for group, g in r["groups"].items():
            for i, name in enumerate(g["names"]):
                pts.append({"case": r["case"], "group": group, "name": name, "mm": g[cond][i],
                            "unprojected_mm": g["unprojected"][i], "floor_mm": g["floor_label_skin"][i],
                            "fov_margin_mm": g["fov_margin_mm"][i],
                            "fov_margin_unprojected_mm": g["fov_margin_unprojected_mm"][i],
                            "in_t1_fov": g["fov_margin_unprojected_mm"][i] >= 0})
    return sorted(pts, key=lambda p: -p["mm"])[:n]


def _over_gate_lines(rows: list[dict], cond: str) -> list[str]:
    """Which electrodes and cases carry the EEG points over GATE_MAX, mean signed error (scanner RAS)."""
    import collections

    by_name, by_case, vec = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    out_fov = n_out = 0
    for r in rows:
        g = r["groups"]["eeg"]
        for i, name in enumerate(g["names"]):
            outside = g["fov_margin_unprojected_mm"][i] < 0
            n_out += outside
            if g[cond][i] > GATE_MAX:
                by_name[name] += 1
                by_case[r["case"]] += 1
                out_fov += outside
            vec[name].append(np.subtract(g["ours_xyz_label_skin"][i], g["charm_xyz"][i]))
    tot = sum(by_name.values())
    L = [f"{tot} points in {len(by_case)}/{len(rows)} cases; {out_fov} of them have charm's unprojected point outside "
         f"the T1 FOV ({n_out} such points overall).", "",
         "| electrode | cases > 5 mm | mean signed error x / y / z (mm, ours - charm) |", "|---|---|---|"]
    for name, k in by_name.most_common(12):
        m = np.mean(vec[name], 0)
        L.append(f"| {name} | {k} | {m[0]:+.2f} / {m[1]:+.2f} / {m[2]:+.2f} |")
    L += ["", "Cases with most points > 5 mm: " + ", ".join(f"{c} {k}" for c, k in by_case.most_common(12))]
    return L


def markdown_projected(s: dict, rows: list[dict], worst: dict) -> str:
    L = [f"# WP3 EEG / fiducial positions as the contract delivers them ({s['n_cases']} held-out cases)", "",
         "Distances (mm) to charm's `eeg_positions/{EEG10-10_UI_Jurak_2007,Fiducials}.csv`. Our warps are the "
         "unchanged runs/wp3 outputs (charm's labels as input). Projection: simnibs warp_coordinates "
         "(mni2subject, skin tag 1005) with mesh_in = charm's mesh skin, or the closed skin of "
         "charm's label_prep/tissue_labeling_upsampled (rTMS locator locator_ready.skin_surface). "
         f"Gate per case: mean <= {GATE_MEAN:g} and max <= {GATE_MAX:g} mm.", ""]
    L += [f"- `{k}`: {v}" for k, v in CONDITIONS.items()] + [""]
    L += ["| points | condition | mean of case means | SD | worst case mean | worst point | p95 points | "
          "points > 5 mm | cases passing |", "|---|---|---|---|---|---|---|---|---|"]
    for group in CAPS:
        for cond in CONDITIONS:
            a = s[group][cond]
            L.append(f"| {group} | {cond} | {a['mean']:.2f} | {a['sd']:.2f} | {a['max_of_case_means']:.2f} | "
                     f"{a['max']:.2f} | {a['p95']:.2f} | {a['points_over_max']}/{a['n_points']} | "
                     f"{a['cases_pass']}/{s['n_cases']} |")
    for cond, pts in worst.items():
        L += ["", f"10 worst points, `{cond}` (FOV margin: distance of charm's position to the T1 box, "
              "< 0 = outside the T1 field of view):", "",
              "| case | point | mm | unprojected mm | label-skin floor mm | FOV margin mm (csv / unprojected) | "
              "T1 FOV contains charm's point |", "|---|---|---|---|---|---|---|"]
        L += [f"| {p['case']} | {p['name']} | {p['mm']:.2f} | {p['unprojected_mm']:.2f} | {p['floor_mm']:.2f} | "
              f"{p['fov_margin_mm']:.1f} / {p['fov_margin_unprojected_mm']:.1f} | {'yes' if p['in_t1_fov'] else 'no'} |"
              for p in pts]
    L += ["", "Where the > 5 mm EEG points are (`proj_label_skin`):", ""] + _over_gate_lines(rows, "proj_label_skin")
    L += ["", "| case | EEG unproj mean/max | EEG charm-skin mean/max | EEG label-skin mean/max | "
          "EEG floor mean/max | fid charm-skin mean/max | fid label-skin mean/max |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        g, f = r["groups"]["eeg"], r["groups"]["fiducials"]
        mm = lambda x: f"{np.mean(x):.2f} / {np.max(x):.2f}"   # noqa: E731
        L.append(f"| {r['case']} | {mm(g['unprojected'])} | {mm(g['proj_charm_skin'])} | {mm(g['proj_label_skin'])} | "
                 f"{mm(g['floor_label_skin'])} | {mm(f['proj_charm_skin'])} | {mm(f['proj_label_skin'])} |")
    return "\n".join(L) + "\n"


def main_projected(cases: list[str] | None, redo: bool = False) -> None:
    man = load_manifest()
    test = splits.wp3_test_cases()   # the 57 WP3 test cases (frozen)
    if cases:
        test = [c for c in cases if c in test]
    rows = []
    for case in test:
        f = OUT_B / "projected" / f"{case}.json"
        if f.is_file() and not redo:
            rows.append(json.loads(f.read_text()))
            continue
        r = eval_projected(OUT / case, Path(man["cases"][case]["m2m"]), OUT_B / "label_skin" / f"{case}.npz")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(r))
        rows.append(r)
        g = r["groups"]["eeg"]
        print(f"{case}: EEG unproj {np.mean(g['unprojected']):.2f}  charm-skin {np.mean(g['proj_charm_skin']):.2f}"
              f"/{np.max(g['proj_charm_skin']):.2f}  label-skin {np.mean(g['proj_label_skin']):.2f}/"
              f"{np.max(g['proj_label_skin']):.2f}  floor {np.mean(g['floor_label_skin']):.2f}  "
              f"check {np.max(g['check_charm_skin']):.3f}  {r['seconds']:.0f} s", flush=True)
    s = summarise_projected(rows)
    worst = {c: worst_points(rows, c) for c in ("proj_charm_skin", "proj_label_skin", "unprojected")}
    OUT_B.mkdir(parents=True, exist_ok=True)
    (OUT_B / "eval_projected.json").write_text(json.dumps({"summary": s, "worst": worst, "cases": rows}))
    md = markdown_projected(s, rows, worst)
    (OUT_B / "eval_projected.md").write_text(md)
    print(md.split("\n\n| case | EEG")[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=0, help="first N test cases (0: all)")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--projected", action="store_true", help="score the existing warps as projected "
                    "EEG positions (CPU only, writes runs/wp3b/eval_projected.*)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.projected:
        return main_projected(args.cases, args.redo)
    man = load_manifest()
    test = splits.wp3_test_cases()   # the 57 WP3 test cases (frozen; later manifest test cases are not WP3's)
    assert all(man["cases"][c]["split"] == "test" for c in test)
    if args.cases:
        test = [c for c in args.cases if c in test]
    if args.n:
        test = test[: args.n]
    tpl = json.loads((WP3_TEMPLATE_DIR / "template.json").read_text())
    leak = {r["case"] for r in tpl["cases"]} & set(test)
    if leak:
        raise SystemExit(f"template was built with test cases: {sorted(leak)}")
    targets = np.array(list(TARGETS.values()), float)
    names, eeg = eeg_cap()
    rows = []
    for case in test:
        f = OUT / case / "eval.json"
        if f.is_file() and not args.redo:
            r = json.loads(f.read_text())
            if "affine_raw" not in r:   # rows from before affine_raw was recorded
                r = backfill_affine_raw(r, man["cases"][case], targets, eeg, args.device)
                f.write_text(json.dumps(r, indent=1))
            rows.append(r)
            continue
        r = eval_case(case, man["cases"][case], targets, eeg, args.device)
        f.write_text(json.dumps(r, indent=1))
        rows.append(r)
        nl = r["nonlinear"]
        print(f"{case}: targets {nl['targets']['mean']:.2f}/{nl['targets']['max']:.2f}  EEG "
              f"{nl['eeg']['mean']:.2f}/{nl['eeg']['max']:.2f}  GPU {r['gpu_s']:.1f} s  {r['peak_mem_gb']:.2f} GB",
              flush=True)
    s = summarise(rows)
    s["targets"] = TARGETS
    s["eeg_names"] = names
    s["template"] = {"n_cases": tpl["n_cases"], "licence": tpl["licence"]}
    (OUT / "eval.json").write_text(json.dumps({"summary": s, "cases": rows}, indent=1))
    (OUT / "eval.md").write_text(markdown(s, rows, len(names)))
    print(markdown(s, rows, len(names)).split("\n\n| case")[0])


if __name__ == "__main__":
    main()
