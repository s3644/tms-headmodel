"""WP3b task B: the full registration (SVF stage unchanged) started from a given affine frame, on the
30 VALIDATION train cases, with the label template refitted without them (runs/wp3b/template_trainminusval).

A setting names a frame stored by scripts/wp3b_frame.py in runs/wp3b/frame/<case>.json ("calibrated" =
the current pipeline's frame; "W" = charm's own worldToWorldTransformMatrix, an upper bound). The SVF
starts from that frame (register(init_affine=A, Params(affine_stages=(), calibrate=False))); settings
in PARAM_FRAMES (params_ridge, params_mi_mni_brain_label1) compute it inside register() (Params.frame).
Scores: the 10 targets and 80 EEG10-10 positions (mni2subject_coords, unprojected) vs charm, and the
EEG cap + fiducials warped and projected on charm's skin (eval_register.eval_projected).

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_frame_full.py SETTING [SETTING ...] [--cases ...]
Writes runs/wp3b/frame_full/<setting>/<case>/{m2m_<case>/, eval.json} and runs/wp3b/frame_full/full.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import dataset, register  # noqa: E402
import eval_register as ev  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3b/frame_full"
TDIR = REPO / "runs/wp3b/template_trainminusval"
FRAMES = REPO / "runs/wp3b/frame"
GATE_MEAN, GATE_MAX = 2.0, 5.0
WP3 = {"frame": "labels", "svf_class_weights": None}   # the WP3 defaults the settings start from
RIDGE_LAM001 = "frame_ridge_lam0.01.json"
PARAM_FRAMES = {   # settings computed inside register() through Params.frame
    "params_mi_mni_brain_label1": {"frame": "intensity",
                                   "frame_params": {"template": "mni", "region": "brain", "w_label": 1.0}},
    # the first ridge fit (lambda grid 0.01 .. 3.0, kept as RIDGE_LAM001 in TDIR by wp3b_refit_ridge.py)
    "params_ridge": {"frame": "ridge", "ridge_file": RIDGE_LAM001},
    # Combine step: task C's brain-Dice options (SVF stages) with the current and the ridge frame
    "params_wm_only": {"svf_class_weights": {"brain": 0.0}},
    "params_ridge_wm_only": {"frame": "ridge", "svf_class_weights": {"brain": 0.0}, "ridge_file": RIDGE_LAM001},
    "params_ridge_brain0.3": {"frame": "ridge", "svf_class_weights": {"brain": 0.3}, "ridge_file": RIDGE_LAM001},
    # Fix step: the same with the ridge refitted on the extended lambda grid (frame.RIDGE_LAMS, TDIR/frame_ridge.json)
    "params_ridge_wm_only_lamgrid": {"frame": "ridge", "svf_class_weights": {"brain": 0.0}},
}


def run(setting: str, case: str, m2m: Path, dev: str, project: bool) -> dict:
    d = OUT / setting / case
    f = d / "eval.json"
    if f.is_file():
        r = json.loads(f.read_text())
        if not project or "proj" in r:
            return r
    out_m2m = d / f"m2m_{case}"
    t1 = nib.load(m2m / "T1.nii.gz")
    lab = np.load(REPO / f"runs/wp3b/labels_cache/{case}.npz")["lab"]
    if not (out_m2m / "toMNI/MNI2Conform_nonl.nii.gz").is_file():
        if setting in PARAM_FRAMES:
            p, A0 = register.Params(**{**WP3, **PARAM_FRAMES[setting]}), None
        else:
            fr = json.loads((FRAMES / f"{case}.json").read_text())
            A0 = np.asarray(fr["W"] if setting == "W" else fr["A"][setting])
            p = register.Params(affine_stages=(), calibrate=False, **WP3)
        res = register.register(t1, nib.Nifti1Image(lab, t1.affine), out_m2m, device=dev, params=p,
                                template_dir=TDIR, init_affine=A0)
        shutil.copyfile(m2m / "T1.nii.gz", out_m2m / "T1.nii.gz")
        info = {"gpu_s": res["timings"]["gpu_total"], "peak_mem_gb": res["peak_mem_gb"],
                "jacobian_min": res["losses"]["jacobian_min"], "A": res["affine"]}
        torch.cuda.empty_cache()
    else:
        info = json.loads(f.read_text()).get("info", {}) if f.is_file() else {}
    import simnibs

    ref = json.loads((REPO / f"runs/wp3b/charm_ref/{case}.json").read_text())
    ours = simnibs.mni2subject_coords(np.asarray(ref["mni"]), str(out_m2m))
    e = np.linalg.norm(ours - np.asarray(ref["charm"]), axis=1)
    nt = ref["n_targets"]
    r = {"case": case, "setting": setting, "info": info,
         "targets": {"mean": float(e[:nt].mean()), "max": float(e[:nt].max())},
         "eeg": {"mean": float(e[nt:].mean()), "max": float(e[nt:].max())}}
    if project:
        pr = ev.eval_projected(d, m2m, REPO / f"runs/wp3b/label_skin/{case}.npz")
        r["proj"] = {g: {"mean": float(np.mean(pr["groups"][g]["proj_charm_skin"])),
                         "max": float(np.max(pr["groups"][g]["proj_charm_skin"]))} for g in pr["groups"]}
    f.write_text(json.dumps(r, indent=1))
    return r


def agg(rows, get):
    m = np.array([get(r)["mean"] for r in rows])
    x = np.array([get(r)["max"] for r in rows])
    return f"{m.mean():.2f} / {x.max():.2f} ({int(((m <= GATE_MEAN) & (x <= GATE_MAX)).sum())}/{len(rows)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("settings", nargs="+")
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-project", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    man = dataset.manifest(CT_ROOT)
    val = json.loads((REPO / "runs/wp3b/val_cases.json").read_text())
    tpl = json.loads((TDIR / "template.json").read_text())
    assert not ({c["case"] for c in tpl["cases"]} & set(val)), "template contains validation cases"
    cases = args.cases or val
    assert all(man["cases"][c]["split"] == "train" for c in cases)
    lines = ["| frame | targets mean / worst (pass) | EEG unproj mean / worst (pass) | EEG proj charm skin | "
             "fiducials proj | GPU s | peak GB |", "|---|---|---|---|---|---|---|"]
    for s in args.settings:
        rows = []
        for c in cases:
            r = run(s, c, Path(man["cases"][c]["m2m"]), args.device, not args.no_project)
            rows.append(r)
            print(s, c, f"T {r['targets']['mean']:.2f}/{r['targets']['max']:.2f} EEG {r['eeg']['mean']:.2f}/"
                  f"{r['eeg']['max']:.2f}", flush=True)
        pj = (agg(rows, lambda r: r["proj"]["eeg"]), agg(rows, lambda r: r["proj"]["fiducials"])) \
            if all("proj" in r for r in rows) else ("-", "-")
        gs = [r["info"].get("gpu_s", np.nan) for r in rows]
        pk = [r["info"].get("peak_mem_gb", np.nan) for r in rows]
        lines.append(f"| {s} | {agg(rows, lambda r: r['targets'])} | {agg(rows, lambda r: r['eeg'])} | {pj[0]} | "
                     f"{pj[1]} | {np.nanmean(gs):.1f} | {np.nanmax(pk):.2f} |")
    md = OUT / "full.md"
    old = md.read_text().splitlines() if md.is_file() else []
    keep = [l for l in old[2:] if l.split(" | ")[0][2:] not in args.settings]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines[:2] + keep + lines[2:]) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
