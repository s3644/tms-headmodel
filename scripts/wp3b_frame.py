"""WP3b task B: how close do affine frames get to charm's worldToWorldTransformMatrix (W)?

On the 30 VALIDATION train cases (runs/wp3b/val_cases.json), with the label template and calibration
refitted without them (runs/wp3b/template_trainminusval/), each variant's 4x4 MNI -> subject affine A is
compared with W: mean |A p - W p| over a head-sized box of 27 points, and over the 80 EEG10-10 cap
positions (MNI, affine only). Variants: our label affine (raw), its ridge calibration, the intensity
affines of headmodel.frame, and (diagnostic only, not usable in the product) charm's own initialiser.

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_frame.py [--variants a b] [--cases A B]
Writes runs/wp3b/frame/<case>.json (resumable per variant), runs/wp3b/frame/frame.json and frame.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import dataset, frame, register  # noqa: E402
from eval_register import eeg_cap  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3b/frame"
TDIR = REPO / "runs/wp3b/template_trainminusval"
BOX = np.array([[x, y, z] for x in (-70, 0, 70) for y in (-100, 0, 70) for z in (-40, 30, 90)], float)
BRAIN_CLASSES = {k: register.MERGED[k] for k in ("brain", "wm", "csf")}

FP = frame.FrameParams
VARIANTS = {   # name: (FrameParams, init) ; init "raw" = our label affine, "principal" = moments only
    "mi_atlas_head": (FP(), "raw"),
    "mi_atlas_noneck": (FP(region="noneck"), "raw"),
    "mi_atlas_brain": (FP(region="brain"), "raw"),
    "gauss_atlas_head": (FP(metric="gauss"), "raw"),
    "gauss_atlas_brain": (FP(metric="gauss", region="brain"), "raw"),
    "mi_mni_head": (FP(template="mni"), "raw"),
    "mi_mni_brain": (FP(template="mni", region="brain"), "raw"),
    "mi_atlas_head_principal": (FP(), "principal"),
    "mi_atlas_head_coarse": (FP(stages=((8.0, 120), (4.0, 100))), "raw"),
    "mi_atlas_head_label1": (FP(w_label=1.0), "raw"),
    "mi_mni_brain_label1": (FP(template="mni", region="brain", w_label=1.0), "raw"),
    "mi_mni_brain_d3": (FP(template="mni", region="brain", brain_dilate_mm=3.0), "raw"),
    "mi_mni_noneck": (FP(template="mni", region="noneck", z_cut=-40.0), "raw"),
    "mi_mni_brain_principal": (FP(template="mni", region="brain"), "principal"),
    "mi_atlas_noneck_z40": (FP(region="noneck", z_cut=-40.0), "raw"),
    "gauss_atlas_noneck": (FP(metric="gauss", region="noneck"), "raw"),
}


def ap(A, p):
    A = np.asarray(A, float)
    return p @ A[:3, :3].T + A[:3, 3]


def dist(A, W, eeg):
    db = np.linalg.norm(ap(A, BOX) - ap(W, BOX), axis=1)
    de = np.linalg.norm(ap(A, eeg) - ap(W, eeg), axis=1)
    return {"box_mean": float(db.mean()), "box_max": float(db.max()), "eeg_mean": float(de.mean()),
            "eeg_max": float(de.max())}


def label_affine(t1, lab, params, dev):
    r = register.register(t1, nib.Nifti1Image(lab, t1.affine), None, device=dev, params=params,
                          template_dir=TDIR, write=False)
    return np.asarray(r["affine_raw"])


def principal(t1, lab):
    prob = np.asarray(nib.load(TDIR / register.TEMPLATE_FILE).dataobj, np.float32)
    tpl = nib.load(TDIR / register.TEMPLATE_FILE)
    return register._principal_init(np.isin(lab, (1, 2)), t1.affine, prob[..., 1:3].sum(-1), tpl.affine)


def run_case(case, m2m, variants, eeg, dev, redo=False):
    f = OUT / f"{case}.json"
    r = json.loads(f.read_text()) if f.is_file() else {"case": case, "A": {}, "d": {}, "s": {}}
    t1 = nib.load(Path(m2m) / "T1.nii.gz")
    lab = np.load(REPO / f"runs/wp3b/labels_cache/{case}.npz")["lab"]
    W = scipy.io.loadmat(Path(m2m) / "segmentation/coregistrationMatrices.mat")["worldToWorldTransformMatrix"]
    r["W"] = W.tolist()

    def put(name, A, secs, extra=None):
        r["A"][name], r["d"][name], r["s"][name] = np.asarray(A).tolist(), dist(A, W, eeg), secs
        if extra:
            r.setdefault("extra", {})[name] = extra

    if "raw" not in r["A"] or redo:
        t = time.perf_counter()
        A = label_affine(t1, lab, register.Params(svf_stages=(), calibrate=False, frame="labels"), dev)
        put("raw", A, time.perf_counter() - t)
        put("calibrated", register.calibrate(np.asarray(r["A"]["raw"]), register.load_calibration(TDIR)), 0.0)
    for name in variants:
        if name in r["A"] and not redo:
            continue
        t = time.perf_counter()
        if name == "labels_brainonly":
            A = label_affine(t1, lab, register.Params(svf_stages=(), calibrate=False, frame="labels",
                                                         classes=dict(BRAIN_CLASSES)), dev)
            put(name, A, time.perf_counter() - t)
            continue
        if name == "charm_init_diagnostic":   # charm's own initialiser (brainnet TREGA), NOT for the product
            from simnibs.segmentation import brain_surface
            from simnibs.utils import file_finder
            A = brain_surface.estimate_mni152_to_ras_affine([file_finder.SubjectFiles(subpath=str(m2m))], device=dev)[0]
            put(name, A, time.perf_counter() - t)
            continue
        fp, init = VARIANTS[name]
        A0 = np.asarray(r["A"]["raw"]) if init == "raw" else principal(t1, lab)
        torch.cuda.reset_peak_memory_stats()
        res = frame.intensity_affine(t1, A0, fp, device=dev, labels=lab, label_template_dir=TDIR)
        put(name, res["affine"], time.perf_counter() - t, {"loss": res["loss"], "peak_mem_gb": res["peak_mem_gb"]})
        torch.cuda.empty_cache()
    f.write_text(json.dumps(r, indent=1))
    return r


def summary(rows, names):
    out = {}
    for n in names:
        d = [r["d"][n] for r in rows if n in r["d"]]
        if not d:
            continue
        out[n] = {k: float(np.mean([x[k] for x in d])) for k in ("box_mean", "eeg_mean")}
        out[n].update({"box_max_case": float(np.max([x["box_mean"] for x in d])),
                       "eeg_max_point": float(np.max([x["eeg_max"] for x in d])),
                       "eeg_sd": float(np.std([x["eeg_mean"] for x in d], ddof=1)),
                       "s": float(np.mean([r["s"][n] for r in rows if n in r["s"]])), "n": len(d)})
        pk = [r.get("extra", {}).get(n, {}).get("peak_mem_gb") for r in rows]
        pk = [x for x in pk if x is not None]
        if pk:
            out[n]["peak_mem_gb"] = float(max(pk))
    return out


def ridge_frame(fit_rows, val_rows, feats):
    """headmodel.frame.fit_ridge on the fit (train minus val) rows only; the corrected val affines."""
    model = frame.fit_ridge([[r["A"][n] for n in feats] for r in fit_rows], [r["W"] for r in fit_rows],
                            meta={"features": feats}, cases=[r["case"] for r in fit_rows])
    As = [frame.apply_ridge([np.asarray(r["A"][n]) for n in feats], model) for r in val_rows]
    return As, model


def main():
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--variants", nargs="*", default=list(VARIANTS) + ["labels_brainonly"])
    ap_.add_argument("--cases", nargs="*", default=None)
    ap_.add_argument("--device", default="cuda")
    ap_.add_argument("--redo", action="store_true")
    ap_.add_argument("--fit", action="store_true", help="also compute the variants on the fit (train - val) cases")
    ap_.add_argument("--ridge", nargs="*", default=[], help="feature sets 'a+b+c' for ridge frames (needs --fit rows)")
    ap_.add_argument("--write-ridge", action="store_true",
                     help="write the raw+labels_brainonly ridge (Params(frame='ridge')) into the template dir")
    args = ap_.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    man = dataset.manifest(CT_ROOT)
    val = json.loads((REPO / "runs/wp3b/val_cases.json").read_text())
    tpl = json.loads((TDIR / "template.json").read_text())
    leak = {c["case"] for c in tpl["cases"]} & set(val)
    if leak:
        raise SystemExit(f"template contains validation cases: {sorted(leak)}")
    cases = args.cases or val
    assert all(man["cases"][c]["split"] == "train" for c in cases)
    _, eeg = eeg_cap()
    if args.fit:
        # the cases the refitted template was built from (the manifest grows while charm queues run)
        fit_cases = sorted(c["case"] for c in tpl["cases"] if c["case"] not in set(val)
                           and (Path(man["cases"][c["case"]]["m2m"]) / "segmentation/coregistrationMatrices.mat").is_file())
        for c in fit_cases:
            r = run_case(c, man["cases"][c]["m2m"], args.variants, eeg, args.device, args.redo)
            print("fit", c, " ".join(f"{n}={r['d'][n]['box_mean']:.2f}" for n in r["d"]), flush=True)
    rows = []
    for c in cases:
        r = run_case(c, man["cases"][c]["m2m"], args.variants, eeg, args.device, args.redo)
        rows.append(r)
        print(c, " ".join(f"{n}={r['d'][n]['eeg_mean']:.2f}" for n in ["raw", "calibrated"] + args.variants
                          if n in r["d"]), flush=True)
    names = ["raw", "calibrated"] + [n for n in args.variants if n not in ("raw", "calibrated")]
    ridge_info = {}
    if args.ridge:
        fitset = {c["case"] for c in tpl["cases"]} - set(val)
        fit_rows = [json.loads(f.read_text()) for f in sorted(OUT.glob("*.json")) if f.stem in fitset]
        for spec in args.ridge:
            feats = spec.split("+")
            fr = [r for r in fit_rows if all(n in r["A"] for n in feats)]
            As, info = ridge_frame(fr, rows, feats)
            name = "ridge(" + spec + ")"
            for r, A in zip(rows, As):
                r["A"][name], r["d"][name], r["s"][name] = A.tolist(), dist(A, r["W"], eeg), 0.0
            ridge_info[name] = {k: info[k] for k in ("lam", "loo_box_mm", "n_cases")}
            if args.write_ridge and feats == ["raw", "labels_brainonly"]:
                info.update({"what": "ridge: [label affine, brain-only label affine] -> charm's frame",
                             "split": "train minus the 30 validation cases", "built": time.strftime("%Y-%m-%d %H:%M"),
                             "licence": "research only: derived from CC BY-NC 4.0 data"})
                (TDIR / frame.RIDGE_FILE).write_text(json.dumps(info, indent=1))
            print(name, ridge_info[name], flush=True)
            names.append(name)
    s = summary(rows, names)
    (OUT / "frame.json").write_text(json.dumps({"summary": s, "ridge": ridge_info,
                                                 "cases": [r["case"] for r in rows]}, indent=1))
    lines = [f"# Affine frames vs charm's worldToWorldTransformMatrix ({len(rows)} validation cases)", "",
             "mm; box = 27 points of a head-sized box; EEG = 80 EEG10-10 positions, affine only.", "",
             "| frame | box mean | worst case box mean | EEG mean | EEG SD | EEG worst point | s/case | peak GB |",
             "|---|---|---|---|---|---|---|---|"]
    for n, a in s.items():
        lines.append(f"| {n} | {a['box_mean']:.2f} | {a['box_max_case']:.2f} | {a['eeg_mean']:.2f} | {a['eeg_sd']:.2f} | "
                     f"{a['eeg_max_point']:.2f} | {a['s']:.1f} | {a.get('peak_mem_gb', float('nan')):.2f} |")
    (OUT / "frame.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
