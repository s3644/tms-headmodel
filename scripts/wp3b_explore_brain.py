"""WP3b explore C (target slips): brain-warp variants of headmodel.register on the 30 VALIDATION train
cases (runs/wp3b/val_cases.json), with the label template and affine calibration refitted without them
(runs/wp3b/template_trainminusval, scripts/wp3b_prepare.py). No test case is read.

Per case: register(write=False); our MNI2Conform map is read exactly as simnibs.mni2subject_coords
reads the file (simnibs.utils.transformations.coordinates_nonlinear, trilinear) at the 10 targets and the
EEG10-10_UI_Jurak_2007 positions (unprojected), against charm's (runs/wp3b/charm_ref, cached).

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_explore_brain.py VARIANT [VARIANT ...]
                       PYTHONPATH=src python scripts/wp3b_explore_brain.py --summary
Writes runs/wp3b/brain/<variant>/<case>.json (resumable) and runs/wp3b/brain/summary.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import register  # noqa: E402

W = REPO / "runs/wp3b"
OUT = W / "brain"
TEMPLATE = W / "template_trainminusval"
GATE_MEAN, GATE_MAX = 2.0, 5.0

S3 = ((4.0, 16.0, 100), (2.0, 8.0, 150), (2.0, 4.0, 150))   # WP3 default
VARIANTS = {
    "base": {},
    "stiff3": {"stiff_brain": 3.0}, "stiff30": {"stiff_brain": 30.0}, "stiff100": {"stiff_brain": 100.0},
    "v8": {"svf_stages": ((4.0, 16.0, 100), (2.0, 8.0, 300))},
    "v6": {"svf_stages": ((4.0, 16.0, 100), (2.0, 8.0, 150), (2.0, 6.0, 150))},
    "bend1": {"w_bend": 1.0}, "bend5": {"w_bend": 5.0},
    "it_half": {"svf_stages": ((4.0, 16.0, 50), (2.0, 8.0, 75), (2.0, 4.0, 75))},
    "it_last300": {"svf_stages": ((4.0, 16.0, 100), (2.0, 8.0, 150), (2.0, 4.0, 300))},
    # Dice on WM only inside the brain (brain class off in the SVF stages; the affine is unchanged)
    "wm_only": {"svf_class_weights": {"brain": 0.0}},
    "wm2": {"svf_class_weights": {"wm": 2.0}},
    "csf2": {"svf_class_weights": {"csf": 2.0}},
    # an extra GM class (GM/WM and GM/CSF boundaries), SVF stages only
    "gm": {"classes": {**register.MERGED, "gm": (2,)}, "class_weights": {"gm": 0.0},
           "svf_class_weights": {"gm": 1.0}},
    "ncc_b0.2": {"w_ncc_brain": 0.2}, "ncc_b1": {"w_ncc_brain": 1.0},
    "sdt0.3": {"w_sdt": 0.3}, "sdt1": {"w_sdt": 1.0}, "sdt3": {"w_sdt": 3.0}, "sdt10": {"w_sdt": 10.0},
    "clip10": {"sdt_clip": 10.0}, "sdt_csf": {"sdt_classes": ("brain", "wm", "csf")},
    "sdt_wm": {"sdt_classes": ("wm",)},
    "reg_up": {"svf_reg_scale": (1.0, 2.0, 4.0)},
    "stiff5": {"stiff_brain": 5.0}, "stiff20": {"stiff_brain": 20.0},
    "bend0.5": {"w_bend": 0.5}, "bend2": {"w_bend": 2.0},
    "brain0.3": {"svf_class_weights": {"brain": 0.3}},
}


WP3 = {"frame": "labels", "svf_class_weights": None}   # the WP3 defaults these variants start from


def variant_params(name: str) -> register.Params:
    if name in VARIANTS:
        return register.Params(**{**WP3, **VARIANTS[name]})
    # ad-hoc combination "a+b+c": later keys override (dicts merged)
    kw = dict(WP3)
    for part in name.split("+"):
        for k, v in VARIANTS[part].items():
            kw[k] = {**kw[k], **v} if isinstance(v, dict) and isinstance(kw.get(k), dict) and k != "classes" else v
    return register.Params(**kw)


def labels(case):
    return np.load(W / "labels_cache" / f"{case}.npz")["lab"]


def run_case(case: str, m2m: str, params: register.Params, device: str, oracle: bool = False) -> dict:
    """oracle: DIAGNOSTIC only (never for a choice): the SVF starts from charm's own affine
    (coregistrationMatrices.mat), to split the error into frame and brain-warp parts."""
    import dataclasses

    import scipy.io
    from simnibs.utils.transformations import coordinates_nonlinear

    init = None
    if oracle:
        init = scipy.io.loadmat(Path(m2m) / "segmentation/coregistrationMatrices.mat")["worldToWorldTransformMatrix"]
        params = dataclasses.replace(params, affine_stages=(), calibrate=False)

    ref = json.loads((W / "charm_ref" / f"{case}.json").read_text())
    t1 = nib.load(Path(m2m) / "T1.nii.gz")
    lab = labels(case)
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), None, device=device, params=params,
                            template_dir=TEMPLATE, write=False, init_affine=init)
    tr = res["transform"]
    ours = coordinates_nonlinear(np.asarray(ref["mni"]), (tr["mni2conform"], tr["atlas_affine"]))
    e = np.linalg.norm(ours - np.asarray(ref["charm"]), axis=1)
    nt = ref["n_targets"]
    return {"case": case, "targets": {"mean": float(e[:nt].mean()), "max": float(e[:nt].max())},
            "eeg": {"mean": float(e[nt:].mean()), "max": float(e[nt:].max())},
            "err_targets": e[:nt].tolist(), "err_eeg": e[nt:].tolist(),
            "gpu_s": res["timings"]["gpu_total"], "peak_mem_gb": res["peak_mem_gb"],
            "jacobian_min": res["losses"].get("jacobian_min"), "dice_svf": res["losses"].get("dice_svf")}


def agg(rows: list[dict]) -> dict:
    s = {}
    for w in ("targets", "eeg"):
        m = np.array([r[w]["mean"] for r in rows])
        x = np.array([r[w]["max"] for r in rows])
        s[w] = {"mean": float(m.mean()), "worst_case_mean": float(m.max()), "max": float(x.max()),
                "pass": int(((m <= GATE_MEAN) & (x <= GATE_MAX)).sum()), "n_max_gt5": int((x > GATE_MAX).sum())}
    s["n"] = len(rows)
    s["gpu_s"] = float(np.mean([r["gpu_s"] for r in rows]))
    s["peak_mem_gb"] = float(np.max([r["peak_mem_gb"] for r in rows]))
    s["jacobian_min"] = float(np.min([r["jacobian_min"] for r in rows]))
    return s


def summary(val: list[str]):
    out, lines = {}, ["| variant | targets mean | worst case mean | worst point | pass | pts>5 | EEG mean | EEG worst | "
                      "Jmin | GPU s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        rows = [json.loads((d / f"{c}.json").read_text()) for c in val if (d / f"{c}.json").is_file()]
        if len(rows) < len(val):
            continue
        s = out[d.name] = agg(rows)
        t, e = s["targets"], s["eeg"]
        lines.append(f"| {d.name} | {t['mean']:.2f} | {t['worst_case_mean']:.2f} | {t['max']:.2f} | {t['pass']}/{s['n']} | "
                     f"{t['n_max_gt5']} | {e['mean']:.2f} | {e['max']:.2f} | {s['jacobian_min']:.2f} | {s['gpu_s']:.1f} |")
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(4)
    val = json.loads((W / "val_cases.json").read_text())
    tpl = json.loads((TEMPLATE / "template.json").read_text())
    leak = {r["case"] for r in tpl["cases"]} & set(val)
    if leak:
        raise SystemExit(f"template was built with validation cases: {sorted(leak)}")
    cal = json.loads((TEMPLATE / register.CALIBRATION_FILE).read_text())
    if set(cal["cases"]) & set(val):
        raise SystemExit("calibration was fitted with validation cases")
    from headmodel import dataset
    man = dataset.manifest(Path.home() / "Projects/ct-headmodel")
    assert all(man["cases"][c]["split"] == "train" for c in val)
    for name in args.variants:
        oracle = name.startswith("oracle:")   # diagnostic: charm's affine as the frame
        params = variant_params(name.removeprefix("oracle:"))
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "params.json").write_text(json.dumps(params.__dict__, indent=1, default=str))
        t0 = time.time()
        for case in val:
            f = d / f"{case}.json"
            if f.is_file():
                continue
            r = run_case(case, man["cases"][case]["m2m"], params, args.device, oracle)
            f.write_text(json.dumps(r, indent=1))
            torch.cuda.empty_cache()
        rows = [json.loads((d / f"{c}.json").read_text()) for c in val]
        s = agg(rows)
        print(f"{name}: targets {s['targets']['mean']:.2f} / {s['targets']['max']:.2f} pass {s['targets']['pass']}  "
              f"EEG {s['eeg']['mean']:.2f} / {s['eeg']['max']:.2f}  Jmin {s['jacobian_min']:.2f}  "
              f"peak {s['peak_mem_gb']:.2f} GB  {time.time() - t0:.0f} s", flush=True)
    summary(val)


if __name__ == "__main__":
    main()
