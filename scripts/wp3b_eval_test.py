"""WP3b Combine: the ONE held-out run of the chosen registration setting on the 57 WP3 test cases.

The setting was chosen on the validation split of the train cases only (runs/wp3b/frame_full,
runs/wp3b/brain); the template, calibration and ridge frame come from all train cases
(runs/wp3b/template, scripts/wp3b_build_template.py). Input per case: charm's own labels
(block-majority to the T1 grid), as in WP3. Scores, against charm:
  unprojected   simnibs.mni2subject_coords of the 10 MNI targets and the 80 EEG10-10 positions;
  projected     the EEG10-10 cap and the fiducials warped and projected on the skin as the contract
                delivers them (eval_register.eval_projected): on charm's mesh skin and on the label skin;
plus GPU time, peak GPU memory, min Jacobian of the two written toMNI maps (folding; the SVF's own
velocity-grid value is kept as jacobian_min_svf) and the round trip subject2mni(mni2subject(x)).
Gate per case: mean <= 2 mm and max <= 5 mm.

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_eval_test.py [--setting ridge_wm_only] [--cases ...]
                       [--out runs/wp3b/test --report runs/wp3b/eval_test]
Writes <out>/<case>/{m2m_<case>/, eval.json, projected.json} (resumable), <report>.json and <report>.md.
"""
from __future__ import annotations

import argparse
import collections
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
from headmodel import dataset, frame, register, splits  # noqa: E402
import eval_register as ev  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3b/test"
TDIR = REPO / "runs/wp3b/template"
GATE_MEAN, GATE_MAX = ev.GATE_MEAN, ev.GATE_MAX
SETTINGS = {   # chosen on validation (runs/wp3b/frame_full/full.md)
    "ridge_wm_only": {"frame": "ridge", "svf_class_weights": {"brain": 0.0}},
    # the first held-out run (ridge on the lambda grid 0.01 .. 3.0; runs/wp3b/test_ridge_lam0.01)
    "ridge_wm_only_lam0.01": {"frame": "ridge", "svf_class_weights": {"brain": 0.0},
                              "ridge_file": "frame_ridge_lam0.01.json"},
    "defaults": {},
}
# metric rows: (key, label)
METRICS = (("targets", "targets, unprojected"), ("eeg", "EEG, unprojected"),
           ("eeg_charm_skin", "EEG, projected on charm's skin"), ("eeg_label_skin", "EEG, projected on label skin"),
           ("fid_charm_skin", "fiducials, projected on charm's skin"),
           ("fid_label_skin", "fiducials, projected on label skin"))


def test_cases() -> list[str]:
    return splits.wp3_test_cases()   # frozen 57 WP3 test cases


def eval_case(case: str, info: dict, params: register.Params, targets, eeg, dev: str, out: Path = OUT) -> dict:
    import simnibs

    d = out / case
    f = d / "eval.json"
    if f.is_file():
        r = json.loads(f.read_text())
        if "jacobian_written" not in r:   # rows from before the written maps' Jacobian was recorded
            r["jacobian_min_svf"] = r.pop("jacobian_min")
            r["jacobian_written"] = ev.written_jacobian(d / f"m2m_{case}")
            r["jacobian_min"] = r["jacobian_written"]["min"]
            f.write_text(json.dumps(r, indent=1))
        return r
    ref = Path(info["m2m"])
    out_m2m = d / f"m2m_{case}"
    t1, up = nib.load(ref / "T1.nii.gz"), nib.load(ref / dataset.SEG_LABELS)
    lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), out_m2m, device=dev, params=params,
                            template_dir=TDIR)
    shutil.copyfile(ref / "T1.nii.gz", out_m2m / "T1.nii.gz")
    torch.cuda.empty_cache()
    pts = np.vstack([targets, eeg])
    ours = simnibs.mni2subject_coords(pts, str(out_m2m))
    charm = simnibs.mni2subject_coords(pts, str(ref))
    back = simnibs.subject2mni_coords(ours, str(out_m2m))
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), ours)
    inside = ((ijk >= 0) & (ijk <= np.array(t1.shape[:3]) - 1)).all(1)
    e = np.linalg.norm(ours - charm, axis=1)
    nt = len(targets)
    pr = ev.eval_projected(d, ref, REPO / f"runs/wp3b/label_skin/{case}.npz")
    (d / "projected.json").write_text(json.dumps(pr))
    g, fd = pr["groups"]["eeg"], pr["groups"]["fiducials"]
    r = {"case": case, "dataset": info.get("dataset"),
         "err": {"targets": e[:nt].tolist(), "eeg": e[nt:].tolist(),
                 "eeg_charm_skin": g["proj_charm_skin"], "eeg_label_skin": g["proj_label_skin"],
                 "fid_charm_skin": fd["proj_charm_skin"], "fid_label_skin": fd["proj_label_skin"]},
         "names": {"targets": list(ev.TARGETS), "eeg": g["names"], "fid": fd["names"]},
         "round_trip_max_mm": float(np.linalg.norm(back - pts, axis=1)[inside].max()),
         "points_outside_t1_fov": int((~inside).sum()),
         "gpu_s": res["timings"]["gpu_total"], "peak_mem_gb": res["peak_mem_gb"],
         "jacobian_min_svf": res["losses"].get("jacobian_min"), "timings": res["timings"],
         "A": res["affine"], "A_raw": res["affine_raw"]}
    r["jacobian_written"] = ev.written_jacobian(out_m2m)
    r["jacobian_min"] = r["jacobian_written"]["min"]
    f.write_text(json.dumps(r, indent=1))
    return r


# the validation rows (runs/wp3b/frame_full/<setting>, template_trainminusval) of each test setting
VAL_SETTING = {"ridge_wm_only": "params_ridge_wm_only_lamgrid", "ridge_wm_only_lam0.01": "params_ridge_wm_only"}


def validation(setting: str | None) -> dict | None:
    """The same setting on the 30 validation cases: (mean of case means, worst point, cases passing) for
    targets, EEG unprojected, and EEG / fiducials projected on charm's skin."""
    if not setting:
        return None
    d = REPO / "runs/wp3b/frame_full" / setting
    rows = [json.loads((d / c / "eval.json").read_text()) for c in splits.val_cases()
            if (d / c / "eval.json").is_file()]
    if not rows or not all("proj" in r for r in rows):
        return None
    get = {"targets": lambda r: r["targets"], "eeg": lambda r: r["eeg"],
           "eeg_charm_skin": lambda r: r["proj"]["eeg"], "fid_charm_skin": lambda r: r["proj"]["fiducials"]}
    out = {"setting": setting, "n": len(rows)}
    for k, g in get.items():
        m, x = np.array([g(r)["mean"] for r in rows]), np.array([g(r)["max"] for r in rows])
        out[k] = {"mean": float(m.mean()), "max": float(x.max()),
                  "cases_pass": int(((m <= GATE_MEAN) & (x <= GATE_MAX)).sum())}
    return out


def gate_text(s: dict) -> str:
    """The gate verdict. Positions = the EEG10-10 cap (whose 80 rows include Nz, Iz, LPA and RPA) and the
    fiducials together; a row under the gate on this one run is not the gate met when validation fails it."""
    m, v = s["metrics"], s.get("validation")
    pos = all(m[k]["gate"] for k in ("eeg", "eeg_charm_skin", "eeg_label_skin", "fid_charm_skin", "fid_label_skin"))
    t = (f"Gate: targets {'met' if m['targets']['gate'] else 'NOT met'}; EEG/fiducial positions "
         f"{'met' if pos else 'NOT met'} (the 4 fiducials are also rows of the 80-position EEG cap).")
    if v:
        fv = v["fid_charm_skin"]
        t += (f" Validation ({v['setting']}, {v['n']} train cases): targets {v['targets']['mean']:.2f} / "
              f"{v['targets']['max']:.2f} ({v['targets']['cases_pass']}/{v['n']}), EEG on charm's skin "
              f"{v['eeg_charm_skin']['mean']:.2f} / {v['eeg_charm_skin']['max']:.2f}, fiducials on charm's skin "
              f"{fv['mean']:.2f} / {fv['max']:.2f} ({fv['cases_pass']}/{v['n']}).")
        for k, lab in (("fid_charm_skin", "fiducials"), ("targets", "targets")):
            if m[k]["gate"] and v[k]["max"] > GATE_MAX:
                t += (f" The {lab} are under the gate on this test run only, not on validation (worst point "
                      f"{v[k]['max']:.2f} mm): not counted as met.")
    t += (" About 35 settings were compared on the same 30 validation cases, so validation numbers are "
          "optimistic (selection); the claim rests on this test run.")
    return t


def agg(rows: list[dict], key: str) -> dict:
    per = [np.asarray(r["err"][key]) for r in rows]
    means, maxs = np.array([p.mean() for p in per]), np.array([p.max() for p in per])
    return {"mean": float(means.mean()), "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
            "max_of_case_means": float(means.max()), "max": float(maxs.max()),
            "p95": float(np.percentile(np.concatenate(per), 95)),
            "points_over_max": int((np.concatenate(per) > GATE_MAX).sum()),
            "cases_pass": int(((means <= GATE_MEAN) & (maxs <= GATE_MAX)).sum()), "n": len(rows),
            "gate": bool(means.mean() <= GATE_MEAN and maxs.max() <= GATE_MAX)}


def baseline() -> dict:
    """The current defaults (WP3, runs/wp3/eval.json + task E's runs/wp3b/eval_projected.json)."""
    w = json.loads((REPO / "runs/wp3/eval.json").read_text())
    p = json.loads((REPO / "runs/wp3b/eval_projected.json").read_text())["summary"]
    out = {}
    for k in ("targets", "eeg"):
        a = w["summary"]["nonlinear"][k]
        out[k] = {**a, "n": w["summary"]["n_cases"]}
    for k, (g, c) in {"eeg_charm_skin": ("eeg", "proj_charm_skin"), "eeg_label_skin": ("eeg", "proj_label_skin"),
                      "fid_charm_skin": ("fiducials", "proj_charm_skin"),
                      "fid_label_skin": ("fiducials", "proj_label_skin")}.items():
        out[k] = {**p[g][c], "n": p["n_cases"]}
    out["gpu_s"], out["peak_mem_gb"] = w["summary"]["gpu_s"]["mean"], w["summary"]["peak_mem_gb"]
    out["round_trip_max_mm"] = w["summary"]["round_trip_max_mm"]
    return out


def worst(rows: list[dict], n: int = 10) -> dict:
    """The n worst targets (unprojected) and the n worst EEG / fiducial positions (projected on the
    label skin, with their unprojected and charm-skin errors)."""
    t, p = [], []
    for r in rows:
        e = r["err"]
        t += [{"case": r["case"], "point": nm, "mm": float(x)} for nm, x in zip(r["names"]["targets"], e["targets"])]
        for g, nms in (("eeg", r["names"]["eeg"]), ("fid", r["names"]["fid"])):
            unp = e["eeg"] if g == "eeg" else [float("nan")] * len(nms)   # fiducials: no unprojected column
            p += [{"case": r["case"], "point": nm, "mm": float(x), "charm_skin_mm": float(c), "unprojected_mm": float(u)}
                  for nm, x, c, u in zip(nms, e[f"{g}_label_skin"], e[f"{g}_charm_skin"], unp)]
    return {"targets": sorted(t, key=lambda q: -q["mm"])[:n], "positions": sorted(p, key=lambda q: -q["mm"])[:n]}


def cohort(case: str) -> str:
    return case[:3] if case[0].isdigit() else "".join(ch for ch in case if ch.isalpha())


def markdown(s: dict, base: dict, rows: list[dict], setting: str, params: dict, note: str = "") -> str:
    def cell(a):
        return f"{a['mean']:.2f} / {a['max']:.2f} ({a['cases_pass']}/{a['n']})"

    L = [f"# WP3b: registration vs charm, {s['n_cases']} held-out test cases", "",
         f"Setting `{setting}`: `Params(**{json.dumps(params)})`, template/calibration/ridge from all train "
         f"cases (runs/wp3b/template, {s['template_cases']} cases); chosen on the 30-case validation split of the "
         "train cases only. Input: charm's own labels. Cells: mean of case means / worst point (cases passing: "
         f"case mean <= {GATE_MEAN:g} and case max <= {GATE_MAX:g} mm). Gate: mean <= 2 mm and worst point <= 5 mm.", "",
         *([note, ""] if note else []),
         "| metric | WP3 defaults (runs/wp3) | this setting | SD | worst case mean | p95 | points > 5 mm | "
         "this run under the gate |",
         "|---|---|---|---|---|---|---|---|"]
    for k, lab in METRICS:
        a = s["metrics"][k]
        L.append(f"| {lab} | {cell(base[k])} | {cell(a)} | {a['sd']:.2f} | {a['max_of_case_means']:.2f} | "
                 f"{a['p95']:.2f} | {a['points_over_max']} | {'yes' if a['gate'] else 'no'} |")
    L += ["", gate_text(s)]
    L += ["", f"GPU time per case: mean {s['gpu_s']['mean']:.1f} s, max {s['gpu_s']['max']:.1f} s (WP3 defaults "
          f"{base['gpu_s']:.1f} s); peak GPU memory {s['peak_mem_gb']:.2f} GB (WP3 {base['peak_mem_gb']:.2f}); "
          f"min Jacobian of the written maps {s['jacobian_min']:.3f} (MNI2Conform_nonl "
          f"{s['jacobian_written']['MNI2Conform_nonl']:.3f}, Conform2MNI_nonl "
          f"{s['jacobian_written']['Conform2MNI_nonl']:.3f}; "
          f"{s['jacobian_written']['n_le0']} voxels <= 0; SVF velocity grid {s['jacobian_min_svf']:.3f}); "
          f"round trip max {s['round_trip_max_mm']:.2f} mm "
          f"(WP3 {base['round_trip_max_mm']:.2f}; points inside the T1 field of view, {s['points_outside_t1_fov']} "
          "case-points outside).", "", "Per dataset / cohort (this setting):", "",
          "| group | n | " + " | ".join(lab for _, lab in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for g, a in s["groups"].items():
        L.append(f"| {g} | {a['n']} | " + " | ".join(cell(a[k]) for k, _ in METRICS) + " |")
    L += ["", "10 worst targets (unprojected) | 10 worst EEG/fiducial positions (projected on the label skin; "
          "charm-skin and unprojected errors alongside):", "",
          "| case | target | mm | | case | position | label skin mm | charm skin mm | unprojected mm |",
          "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {a['case']} | {a['point']} | {a['mm']:.2f} | | {b['case']} | {b['point']} | {b['mm']:.2f} | "
          f"{b['charm_skin_mm']:.2f} | {b['unprojected_mm']:.2f} |" for a, b in zip(s["worst"]["targets"], s["worst"]["positions"])]
    L += ["", "| case | targets mean/max | EEG mean/max | EEG label-skin mean/max | fid label-skin mean/max | "
          "GPU s | min J |", "|---|---|---|---|---|---|---|"]
    mm = lambda x: f"{np.mean(x):.2f} / {np.max(x):.2f}"   # noqa: E731
    for r in rows:
        e = r["err"]
        L.append(f"| {r['case']} | {mm(e['targets'])} | {mm(e['eeg'])} | {mm(e['eeg_label_skin'])} | "
                 f"{mm(e['fid_label_skin'])} | {r['gpu_s']:.1f} | {r['jacobian_min']:.2f} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setting", default="ridge_wm_only", choices=list(SETTINGS))
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=OUT, help="per case outputs")
    ap.add_argument("--report", type=Path, default=REPO / "runs/wp3b/eval_test", help="<report>.json / .md")
    ap.add_argument("--note", default="", help="a line for the report (e.g. which run this is)")
    args = ap.parse_args()
    torch.set_num_threads(4)
    man = dataset.manifest(CT_ROOT)
    test = test_cases()
    assert len(test) == 57 and all(man["cases"][c]["split"] == "test" for c in test)
    tpl = json.loads((TDIR / "template.json").read_text())
    fitted = {c["case"] for c in tpl["cases"]} | set(json.loads((TDIR / "affines.json").read_text()))
    if fitted & set(test):
        raise SystemExit(f"template/ridge fitted with test cases: {sorted(fitted & set(test))}")
    params = register.Params(**SETTINGS[args.setting])
    if params.frame == "ridge":
        assert (TDIR / params.ridge_file).is_file()
        ridge = json.loads((TDIR / params.ridge_file).read_text())
        if set(ridge.get("cases", [])) & set(test):
            raise SystemExit("ridge frame fitted with test cases")
    targets = np.array(list(ev.TARGETS.values()), float)
    _, eeg = ev.eeg_cap()
    rows = []
    for case in args.cases or test:
        r = eval_case(case, man["cases"][case], params, targets, eeg, args.device, args.out)
        rows.append(r)
        e = r["err"]
        print(f"{case}: T {np.mean(e['targets']):.2f}/{np.max(e['targets']):.2f}  EEG {np.mean(e['eeg']):.2f}/"
              f"{np.max(e['eeg']):.2f}  EEG label-skin {np.mean(e['eeg_label_skin']):.2f}/{np.max(e['eeg_label_skin']):.2f}"
              f"  fid {np.mean(e['fid_label_skin']):.2f}/{np.max(e['fid_label_skin']):.2f}  GPU {r['gpu_s']:.1f} s "
              f"{r['peak_mem_gb']:.2f} GB  J {r['jacobian_min']:.2f}", flush=True)
    if args.cases:
        return
    keys = [k for k, _ in METRICS]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["dataset"] or "").split(" (")[0]].append(r)
        by["  " + cohort(r["case"])].append(r)
    s = {"n_cases": len(rows), "template_cases": tpl["n_cases"], "setting": args.setting,
         "params": SETTINGS[args.setting], "metrics": {k: agg(rows, k) for k in keys},
         "groups": {g.strip(): {"n": len(v), **{k: agg(v, k) for k in keys}}
                    for g, v in sorted(by.items(), key=lambda kv: (kv[0].startswith(" "), kv[0]))},
         "gpu_s": {"mean": float(np.mean([r["gpu_s"] for r in rows])), "max": float(np.max([r["gpu_s"] for r in rows]))},
         "peak_mem_gb": float(np.max([r["peak_mem_gb"] for r in rows])),
         "jacobian_min": float(np.min([r["jacobian_min"] for r in rows])),
         "jacobian_written": {**{n: float(np.min([r["jacobian_written"][n]["min"] for r in rows])) for n in ev.WARP_FILES},
                              "n_le0": int(sum(r["jacobian_written"][n]["n_le0"] for r in rows for n in ev.WARP_FILES))},
         "jacobian_min_svf": float(np.min([r["jacobian_min_svf"] for r in rows])),
         "validation": validation(VAL_SETTING.get(args.setting)),
         "round_trip_max_mm": float(np.max([r["round_trip_max_mm"] for r in rows])),
         "points_outside_t1_fov": int(np.sum([r["points_outside_t1_fov"] for r in rows])),
         "worst": worst(rows), "gate": {"mean": GATE_MEAN, "max": GATE_MAX}}
    base = baseline()
    Path(f"{args.report}.json").write_text(json.dumps({"summary": s, "wp3_defaults": base, "cases": rows}))
    md = markdown(s, base, rows, args.setting, SETTINGS[args.setting], args.note)
    Path(f"{args.report}.md").write_text(md)
    print(md.split("\n\n| case | targets")[0])


if __name__ == "__main__":
    main()
