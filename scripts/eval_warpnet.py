"""WP3b task A: score a warp-network checkpoint on the VALIDATION cases, through the written maps.

For each case: register() with Params(init_warp=<checkpoint>) and the net's template
(runs/wp3b/template_trainminusval), charm's labels (runs/wp3b/labels_cache) as input; the net's warp
alone (default) or the SVF refinement started from it (--svf, register's default stages, or
--svf-stages; --svf-class-weights, --w-init). charm's toMNI/MNI2Conform_nonl, coregistrationMatrices.mat
and T1 are written to <out>/<case>/m2m_<case>, then simnibs.mni2subject_coords of the 10 targets and the 80
EEG10-10_UI_Jurak_2007 positions (incl. fiducials; unprojected) is compared with charm's, and the EEG cap and
the fiducials warped and projected on the skin as the contract delivers them (eval_register.eval_projected)
on charm's mesh skin and on the label skin are compared with charm's eeg_positions/*.csv. Also recorded: GPU
time, peak GPU memory, min Jacobian of the two written toMNI maps (the SVF velocity grid's as jacobian_min_svf).

Validation cases only by default; --allow-test is for the orchestrator's final held-out run.

Usage (rtms-gpu env):
  PYTHONPATH=src python scripts/eval_warpnet.py [--ckpt runs/wp3b/warpnet/best.pt] [--svf] [--tag net]
      [--svf-stages '[[2,4,150]]'] [--svf-class-weights '{"brain": 0}' | none] [--w-init 0.1] [--no-project]
      [--no-fill] [--unfold 50] [--rule NAME]   (a rule sealed beforehand with scripts/preregister.py; it must include no folding)
Writes <out>/<case>/eval.json (resumable), <out>/eval.json and <out>/eval.md; out defaults to
runs/wp3b/warpnet/eval_<tag>.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import dataset, register, warpnet  # noqa: E402
from train_warpnet import LABELS_CACHE, case_splits, splits  # noqa: E402
import eval_register as ev  # noqa: E402

GATE_MEAN, GATE_MAX = 2.0, 5.0
LABEL_SKIN = REPO / "runs/wp3b/label_skin"
# projected per point errors: key -> (eval_projected group, condition)
PROJ = {"eeg_charm_skin": ("eeg", "proj_charm_skin"), "eeg_label_skin": ("eeg", "proj_label_skin"),
        "fid_charm_skin": ("fiducials", "proj_charm_skin"), "fid_label_skin": ("fiducials", "proj_label_skin")}
METRICS = (("targets", "targets, unprojected"), ("eeg", "EEG, unprojected"),
           ("eeg_charm_skin", "EEG, projected on charm's skin"), ("eeg_label_skin", "EEG, projected on label skin"),
           ("eeg_rtms", "EEG rTMS set, unprojected"), ("eeg_charm_skin_rtms", "EEG rTMS set, projected on charm's skin"),
           ("eeg_label_skin_rtms", "EEG rTMS set, projected on label skin"),
           ("fid_charm_skin", "fiducials, projected on charm's skin"),
           ("fid_label_skin", "fiducials, projected on label skin"))
# the WP3 gate from round 4 on (sealed rule wp3_gate_rtms_set): these rows, each mean <= 2 and worst <= 5 mm,
# plus no folding; the whole-cap EEG rows are reported only
GATE_ROWS = ("targets", "eeg_rtms", "eeg_charm_skin_rtms", "eeg_label_skin_rtms", "fid_charm_skin", "fid_label_skin")


def add_rtms(r: dict) -> dict:
    """The rTMS-set EEG rows of a case (in place) from its whole-cap per point errors (cap order)."""
    names = ev.eeg_cap()[0]
    idx = [names.index(n) for n in ev.RTMS_EEG]
    for k in ("eeg", "eeg_charm_skin", "eeg_label_skin"):
        if f"err_{k}" in r:
            e = np.asarray(r[f"err_{k}"])
            if len(e) != len(names):
                raise ValueError(f"{r['case']}: {len(e)} {k} errors for a {len(names)}-position cap")
            r[f"{k}_rtms"] = {"mean": float(e[idx].mean()), "max": float(e[idx].max())}
    return r


def add_projected(r: dict, case: str, m2m: Path, out: Path) -> dict:
    """Projected EEG / fiducial errors and the written maps' Jacobian for a scored case row (in place)."""
    pr = ev.eval_projected(out / case, m2m, LABEL_SKIN / f"{case}.npz")
    (out / case / "projected.json").write_text(json.dumps(pr))
    for k, (g, c) in PROJ.items():
        e = np.asarray(pr["groups"][g][c])
        r[k] = {"mean": float(e.mean()), "max": float(e.max())}
        r[f"err_{k}"] = e.tolist()
    if "jacobian_min_svf" not in r:
        r["jacobian_min_svf"] = r.get("jacobian_min")
    r["jacobian_written"] = ev.written_jacobian(out / case / f"m2m_{case}")
    r["jacobian_min"] = r["jacobian_written"]["min"]
    return r


def labels(case: str, m2m: Path, t1) -> np.ndarray:
    f = LABELS_CACHE / f"{case}.npz"
    if f.is_file():
        return np.load(f)["lab"]
    up = nib.load(m2m / dataset.SEG_LABELS)
    return dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)


def eval_case(case: str, m2m: Path, out: Path, params: register.Params, template_dir: Path, device: str) -> dict:
    import simnibs

    out_m2m = out / case / f"m2m_{case}"
    out_m2m.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(m2m / "T1.nii.gz")
    lab = labels(case, m2m, t1)
    res = register.register(t1, nib.Nifti1Image(lab, t1.affine), out_m2m, device=device, params=params,
                            template_dir=template_dir)
    shutil.copyfile(m2m / "T1.nii.gz", out_m2m / "T1.nii.gz")
    pts, nt = warpnet.mni_points()
    ours = np.asarray(simnibs.mni2subject_coords(pts, str(out_m2m)))
    ref_f = REPO / "runs/wp3b/charm_ref" / f"{case}.json"
    if ref_f.is_file() and np.allclose(json.loads(ref_f.read_text())["mni"], pts):
        charm = np.asarray(json.loads(ref_f.read_text())["charm"])
    else:
        charm = np.asarray(simnibs.mni2subject_coords(pts, str(m2m)))
    e = np.linalg.norm(ours - charm, axis=1)
    back = np.asarray(simnibs.subject2mni_coords(ours, str(out_m2m)))
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), ours)
    inside = ((ijk >= 0) & (ijk <= np.array(t1.shape[:3]) - 1)).all(1)
    st = lambda a: {"mean": float(a.mean()), "max": float(a.max())}   # noqa: E731
    return {"case": case, "targets": st(e[:nt]), "eeg": st(e[nt:]), "err_targets": e[:nt].tolist(),
            "err_eeg": e[nt:].tolist(), "round_trip_max_mm": float(np.linalg.norm(back - pts, axis=1)[inside].max()),
            "gpu_s": res["timings"]["gpu_total"], "peak_mem_gb": res["peak_mem_gb"],
            "jacobian_min": res["losses"].get("jacobian_min"), "jacobian_min_svf": res["losses"].get("jacobian_min")}


def summarise(rows: list[dict]) -> dict:
    rows = [add_rtms(r) for r in rows]
    s = {"n_cases": len(rows)}
    for w in [k for k, _ in METRICS if all(k in r for r in rows)]:
        means = np.array([r[w]["mean"] for r in rows])
        maxs = np.array([r[w]["max"] for r in rows])
        s[w] = {"mean": float(means.mean()), "sd": float(means.std(ddof=1)) if len(rows) > 1 else 0.0,
                "max_of_case_means": float(means.max()), "max": float(maxs.max()),
                "cases_pass": int(((means <= GATE_MEAN) & (maxs <= GATE_MAX)).sum())}
    s["both_pass"] = int(sum(r["targets"]["mean"] <= GATE_MEAN and r["targets"]["max"] <= GATE_MAX and
                             r["eeg"]["mean"] <= GATE_MEAN and r["eeg"]["max"] <= GATE_MAX for r in rows))
    s["gpu_s_mean"] = float(np.mean([r["gpu_s"] for r in rows]))
    s["peak_mem_gb"] = float(np.max([r["peak_mem_gb"] for r in rows]))
    s["round_trip_max_mm"] = float(np.max([r["round_trip_max_mm"] for r in rows]))
    s["jacobian_min"] = float(np.min([r["jacobian_min"] for r in rows if r["jacobian_min"] is not None] or [np.nan]))
    s["jacobian_min_svf"] = float(np.min([r["jacobian_min_svf"] for r in rows if r.get("jacobian_min_svf") is not None]
                                         or [np.nan]))
    s["jacobian_written"] = all("jacobian_written" in r for r in rows)
    # no folding: min Jacobian > 0 in both written toMNI maps (only counted when the written maps were scored)
    s["cases_no_fold"] = int(sum(r["jacobian_written"]["min"] > 0 for r in rows if "jacobian_written" in r)) \
        if s["jacobian_written"] else None
    if all(k in s for k in GATE_ROWS) and s["cases_no_fold"] is not None:
        rows_ok = {k: s[k]["mean"] <= GATE_MEAN and s[k]["max"] <= GATE_MAX for k in GATE_ROWS}
        s["gate_rtms"] = {**rows_ok, "no_folding": s["cases_no_fold"] == len(rows),
                          "met": all(rows_ok.values()) and s["cases_no_fold"] == len(rows)}
    return s


def markdown(s: dict, rows: list[dict], title: str) -> str:
    L = [f"# {title}", "", "Distance (mm) of our to charm's points: mni2subject_coords (unprojected), and the EEG cap "
         "and fiducials projected on the skin vs charm's eeg_positions/*.csv. Per case mean / max, then mean / SD / "
         "worst case mean / worst point over cases. Gate: <= 2 mm mean and <= 5 mm worst.", "",
         "| points | mean | SD | worst case mean | worst point | cases passing |", "|---|---|---|---|---|---|"]
    for w, lab in METRICS:
        if w not in s:
            continue
        a = s[w]
        L.append(f"| {lab} | {a['mean']:.2f} | {a['sd']:.2f} | {a['max_of_case_means']:.2f} | {a['max']:.2f} | "
                 f"{a['cases_pass']}/{s['n_cases']} |")
    jw = "written toMNI maps" if s["jacobian_written"] else "SVF velocity grid (written maps not scored)"
    nf = f"{s['cases_no_fold']}/{s['n_cases']}" if s.get("cases_no_fold") is not None else "not scored"
    if "gate_rtms" in s:
        g = s["gate_rtms"]
        L += ["", f"Gate (rule wp3_gate_rtms_set: targets, rTMS-set EEG and fiducials each <= {GATE_MEAN:g} mm mean and "
              f"<= {GATE_MAX:g} mm worst, no folding): {'MET' if g['met'] else 'NOT met'} ("
              + ", ".join(f"{k} {'ok' if v else 'fails'}" for k, v in g.items() if k != "met") + ")."]
    L += ["", f"Both pass: {s['both_pass']}/{s['n_cases']}. No folding (written maps' min J > 0): {nf}. GPU {s['gpu_s_mean']:.1f} s per case, peak "
          f"{s['peak_mem_gb']:.2f} GB, round trip max {s['round_trip_max_mm']:.2f} mm, min Jacobian "
          f"{s['jacobian_min']:.3f} ({jw}; SVF velocity grid {s['jacobian_min_svf']:.3f}).", "",
          "| case | targets mean / max | EEG mean / max | EEG charm skin | fid charm skin | min J |", "|---|---|---|---|---|---|"]
    mm = lambda r, k: f"{r[k]['mean']:.2f} / {r[k]['max']:.2f}" if k in r else "-"   # noqa: E731
    L += [f"| {r['case']} | {mm(r, 'targets')} | {mm(r, 'eeg')} | {mm(r, 'eeg_charm_skin')} | {mm(r, 'fid_charm_skin')} | "
          f"{r['jacobian_min']:.3f} |" for r in rows]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=str(warpnet.OUT / "best.pt"))
    ap.add_argument("--svf", action="store_true", help="refine with register's SVF stages started from the net")
    ap.add_argument("--svf-stages", default=None, help='JSON list of (eval mm, velocity mm, iters), implies --svf')
    ap.add_argument("--w-init", type=float, default=0.0)
    ap.add_argument("--svf-class-weights", default="none",
                    help='SVF soft Dice weights, JSON (e.g. \'{"brain": 0}\', the WP3b default) or none (class_weights)')
    ap.add_argument("--no-project", action="store_true", help="skip the projected EEG / fiducials and written Jacobian")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--unfold", type=int, default=0, help="Params.unfold_iters: repair folded voxels of the written "
                                                         "maps (WP3 round 4); 0 off")
    ap.add_argument("--no-fill", action="store_true", help="raw net velocity outside its training support "
                                                         "(Params.net_fill=False, the task N behaviour)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--allow-test", action="store_true", help="orchestrator only: held-out test cases")
    ap.add_argument("--rule", default=None, help="name of the sealed rule (scripts/preregister.py) this run is judged "
                                              "by: verified against the ledger, its hash recorded in eval.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rule = None
    if args.rule:
        import preregister

        if not preregister.verify(args.rule):
            raise SystemExit(f"rule {args.rule} is not sealed or was changed (scripts/preregister.py verify)")
        rule = {"name": args.rule, **preregister.ledger()[f"{args.rule}.json"]}
    man, fit, val = splits()
    test = case_splits.wp3_test_cases()
    cases = args.cases or val
    bad = [c for c in cases if c in test] if not args.allow_test else []
    if bad or any(c in fit for c in cases):
        raise SystemExit(f"not validation cases: {[c for c in cases if c in test or c in fit][:5]}")
    if args.n:
        cases = cases[: args.n]
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tdir = Path(ck["config"]["template_dir"])
    if args.allow_test is False and set(ck.get("train_cases", [])) & set(cases):
        raise SystemExit("checkpoint was trained on some of these cases")
    stages = json.loads(args.svf_stages) if args.svf_stages else None
    svf = args.svf or stages is not None
    # svf_class_weights none: class_weights in the SVF too (WP3's); the WP3b default turns the brain class off
    cw = None if args.svf_class_weights.lower() == "none" else json.loads(args.svf_class_weights)
    params = register.Params(init_warp=args.ckpt, w_init=args.w_init, svf_class_weights=cw, net_fill=not args.no_fill,
                             unfold_iters=args.unfold,
                             **({"svf_stages": tuple(map(tuple, stages))} if stages else {} if svf else {"svf_stages": ()}))
    tag = args.tag or ("net_svf" if svf else "net")
    out = Path(args.out) if args.out else warpnet.OUT / f"eval_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        f = out / case / "eval.json"
        if f.is_file() and not args.redo:
            r = json.loads(f.read_text())
            if not args.no_project and "eeg_charm_skin" not in r:
                r = add_projected(r, case, Path(man[case]["m2m"]), out)
                f.write_text(json.dumps(r, indent=1))
            rows.append(r)
            continue
        t0 = time.time()
        r = eval_case(case, Path(man[case]["m2m"]), out, params, tdir, args.device)
        torch.cuda.empty_cache()
        if not args.no_project:
            r = add_projected(r, case, Path(man[case]["m2m"]), out)
        f.write_text(json.dumps(r, indent=1))
        rows.append(r)
        print(f"{case}: targets {r['targets']['mean']:.2f}/{r['targets']['max']:.2f}  EEG {r['eeg']['mean']:.2f}/"
              f"{r['eeg']['max']:.2f}  " + (f"EEG charm skin {r['eeg_charm_skin']['mean']:.2f}/{r['eeg_charm_skin']['max']:.2f}"
              f"  fid {r['fid_charm_skin']['mean']:.2f}/{r['fid_charm_skin']['max']:.2f}  " if "eeg_charm_skin" in r else "") +
              f"J {r['jacobian_min']:.3f}  GPU {r['gpu_s']:.1f} s {r['peak_mem_gb']:.2f} GB  ({time.time() - t0:.0f} s)",
              flush=True)
    s = summarise(rows)
    s.update({"ckpt": args.ckpt, "ckpt_epoch": ck.get("epoch"), "svf": svf, "svf_stages": params.svf_stages,
              "w_init": args.w_init, "net_fill": not args.no_fill, "unfold_iters": args.unfold, "svf_class_weights": cw, "template_dir": str(tdir), "rule": rule})
    title = f"Warp network {Path(args.ckpt).name} (epoch {ck.get('epoch')}), {'net + SVF' if svf else 'net alone'}, " \
            f"{len(rows)} {'test' if args.allow_test else 'validation'} cases"
    (out / "eval.json").write_text(json.dumps({"summary": s, "cases": rows}, indent=1))
    (out / "eval.md").write_text(markdown(s, rows, title))
    print(markdown(s, rows, title).split("\n\n| case")[0])


if __name__ == "__main__":
    main()
