"""R5 end-to-end score (PLAN 10.8; rule r5_e2e): the MRI fast route with the R5 models vs the current release,
on the same inputs, against charm on the original 1 mm T1.

Models (navprep exactly as the locator runs it, chosen by environment):
  release  the live release's models: Dataset601 fold 0 + the WP3b round-4 warp (round 3's net)
  r5       Dataset602 (R5 segmenter), folds 0-4 ensembled + the R5 warp net (runs/r5/warp/final/last.pt)
Arms (inputs made from charm's input T1 of the case; scripts/sim_series.py, seeded per case):
  1mm      the 1 mm T1 itself
  ax5      one axial series, 5 mm slices, no gap
  ax5gap   one axial series, 5 mm slices, 1 mm gap
  axsag5   axial + sagittal, 5 mm with a 1 mm gap, the sagittal moved (+-2 deg, +-2 mm) and covering the central
           120 mm (navprep's combine stage)
Sets: pilot (the first 5 sorted R5 warp validation heads: a machinery check), test (the IXI test heads of the
sealed r5_ixi_split: the one use), compare (the 57 WP3 test cases, 1mm, r5 only; the release's numbers there are
M1's, runs/m1_mri_e2e).

Scoring per run as M1 (scripts/m1_mri_e2e.py): targets unprojected; the rTMS EEG set and the fiducials warped and
projected on OUR skin, vs charm's; written-map min Jacobian. Writes runs/r5/e2e/<set>/<model>/<arm>/<case>/
{m2m_<case>/, eval.json, ...} (resumable), <set>/summary.json and summary.md; with --rule, the rule's decision.

Usage (rtms-gpu env):
  PYTHONPATH=src python scripts/r5_e2e.py --set pilot|test|compare [--rule r5_e2e] [--models ...] [--arms ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import splits  # noqa: E402
import eval_warpnet as ew  # noqa: E402
import m1_mri_e2e as m1  # noqa: E402

R5 = REPO / "runs/r5"
OUT = R5 / "e2e"
MANIFEST = Path(os.environ.get("HM_R5_MANIFEST", REPO / "data/manifests/r5.json"))
IXI_SPLIT = REPO / "runs/prereg/r5_ixi_split.json"
R5_SEG = Path(os.environ.get("HM_R5_SEG", REPO / "data/nnunet/results/Dataset602_MRHeadR5/nnUNetTrainerHMThick__nnUNetPlans__3d_fullres"))
R5_NET = Path(os.environ.get("HM_R5_NET", R5 / "warp/final/last.pt"))   # (smoke runs: other paths)
R4_NET = REPO / "runs/wp3b/warpnet_r3/e1000/last.pt"
R1_SEG_TRAIN = REPO / "data/nnunet/raw/Dataset601_MRHead/imagesTr"
R5_SEG_TRAIN = REPO / "data/nnunet/raw/Dataset602_MRHeadR5/imagesTr"
M1 = REPO / "runs/m1_mri_e2e"
CAP_GB = m1.CAP_GB
MODELS = {
    "release": {"HM_NAVPREP_MR_FOLDS": "0", "HM_WARP_NET": str(R4_NET)},
    "r5": {"HM_NAVPREP_MR_MODEL": str(R5_SEG), "HM_NAVPREP_MR_FOLDS": os.environ.get("HM_R5_FOLDS", "0,1,2,3,4"),
           "HM_WARP_NET": str(R5_NET)},
}
ARMS = {   # sim_series.py arguments (None: the 1 mm T1 itself)
    "1mm": None,
    "ax5": ["--series", "ax:5:0"],
    "ax5gap": ["--series", "ax:5:1"],
    "axsag5": ["--series", "ax:5:1", "sag:5:1", "--cover", "sag:120"],
}
GATE_ROWS = m1.GATE_ROWS   # targets, rTMS EEG set on our skin, fiducials on our skin


def charm_m2m(case: str) -> Path:
    return Path(json.loads(MANIFEST.read_text())["cases"][case]["m2m"])


m1.charm_m2m = charm_m2m   # m1.score's reference: IXI heads too (the manifest's charm run of every case)


def case_seed(case: str) -> int:
    return int(hashlib.sha1(f"r5-sim:{case}".encode()).hexdigest()[:8], 16)


def set_cases(name: str) -> list[str]:
    cases = json.loads(MANIFEST.read_text())["cases"]
    if name == "test":
        return sorted(c for c in json.loads(IXI_SPLIT.read_text())["test"] if c in cases)
    if name == "compare":
        return sorted(splits.wp3_test_cases())
    import r5_warp

    return sorted(r5_warp.sets()[1])[: r5_warp.N_PILOT]


def held_out(cases: list[str], models: list[str]) -> None:
    """Test sets only: no case in the training set of a model scored on it."""
    import torch

    trained = {"release": [R1_SEG_TRAIN, R4_NET], "r5": [R5_SEG_TRAIN, R5_NET]}
    for mdl in models:
        seg_dir, net = trained[mdl]
        seg = {p.name.split("_0000")[0] for p in seg_dir.glob("*_0000.nii.gz")}
        warp = set(torch.load(net, map_location="cpu", weights_only=False).get("train_cases", []))
        bad = sorted(set(cases) & (seg | warp))
        if bad:
            raise SystemExit(f"{mdl}: trained on {bad[:5]}")


def inputs(case: str, arm: str, out: Path) -> list[Path]:
    """The arm's input series for a case (written once, shared by both models)."""
    t1 = charm_m2m(case) / "T1.nii.gz"
    if ARMS[arm] is None:
        return [t1]
    d = out / "inputs" / arm / case
    if not (d / "sim.json").is_file():
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, str(REPO / "scripts/sim_series.py"), str(t1), str(d), *ARMS[arm],
                        "--seed", str(case_seed(case)), "--noise", "0.02"], check=True,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    return sorted(d.glob("[0-9]_*.nii.gz"))


def run_navprep(case: str, images: list[Path], model: str, run: Path) -> dict:
    m2m = run / case / f"m2m_{case}"
    if not (m2m / "navprep.json").is_file():
        if m2m.exists():
            shutil.rmtree(m2m)
        work = run / case / "work"
        env = {k: v for k, v in os.environ.items() if k not in ("HM_NAVPREP_MR_MODEL", "HM_NAVPREP_MR_FOLDS",
                                                                  "HM_WARP_NET")}
        env.update(MODELS[model])
        with open(run / case / "navprep.log", "w") as log:
            rc = subprocess.run([sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr", *map(str, images),
                                 str(m2m), "--work", str(work), "--cap-gb", str(CAP_GB)], env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"{case}: navprep exit {rc} (see {run / case / 'navprep.log'})")
        shutil.rmtree(work, ignore_errors=True)
    return json.loads((m2m / "navprep.json").read_text())


def score_run(case: str, arm: str, model: str, out: Path) -> dict:
    run = out / model / arm
    f = run / case / "eval.json"
    if f.is_file():
        return _with_gpu_s(json.loads(f.read_text()))
    (run / case).mkdir(parents=True, exist_ok=True)
    nav = run_navprep(case, inputs(case, arm, out), model, run)
    r = m1.score(case, run)
    r.update({"arm": arm, "model": model, "navprep": {k: nav.get(k) for k in ("total_s", "steps", "stage_wall_s",
                                                                             "checks", "tier", "combine")}})
    r["peak_mem_gb"] = max(st.get("gpu_peak_gb") or 0 for st in nav["steps"])
    r = _with_gpu_s(r)
    f.write_text(json.dumps(r, indent=1))
    return r


def _with_gpu_s(r: dict) -> dict:
    """The warp's GPU time, as M1 and eval_warpnet report it."""
    if "gpu_s" not in r:
        warp = next(st for st in r["navprep"]["steps"] if st["step"] == "warp")
        r["gpu_s"] = warp.get("gpu_total_s") or warp["s"]
    return r


def summarise(rows: list[dict]) -> dict:
    s = ew.summarise(rows)
    s.pop("gate_rtms", None)
    ok = {k: s[k]["mean"] <= ew.GATE_MEAN and s[k]["max"] <= ew.GATE_MAX for k in GATE_ROWS}
    s["gate"] = {**ok, "no_folding": s["cases_no_fold"] == len(rows),
                 "met": all(ok.values()) and s["cases_no_fold"] == len(rows)}
    nav = [r["navprep"] for r in rows]
    s["navprep"] = {"total_s_mean": float(np.mean([n["total_s"] for n in nav])),
                    "total_s_max": float(np.max([n["total_s"] for n in nav])),
                    "gpu_peak_gb": float(max(r["peak_mem_gb"] for r in rows))}
    return s


def decide(res: dict) -> dict:
    """Rule r5_e2e's adoption test on the test set's 1mm arm: r5 vs release (see the rule's text)."""
    a, b = res["r5"]["1mm"], res["release"]["1mm"]
    means = {k: (a[k]["mean"], b[k]["mean"]) for k in GATE_ROWS}
    maxes = {k: (a[k]["max"], b[k]["max"]) for k in GATE_ROWS}
    checks = {
        "no_folding": a["cases_no_fold"] == a["n_cases"],
        "sum_of_means_lower": sum(x for x, _ in means.values()) < sum(y for _, y in means.values()),
        "no_row_mean_worse_0.05": all(x <= y + 0.05 for x, y in means.values()),
        "no_row_worst_worse_1.0": all(x <= y + 1.0 for x, y in maxes.values()),
        "navprep_mean_le_120s": a["navprep"]["total_s_mean"] <= 120.0,
        "gpu_peak_le_7.5gb": a["navprep"]["gpu_peak_gb"] <= CAP_GB,
    }
    return {"adopt": all(checks.values()), "checks": checks, "means": means, "maxes": maxes,
            "gate_r5_1mm": a["gate"], "gate_by_arm": {arm: res["r5"][arm]["gate"] for arm in res["r5"]}}


def paired(res_rows: dict) -> dict:
    """Per row, r5 - release over the cases both have (1mm arm): mean difference and Wilcoxon p (reported)."""
    from scipy.stats import wilcoxon

    a = {r["case"]: r for r in res_rows["r5"]["1mm"]}
    b = {r["case"]: r for r in res_rows["release"]["1mm"]}
    common = sorted(set(a) & set(b))
    out = {}
    for k in GATE_ROWS:
        d = np.array([a[c][k]["mean"] - b[c][k]["mean"] for c in common])
        out[k] = {"mean_diff": float(d.mean()), "n": len(d),
                  "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d) else None}
    return out


def markdown(name: str, res: dict, decision: dict | None) -> str:
    lines = [f"# R5 end to end: {name} set", "",
             "| model | arm | n | targets mean / worst | rTMS EEG (our skin) | fiducials (our skin) | folded | "
             "navprep s | GPU GB | gate |", "|---|---|---|---|---|---|---|---|---|---|"]
    for mdl, arms in res.items():
        for arm, s in arms.items():
            cells = " | ".join(f"{s[k]['mean']:.2f} / {s[k]['max']:.2f}" for k in GATE_ROWS)
            lines.append(f"| {mdl} | {arm} | {s['n_cases']} | {cells} | {s['n_cases'] - s['cases_no_fold']} | "
                         f"{s['navprep']['total_s_mean']:.0f} | {s['navprep']['gpu_peak_gb']:.2f} | "
                         f"{'MET' if s['gate']['met'] else 'not met'} |")
    if decision:
        lines += ["", f"Rule r5_e2e: **{'ADOPT R5' if decision['adopt'] else 'keep the release'}** ("
                  + ", ".join(f"{k} {'ok' if v else 'fails'}" for k, v in decision["checks"].items()) + ")."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=("pilot", "test", "compare"), required=True)
    ap.add_argument("--rule", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--cases", nargs="*", default=None, help="dry runs only (no --rule)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    models = a.models or (["r5"] if a.set == "compare" else list(MODELS))
    arms = a.arms or (["1mm"] if a.set == "compare" else list(ARMS))
    cases = a.cases or set_cases(a.set)
    rule = None
    if a.set in ("test", "compare"):
        if not a.rule or a.cases or a.models or a.arms:
            raise SystemExit("the test and compare sets run only under the sealed rule, with its models and arms")
        import preregister

        if not preregister.verify(a.rule):
            raise SystemExit(f"rule {a.rule} is not sealed or was changed (scripts/preregister.py verify)")
        rule = {"name": a.rule, **preregister.ledger()[f"{a.rule}.json"]}
        held_out(cases, models)
    out = Path(a.out) / a.set
    out.mkdir(parents=True, exist_ok=True)
    rows: dict = {m: {arm: [] for arm in arms} for m in models}
    for case in cases:
        for arm in arms:
            for mdl in models:
                t0 = time.time()
                r = score_run(case, arm, mdl, out)
                rows[mdl][arm].append(r)
                print(f"{case} {mdl} {arm}: targets {r['targets']['mean']:.2f}/{r['targets']['max']:.2f} "
                      f"fid {r['fid_label_skin']['mean']:.2f}/{r['fid_label_skin']['max']:.2f} "
                      f"J {r['jacobian_min']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    res = {m: {arm: summarise(rs) for arm, rs in by_arm.items()} for m, by_arm in rows.items()}
    decision = decide(res) if a.set == "test" and rule else None
    extra = {}
    if a.set == "test":
        extra["paired_1mm"] = paired(rows)
    if a.set == "compare" and (M1 / "eval.json").is_file():
        extra["release_m1"] = json.loads((M1 / "eval.json").read_text())["summary"]
    (out / "summary.json").write_text(json.dumps({"set": a.set, "rule": rule, "cases": cases, "summary": res,
                                                   "decision": decision, **extra}, indent=1))
    (out / "summary.md").write_text(markdown(a.set, res, decision))
    print(markdown(a.set, res, decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
