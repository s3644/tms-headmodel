"""R6 end-to-end score (docs/r6_plan_draft.md; rule r6_e2e): the MRI fast route with the R6 models vs R5 (the live
release's models), on the same inputs, against charm on the original 1 mm T1. Built on scripts/r5_e2e.py (same
arms, scorer, folding count and per-run files); this file sets the models, the case sets, the decision and runs
cases in parallel.

Models (navprep exactly as the locator runs it):
  r5   Dataset602 folds 0-4 + runs/r5/warp/final/last.pt  (= release 20261105-e519286)
  r6   Dataset603 folds 0-4 + runs/r6/warp/final/last.pt
Sets: pilot (the first 5 sorted R6 warp validation heads: machinery), test (the OASIS-3 test heads of the sealed
r6_oasis3_split whose charm run finished: their one use), compare (the 57 WP3 test cases, 1mm, r6 only, against
R5's rows in runs/r5/e2e/compare, never re-run).
Decision (test set, 1mm, r6 vs r5): r5_e2e's six checks; plus, on the compare set, no gate-row mean worse than
R5's by > 0.10 mm and no worst point worse by > 1.0 mm. The test set writes checks 1-6; the compare set writes
check 7 and the final decision (runs/r6/e2e/decision.{json,md}). The test set is also summarised per OASIS-3
stratum (normal / decline), reported only.
--workers N scores N cases at a time (each case: all arms, both models, one navprep subprocess at a time per
worker). navprep's wall time is therefore measured under N-way load: conservative for the <= 120 s check.

Usage (rtms-gpu env):
  PYTHONPATH=src python scripts/r6_e2e.py --set pilot|test|compare [--rule r6_e2e] [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R6 = Path(os.environ.get("HM_R6_RUNS", REPO / "runs/r6"))
OUT = R6 / "e2e"
# r5_e2e and r5_warp read their paths from the environment at import: point them at R6 (a smoke run sets others)
os.environ.setdefault("HM_R5_MANIFEST", str(REPO / "data/manifests/r6.json"))
os.environ.setdefault("HM_R5_RUNS", str(R6))
os.environ.setdefault("HM_R5_DATASET", "Dataset603_MRHeadR6")
os.environ.setdefault("HM_R5_TRAINER_DIR", "nnUNetTrainerHMThick430__nnUNetPlans__3d_fullres")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import r5_e2e as e  # noqa: E402
from headmodel import splits  # noqa: E402

OASIS_SPLIT = REPO / "runs/prereg/r6_oasis3_split.json"
R5_SEG = REPO / "data/nnunet/results/Dataset602_MRHeadR5/nnUNetTrainerHMThick__nnUNetPlans__3d_fullres"
R5_NET = REPO / "runs/r5/warp/final/last.pt"
R6_SEG = Path(os.environ.get("HM_R6_SEG", REPO / "data/nnunet/results" / os.environ["HM_R5_DATASET"] / os.environ["HM_R5_TRAINER_DIR"]))
R6_NET = Path(os.environ.get("HM_R6_NET", R6 / "warp/final/last.pt"))
R6_SEG_TRAIN = Path(os.environ.get("HM_R6_SEG_TRAIN", REPO / "data/nnunet/raw" / os.environ["HM_R5_DATASET"] / "imagesTr"))
R5_COMPARE = REPO / "runs/r5/e2e/compare/summary.json"
MODELS = {
    "r5": {"HM_NAVPREP_MR_MODEL": str(R5_SEG), "HM_NAVPREP_MR_FOLDS": "0,1,2,3,4", "HM_WARP_NET": str(R5_NET)},
    "r6": {"HM_NAVPREP_MR_MODEL": str(R6_SEG), "HM_NAVPREP_MR_FOLDS": os.environ.get("HM_R6_FOLDS", "0,1,2,3,4"),
           "HM_WARP_NET": str(R6_NET)},
}
e.MODELS = MODELS
ARMS, GATE_ROWS, CAP_GB = e.ARMS, e.GATE_ROWS, e.CAP_GB
NEW, OLD = "r6", "r5"
COMPARE_MEAN_TOL, COMPARE_MAX_TOL = 0.10, 1.0


def manifest_cases() -> dict:
    return json.loads(Path(os.environ["HM_R5_MANIFEST"]).read_text())["cases"]


def set_cases(name: str) -> list[str]:
    cases = manifest_cases()
    if name == "test":
        return sorted(c for c in json.loads(OASIS_SPLIT.read_text())["test"] if c in cases)
    if name == "compare":
        return sorted(splits.wp3_test_cases())
    import r5_warp

    return sorted(r5_warp.sets()[1])[: r5_warp.N_PILOT]


def held_out(cases: list[str], models: list[str]) -> None:
    """Test sets only: no case in the training set of a model scored on it."""
    import torch

    trained = {OLD: [e.R5_SEG_TRAIN, R5_NET], NEW: [R6_SEG_TRAIN, R6_NET]}
    for mdl in models:
        seg_dir, net = trained[mdl]
        seg = {p.name.split("_0000")[0] for p in seg_dir.glob("*_0000.nii.gz")}
        warp = set(torch.load(net, map_location="cpu", weights_only=False).get("train_cases", []))
        bad = sorted(set(cases) & (seg | warp))
        if bad:
            raise SystemExit(f"{mdl}: trained on {bad[:5]}")
        if not seg:
            raise SystemExit(f"{mdl}: no training images found under {seg_dir}")


def case_job(case: str, arms: list[str], models: list[str], out: str) -> dict:
    """One case through every arm and model (a worker); returns {(model, arm): eval row}."""
    rows = {}
    for arm in arms:
        for mdl in models:
            t0 = time.time()
            r = e.score_run(case, arm, mdl, Path(out))
            rows[(mdl, arm)] = r
            print(f"{case} {mdl} {arm}: targets {r['targets']['mean']:.2f}/{r['targets']['max']:.2f} "
                  f"fid {r['fid_label_skin']['mean']:.2f}/{r['fid_label_skin']['max']:.2f} "
                  f"J {r['jacobian_min']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    return rows


def decide_test(res: dict) -> dict:
    """Checks 1-6 (r5_e2e's, r6 vs r5) on the test set's 1mm arm."""
    a, b = res[NEW]["1mm"], res[OLD]["1mm"]
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
    return {"checks_1_6": checks, "all_1_6": all(checks.values()), "means": means, "maxes": maxes,
            "gate_r6_1mm": a["gate"], "gate_by_arm": {arm: res[NEW][arm]["gate"] for arm in res[NEW]},
            "note": "adoption also needs check 7 (compare set); see decision.json"}


def decide_compare(res: dict) -> dict:
    """Check 7: r6 on the 57 (1mm) not worse than R5's rows there by > 0.10 mm mean / > 1.0 mm worst point."""
    base = json.loads(R5_COMPARE.read_text())["summary"]["r5"]["1mm"]
    a = res[NEW]["1mm"]
    means = {k: (a[k]["mean"], base[k]["mean"]) for k in GATE_ROWS}
    maxes = {k: (a[k]["max"], base[k]["max"]) for k in GATE_ROWS}
    ok = all(x <= y + COMPARE_MEAN_TOL for x, y in means.values()) and all(x <= y + COMPARE_MAX_TOL for x, y in maxes.values())
    return {"check_7_compare_not_worse": ok, "means": means, "maxes": maxes, "r5_rows_from": str(R5_COMPARE),
            "cases_pass_all": (a.get("cases_pass_all"), base.get("cases_pass_all"))}


def final_decision(out_root: Path) -> dict:
    t = json.load(open(out_root / "test/summary.json"))["decision"]
    c = json.load(open(out_root / "compare/summary.json"))["decision"]
    d = {"adopt": bool(t["all_1_6"] and c["check_7_compare_not_worse"]), "test": t, "compare": c}
    (out_root / "decision.json").write_text(json.dumps(d, indent=1))
    lines = [f"# R6 decision: **{'ADOPT R6' if d['adopt'] else 'keep R5'}**", "",
             "Test set (OASIS-3, 1mm), r6 vs r5: " + ", ".join(f"{k} {'ok' if v else 'FAILS'}" for k, v in t["checks_1_6"].items()),
             "", f"Compare set (57, 1mm), r6 vs R5's rows: check 7 {'ok' if c['check_7_compare_not_worse'] else 'FAILS'} "
             + "(" + "; ".join(f"{k} {x:.2f}/{mx[0]:.2f} vs {y:.2f}/{mx[1]:.2f}" for (k, (x, y)), mx in zip(c["means"].items(), c["maxes"].values())) + ")",
             "", "A new release needs the user's go (docs/r6_plan_draft.md)."]
    (out_root / "decision.md").write_text("\n".join(lines) + "\n")
    return d


def paired(rows: dict) -> dict:
    """Per row, r6 - r5 over the cases both have (1mm): mean difference and Wilcoxon p (reported)."""
    from scipy.stats import wilcoxon

    a = {r["case"]: r for r in rows[NEW]["1mm"]}
    b = {r["case"]: r for r in rows[OLD]["1mm"]}
    common = sorted(set(a) & set(b))
    out = {}
    for k in GATE_ROWS:
        d = np.array([a[c][k]["mean"] - b[c][k]["mean"] for c in common])
        out[k] = {"mean_diff": float(d.mean()), "n": len(d),
                  "wilcoxon_p": float(wilcoxon(d).pvalue) if len(d) > 5 and np.any(d) else None}
    return out


def by_stratum(rows: dict) -> dict:
    strat = {c: i.get("stratum", "unknown") for c, i in manifest_cases().items()}
    out: dict = {}
    for mdl, arms in rows.items():
        for arm, rs in arms.items():
            for s in sorted({strat.get(r["case"], "unknown") for r in rs}):
                sub = [r for r in rs if strat.get(r["case"], "unknown") == s]
                out.setdefault(s, {}).setdefault(mdl, {})[arm] = e.summarise(sub)
    return out


def markdown(name: str, res: dict, decision: dict | None, strata: dict | None) -> str:
    md = e.markdown(name, res, None).replace("# R5 end to end", "# R6 end to end")
    if strata:
        for s, models in strata.items():
            md += f"\n## stratum: {s}\n\n" + e.markdown(name, models, None).split("\n", 2)[2]
    if decision and "checks_1_6" in decision:
        md += "\nChecks 1-6 (r6 vs r5, 1mm): " + ", ".join(f"{k} {'ok' if v else 'fails'}" for k, v in decision["checks_1_6"].items()) + ". Check 7 waits for the compare set.\n"
    if decision and "check_7_compare_not_worse" in decision:
        md += f"\nCheck 7 (compare set, r6 vs R5's rows): {'ok' if decision['check_7_compare_not_worse'] else 'fails'}.\n"
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=("pilot", "test", "compare"), required=True)
    ap.add_argument("--rule", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--cases", nargs="*", default=None, help="dry runs only (no --rule)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    models = a.models or ([NEW] if a.set == "compare" else [OLD, NEW])
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
    with ProcessPoolExecutor(max_workers=max(1, a.workers), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(case_job, c, arms, models, str(out)) for c in cases]
        for f in futs:                       # in case order: a failure stops the set (resumable)
            for (mdl, arm), r in f.result().items():
                rows[mdl][arm].append(r)
    res = {m: {arm: e.summarise(rs) for arm, rs in by_arm.items()} for m, by_arm in rows.items()}
    strata = by_stratum(rows) if a.set == "test" else None
    decision = None
    if a.set == "test" and rule:
        decision = decide_test(res)
    if a.set == "compare" and rule:
        decision = decide_compare(res)
    extra = {}
    if a.set == "test" and OLD in rows and NEW in rows:
        extra["paired_1mm"] = paired(rows)
    if strata:
        extra["by_stratum"] = strata
    if a.set == "compare" and R5_COMPARE.is_file():
        extra["r5_compare"] = json.loads(R5_COMPARE.read_text())["summary"]
    (out / "summary.json").write_text(json.dumps({"set": a.set, "rule": rule, "cases": cases, "summary": res,
                                                   "decision": decision, **extra}, indent=1))
    (out / "summary.md").write_text(markdown(a.set, res, decision, strata))
    print(markdown(a.set, res, decision, strata))
    if a.set == "compare" and rule and (Path(a.out) / "test/summary.json").is_file():
        d = final_decision(Path(a.out))
        print(f"DECISION: {'ADOPT R6' if d['adopt'] else 'keep R5'} ({Path(a.out) / 'decision.md'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
