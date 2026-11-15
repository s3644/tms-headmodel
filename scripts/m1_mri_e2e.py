"""PLAN 10.4 track 1, M1: the MRI fast route end to end on the held-out cases, scored against charm.

For each case, the route runs exactly as the locator runs it: scripts/navprep.py --modality mr on charm's input T1
(<ct-headmodel>/runs/<case>/m2m_ref/T1.nii.gz). That means:
  - LPS + 1 mm grid;
  - nnU-Net Dataset601 fold 0 labels;
  - the WP3b round-4 warp;
  - TopoFit;
  - 7.5 GB GPU cap.
Then, against charm's MRI model of the same case:
  targets     simnibs.mni2subject_coords of the 10 MNI targets, ours vs charm's (unprojected);
  EEG cap     EEG10-10_UI_Jurak_2007 warped with our toMNI and projected on OUR skin (the outer skin of our
              predicted labels, built with the locator's locator_ready.skin_surface, as the locator delivers
              it) vs charm's eeg_positions/*.csv; also projected on charm's skin, to separate skin from warp;
  fiducials   the same, for Nz, Iz, LPA, RPA;
plus the written maps' minimum Jacobian (folding), the navprep step timings and the GPU peak.
eval_warpnet's "label skin" rows hold OUR skin here.

Each case is compared with the round-4 test run on charm's own labels (runs/wp3b/warpnet_r4/eval_test), so the
segmenter's share of the error is visible.

Held-out only: the 57 frozen WP3 test cases, none in Dataset601's training set (checked) or the warp net's.
Judged by a rule sealed beforehand (--rule, scripts/preregister.py).

Usage (rtms-gpu env):
  PYTHONPATH=src python scripts/m1_mri_e2e.py --rule m1_mri_e2e [--cases ...] [--out runs/m1_mri_e2e]
  (--cases with validation cases and no --rule: a dry run of the machinery; test cases need --rule)
Writes <out>/<case>/{m2m_<case>/, eval.json, projected.json, navprep.log} (resumable), <out>/eval.json and
<out>/eval.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import splits, warpnet  # noqa: E402
import eval_register as ev  # noqa: E402
import eval_warpnet as ew  # noqa: E402

CT_RUNS = Path.home() / "Projects/ct-headmodel/runs"
OUT = REPO / "runs/m1_mri_e2e"
R4_TEST = REPO / "runs/wp3b/warpnet_r4/eval_test"
NNUNET_TRAIN = REPO / "data/nnunet/raw/Dataset601_MRHead/imagesTr"
CAP_GB = 7.5
# the gate (rule wp3_gate_rtms_set thresholds) on what the locator delivers: targets, the rTMS EEG set and the
# fiducials projected on our own skin; each mean <= 2 mm and worst point <= 5 mm, plus no folding
GATE_ROWS = ("targets", "eeg_label_skin_rtms", "fid_label_skin")


def charm_m2m(case: str) -> Path:
    return CT_RUNS / case / "m2m_ref"


def held_out(cases: list[str]) -> None:
    trained = {p.name.split("_0000")[0] for p in NNUNET_TRAIN.glob("*_0000.nii.gz")}
    bad = sorted(set(cases) & trained)
    if bad:
        raise SystemExit(f"in Dataset601's training set: {bad[:5]}")
    ck = __import__("torch").load(register_net(), map_location="cpu", weights_only=False)
    bad = sorted(set(cases) & set(ck.get("train_cases", [])))
    if bad:
        raise SystemExit(f"in the warp net's training set: {bad[:5]}")


def register_net() -> str:
    from headmodel import register

    return register.adopted_setting()[0].init_warp


def run_navprep(case: str, out: Path) -> dict:
    """navprep.py on the case's input T1 -> <out>/<case>/m2m_<case>; returns its navprep.json."""
    m2m = out / case / f"m2m_{case}"
    if not (m2m / "navprep.json").is_file():
        if m2m.exists():
            shutil.rmtree(m2m)
        work = out / case / "work"
        with open(out / case / "navprep.log", "w") as log:
            rc = subprocess.run([sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr",
                                 str(charm_m2m(case) / "T1.nii.gz"), str(m2m), "--work", str(work),
                                 "--cap-gb", str(CAP_GB)], stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"{case}: navprep exit {rc} (see {out / case / 'navprep.log'})")
        shutil.rmtree(work, ignore_errors=True)
    return json.loads((m2m / "navprep.json").read_text())


def our_skin(case: str, out: Path) -> Path:
    """Our skin as eval_register's label-skin cache: the locator's skin surface of our predicted labels."""
    f = out / case / "our_skin.npz"
    if not f.is_file():
        if str(ev.LOCATOR_BACKEND) not in sys.path:
            sys.path.insert(0, str(ev.LOCATOR_BACKEND))
        from app.locator_ready import skin_surface

        img = nib.load(out / case / f"m2m_{case}" / "label_prep/tissue_labeling_upsampled.nii.gz")
        v, f_ = skin_surface(np.asarray(img.dataobj), img.affine)
        np.savez_compressed(f, v=v, f=f_)
    return f


def score(case: str, out: Path) -> dict:
    import simnibs

    m2m, ref = out / case / f"m2m_{case}", charm_m2m(case)
    pts, nt = warpnet.mni_points()
    ours = np.asarray(simnibs.mni2subject_coords(pts, str(m2m)))
    ref_f = REPO / "runs/wp3b/charm_ref" / f"{case}.json"
    if ref_f.is_file() and np.allclose(json.loads(ref_f.read_text())["mni"], pts):
        charm = np.asarray(json.loads(ref_f.read_text())["charm"])
    else:
        charm = np.asarray(simnibs.mni2subject_coords(pts, str(ref)))
    e = np.linalg.norm(ours - charm, axis=1)
    back = np.asarray(simnibs.subject2mni_coords(ours, str(m2m)))
    t1 = nib.load(m2m / "T1.nii.gz")
    ijk = nib.affines.apply_affine(np.linalg.inv(t1.affine), ours)
    inside = ((ijk >= 0) & (ijk <= np.array(t1.shape[:3]) - 1)).all(1)
    st = lambda a: {"mean": float(a.mean()), "max": float(a.max())}   # noqa: E731
    r = {"case": case, "targets": st(e[:nt]), "eeg": st(e[nt:]), "err_targets": e[:nt].tolist(),
         "err_eeg": e[nt:].tolist(), "round_trip_max_mm": float(np.linalg.norm(back - pts, axis=1)[inside].max())}
    pr = ev.eval_projected(out / case, ref, our_skin(case, out))
    (out / case / "projected.json").write_text(json.dumps(pr))
    for k, (g, c) in ew.PROJ.items():
        a = np.asarray(pr["groups"][g][c])
        r[k] = st(a)
        r[f"err_{k}"] = a.tolist()
    r["jacobian_written"] = ev.written_jacobian(m2m)
    r["jacobian_min"] = r["jacobian_written"]["min"]
    return r


def summarise(rows: list[dict]) -> dict:
    s = ew.summarise(rows)   # also adds the rTMS-set rows in place
    s.pop("gate_rtms", None)
    rows_ok = {k: s[k]["mean"] <= ew.GATE_MEAN and s[k]["max"] <= ew.GATE_MAX for k in GATE_ROWS}
    s["gate"] = {**rows_ok, "no_folding": s["cases_no_fold"] == len(rows),
                 "met": all(rows_ok.values()) and s["cases_no_fold"] == len(rows)}
    s["cases_pass_all"] = int(sum(all(r[k]["mean"] <= ew.GATE_MEAN and r[k]["max"] <= ew.GATE_MAX for k in GATE_ROWS)
                                  and r["jacobian_min"] > 0 for r in rows))
    nav = [r["navprep"] for r in rows]
    s["navprep"] = {"total_s_mean": float(np.mean([n["total_s"] for n in nav])),
                    "total_s_max": float(np.max([n["total_s"] for n in nav])),
                    "gpu_peak_gb": float(max([st.get("gpu_peak_gb") or 0 for n in nav for st in n["steps"]] or [0])),
                    "warnings": int(sum(bool(n.get("checks", {}).get("warnings")) for n in nav))}
    # the segmenter's share: the same cases, round 4 on charm's own labels
    paired = [(r, json.loads((R4_TEST / r["case"] / "eval.json").read_text())) for r in rows
              if (R4_TEST / r["case"] / "eval.json").is_file()]
    if paired:
        ew.summarise([p for _, p in paired])   # rTMS rows on the round-4 rows
        s["vs_charm_labels"] = {k: {"e2e": float(np.mean([a[k]["mean"] for a, _ in paired])),
                                    "charm_labels": float(np.mean([b[k]["mean"] for _, b in paired])),
                                    "n": len(paired)}
                                for k in ("targets", "eeg_rtms", "eeg_charm_skin_rtms", "fid_charm_skin")
                                if all(k in b for _, b in paired)}
    return s


def markdown(s: dict, rows: list[dict], title: str) -> str:
    md = ew.markdown({k: v for k, v in s.items() if k != "gate_rtms"}, rows, title)
    md = md.replace("projected on label skin", "projected on OUR skin (predicted labels)")
    g = s["gate"]
    extra = [f"Gate (targets, rTMS EEG set on our skin, fiducials on our skin: each <= {ew.GATE_MEAN:g} mm mean and "
             f"<= {ew.GATE_MAX:g} mm worst point; no folding): **{'MET' if g['met'] else 'NOT met'}** ("
             + ", ".join(f"{k} {'ok' if v else 'fails'}" for k, v in g.items() if k != "met")
             + f"). Cases passing all: {s['cases_pass_all']}/{s['n_cases']}.", "",
             f"navprep: {s['navprep']['total_s_mean']:.1f} s per case (max {s['navprep']['total_s_max']:.1f}), "
             f"GPU peak {s['navprep']['gpu_peak_gb']:.2f} GB, cases with input warnings {s['navprep']['warnings']}."]
    if "vs_charm_labels" in s:
        extra += ["", "Segmenter's share (same cases; round 4 on charm's own labels vs our predicted labels), mean of "
                  "case means:", "", "| points | charm labels | our labels (end to end) |", "|---|---|---|"]
        extra += [f"| {k} | {v['charm_labels']:.2f} | {v['e2e']:.2f} |" for k, v in s["vs_charm_labels"].items()]
    head, tail = md.split("\n\nBoth pass", 1)
    return head + "\n\n" + "\n".join(extra) + "\n\nBoth pass" + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rule", default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    test = splits.wp3_test_cases()
    cases = a.cases or test
    rule = None
    if a.rule:
        import preregister

        if not preregister.verify(a.rule):
            raise SystemExit(f"rule {a.rule} is not sealed or was changed (scripts/preregister.py verify)")
        rule = {"name": a.rule, **preregister.ledger()[f"{a.rule}.json"]}
    elif set(cases) & set(test):
        raise SystemExit("test cases need a sealed --rule")
    if rule:
        held_out(cases)
    else:
        print("dry run: no held-out check, the results are not a score", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        f = out / case / "eval.json"
        if f.is_file():
            rows.append(json.loads(f.read_text()))
            continue
        (out / case).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        nav = run_navprep(case, out)
        r = score(case, out)
        r["navprep"] = {k: nav.get(k) for k in ("total_s", "steps", "stage_wall_s", "checks", "queue_wait_s")}
        warp = next(st for st in nav["steps"] if st["step"] == "warp")
        r["gpu_s"] = warp.get("gpu_total_s") or warp["s"]   # the warp's GPU time, as eval_warpnet reports it
        r["peak_mem_gb"] = max(st.get("gpu_peak_gb") or 0 for st in nav["steps"])
        f.write_text(json.dumps(r, indent=1))
        rows.append(r)
        print(f"{case}: targets {r['targets']['mean']:.2f}/{r['targets']['max']:.2f}  "
              f"fid our skin {r['fid_label_skin']['mean']:.2f}/{r['fid_label_skin']['max']:.2f}  "
              f"J {r['jacobian_min']:.3f}  navprep {nav['total_s']:.0f} s  ({time.time() - t0:.0f} s)", flush=True)
    s = summarise(rows)
    s["rule"] = rule
    title = f"M1: MRI fast route end to end, {len(rows)} {'test' if set(cases) <= set(test) else 'dry-run'} cases"
    (out / "eval.json").write_text(json.dumps({"summary": s, "cases": rows}, indent=1))
    (out / "eval.md").write_text(markdown(s, rows, title))
    print(markdown(s, rows, title).split("\n\n| case")[0])


if __name__ == "__main__":
    main()
