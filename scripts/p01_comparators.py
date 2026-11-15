#!/usr/bin/env python3
"""Paper 01 comparators (rule p01_comparators, sealed before any comparator run): simpler routes on the frozen R5
end-to-end inputs, scored with r5_e2e's own scoring (scripts/m1_mri_e2e.py score) against charm on the original
1 mm T1. R5 itself is not re-run or re-scored: its rows are read from runs/r5/e2e/<set>/r5/.

  C1  charm run directly on the thick-slice T1                        arm ax5
  C2  SynthSR (1 mm synthetic MP-RAGE) -> charm                       arm ax5
  C3  R5's segmentation + an ANTs SyN warp instead of R5's warp net   arms 1mm, ax5

Every route is scored the same way as R5: MNI targets through its own MNI2Conform map (unprojected); EEG positions
and fiducials warped by its own maps and projected on the skin of its own labels (label_prep). C3 uses R5's labels,
so its skin is R5's skin.

CPU only, on the E-cores (8-23), at most --jobs at a time (default 6), each job through resource_guard, nice 10;
charm / SynthSR / ANTs at 4 threads. A job that fails or exceeds 6 h is a failure for that head (reported), never
retried with other settings. Resumable: a finished job has eval.json.

  python scripts/p01_comparators.py run --set pilot|test [--jobs 6]   (rtms-gpu env)
  python scripts/p01_comparators.py one --set S --comp C --arm A --case X        (internal: one job)
  python scripts/p01_comparators.py summary --set S
Output: runs/p01_comparators/<set>/<comp>/<arm>/<case>/ (gitignored), <set>/summary.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "runs/p01_comparators"
E2E = REPO / "runs/r5/e2e"
CHARM = Path.home() / ".local/share/mamba/envs/rtms/bin/charm"
CHARM_INI = Path("/mnt/24tb/headmodel/oasis3_mri/charm_4threads.ini")   # the installed charm.ini, threads = 4
SYNTHSR = Path.home() / "Projects/ct-headmodel/vendor/SynthSR"          # commit 9e63935, Apache-2.0
SYNTHSR_PY = Path.home() / ".local/share/mamba/envs/synthsr/bin/python"
PY = sys.executable
CPUS = "8-23"
THREADS = 4
TIMEOUT_S = 6 * 3600
JOBS = {"c1": ("ax5",), "c2": ("ax5",), "c3": ("1mm", "ax5")}
NEED_GB = {"c1": 9, "c2": 12, "c3": 14}
PAD = 100   # voxels around each coordinate grid: the maps stay defined (no zero jumps) far outside the head
ROWS = ("targets", "eeg_label_skin_rtms", "fid_label_skin")


def env_threads() -> dict:
    t = str(THREADS)
    return dict(os.environ, OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, NUMBA_NUM_THREADS=t,
                ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=t, TF_NUM_INTRAOP_THREADS=t, TF_NUM_INTEROP_THREADS="1",
                PYTHONPATH=str(REPO / "src"))


def r5_m2m(set_: str, arm: str, case: str) -> Path:
    return E2E / set_ / "r5" / arm / case / f"m2m_{case}"


def thick_input(set_: str, arm: str, case: str) -> Path:
    f = E2E / set_ / "inputs" / arm / case / "0_ax.nii.gz"
    if not f.is_file():
        raise FileNotFoundError(f)
    return f


# ---------------------------------------------------------------- C1 / C2: charm
def run_charm(case: str, t1: Path, d: Path) -> float:
    m2m = d / f"m2m_{case}"
    if m2m.exists():
        shutil.rmtree(m2m)
    t0 = time.time()
    with open(d / "charm.out", "w") as log:
        rc = subprocess.run([str(CHARM), case, str(t1), "--forcesform", "--forcerun", "--usesettings",
                             str(CHARM_INI)], cwd=d, env=env_threads(), stdout=log, stderr=subprocess.STDOUT,
                            timeout=TIMEOUT_S).returncode
    if rc != 0 or not (m2m / "toMNI/MNI2Conform_nonl.nii.gz").is_file():
        raise RuntimeError(f"charm exit {rc}")
    return time.time() - t0


def run_synthsr(src: Path, dst: Path) -> float:
    t0 = time.time()
    subprocess.run([str(SYNTHSR_PY), str(SYNTHSR / "scripts/predict_command_line.py"), str(src), str(dst), "--cpu",
                    "--threads", str(THREADS), "--model", str(SYNTHSR / "models/SynthSR_v10_210712.h5")],
                   check=True, env=env_threads(), timeout=TIMEOUT_S, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)
    return time.time() - t0


# ---------------------------------------------------------------- C3: R5 labels + ANTs SyN
def _coord_images(shape, affine, pad: int, tmp: Path, tag: str) -> list[Path]:
    """Three images on a padded copy of a grid whose voxel values are the voxel's own world (RAS) x, y, z."""
    shape2 = tuple(int(s) + 2 * pad for s in shape[:3])
    aff2 = affine @ np.array([[1, 0, 0, -pad], [0, 1, 0, -pad], [0, 0, 1, -pad], [0, 0, 0, 1]], float)
    ijk = np.indices(shape2, dtype=np.float32).reshape(3, -1)
    xyz = (aff2[:3, :3] @ ijk + aff2[:3, 3:4]).reshape(3, *shape2)
    out = []
    for k in range(3):
        f = tmp / f"{tag}_{k}.nii.gz"
        nib.save(nib.Nifti1Image(xyz[k].astype(np.float32), aff2), f)
        out.append(f)
    return out


def ants_maps(r5: Path, m2m: Path) -> float:
    """MNI2Conform_nonl / Conform2MNI_nonl from an ANTs SyN registration (antsRegistrationSyN[s], seed 0) of the
    case's T1 to the MNI152 T1 SimNIBS uses, written on R5's own map grids in charm's format (world RAS mm)."""
    import ants
    import simnibs

    mni_t1 = Path(simnibs.__file__).parent / "resources/templates/MNI152_T1_1mm.nii.gz"
    t0 = time.time()
    fixed, moving = ants.image_read(str(mni_t1)), ants.image_read(str(r5 / "T1.nii.gz"))
    reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="antsRegistrationSyN[s]", random_seed=0)
    (m2m / "toMNI").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=m2m.parent) as td:
        tmp = Path(td)
        for name, src_grid, tlist, inv in (
                ("MNI2Conform_nonl", r5 / "T1.nii.gz", reg["fwdtransforms"], None),
                ("Conform2MNI_nonl", mni_t1, reg["invtransforms"], [True, False])):
            ref = nib.load(r5 / "toMNI" / f"{name}.nii.gz")          # R5's grid for this map
            grid = tmp / f"{name}_grid.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(ref.shape[:3], np.float32), ref.affine), grid)
            g = nib.load(src_grid)
            vals = []
            for f in _coord_images(g.shape, g.affine, PAD, tmp, name):
                w = ants.apply_transforms(fixed=ants.image_read(str(grid)), moving=ants.image_read(str(f)),
                                          transformlist=tlist, whichtoinvert=inv, interpolator="linear")
                vals.append(w.numpy().astype(np.float32))
            nib.save(nib.Nifti1Image(np.stack(vals, -1), ref.affine), m2m / "toMNI" / f"{name}.nii.gz")
    return time.time() - t0


def build_c3(r5: Path, m2m: Path) -> float:
    if m2m.exists():
        shutil.rmtree(m2m)
    m2m.mkdir(parents=True)
    for n in ("T1.nii.gz", "label_prep", "segmentation", "surfaces"):
        if (r5 / n).exists():
            os.symlink(r5 / n, m2m / n)
    return ants_maps(r5, m2m)


# ---------------------------------------------------------------- one job
def one(set_: str, comp: str, arm: str, case: str) -> dict:
    import m1_mri_e2e as m1
    import r5_e2e  # noqa: F401  (sets m1.charm_m2m to the manifest's charm run of every case)

    d = OUT / set_ / comp / arm / case
    f = d / "eval.json"
    if f.is_file():
        return json.loads(f.read_text())
    d.mkdir(parents=True, exist_ok=True)
    timing = {}
    try:
        if comp == "c1":
            timing["charm_s"] = run_charm(case, thick_input(set_, arm, case), d)
        elif comp == "c2":
            sr = d / "synthsr_T1.nii.gz"
            timing["synthsr_s"] = run_synthsr(thick_input(set_, arm, case), sr)
            timing["charm_s"] = run_charm(case, sr, d)
        else:
            timing["ants_s"] = build_c3(r5_m2m(set_, arm, case), d / f"m2m_{case}")
        r = m1.score(case, d.parent)
        r.update({"comp": comp, "arm": arm, "ok": True, "timing": timing,
                  "total_s": float(sum(timing.values()))})
    except Exception as e:   # a failure is an outcome, never retried with other settings
        r = {"case": case, "comp": comp, "arm": arm, "ok": False, "error": f"{type(e).__name__}: {e}"[:500],
             "timing": timing}
    f.write_text(json.dumps(r, indent=1))
    return r


def launch(set_: str, comp: str, arm: str, case: str) -> int:
    d = OUT / set_ / comp / arm / case
    if (d / "eval.json").is_file():
        return 0
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(REPO / "scripts/resource_guard.py"), "run", "--need", str(NEED_GB[comp]), "--label",
           f"p01:{comp}:{arm}:{case}", "--", "timeout", str(TIMEOUT_S + 1800), "taskset", "-c", CPUS, "nice", "-n",
           "10", PY, str(Path(__file__).resolve()), "one", "--set", set_, "--comp", comp, "--arm", arm, "--case", case]
    with open(d / "job.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env_threads()).returncode
    if not (d / "eval.json").is_file():   # killed by timeout or the guard: record the failure
        (d / "eval.json").write_text(json.dumps({"case": case, "comp": comp, "arm": arm, "ok": False,
                                                 "error": f"job exit {rc} without a result"}))
    print(f"{time.strftime('%F %T')} {set_} {comp} {arm} {case} exit {rc}", flush=True)
    return rc


def set_cases(set_: str) -> list[str]:
    import r5_e2e

    return r5_e2e.set_cases(set_)


# ---------------------------------------------------------------- summary
def boot_ci(d: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(set_: str) -> dict:
    from scipy.stats import wilcoxon

    cases = set_cases(set_)
    res, md = {}, [f"# Paper 01 comparators ({set_}; rule p01_comparators)", "",
                   "Against charm on the original 1 mm T1; every route scored with r5_e2e's scoring. Paired "
                   "differences = comparator minus R5 on the same heads and arm (positive = comparator worse); 95 % "
                   "bootstrap CI (10,000, seed 0), Wilcoxon signed-rank p. Failed heads are counted, not scored.", "",
                   "| route | arm | heads scored / failed | targets mean / worst | rTMS EEG mean / worst | fiducials mean "
                   "/ worst | diff targets | diff rTMS EEG | diff fiducials | s per head |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
    for comp, arms in JOBS.items():
        for arm in arms:
            rows, fails = [], []
            for c in cases:
                f = OUT / set_ / comp / arm / c / "eval.json"
                if f.is_file():
                    r = json.loads(f.read_text())
                    (rows if r.get("ok") else fails).append(r)
            import eval_warpnet as ew
            for r in rows:
                ew.add_rtms(r)
            r5rows = {c: ew.add_rtms(json.loads((E2E / set_ / "r5" / arm / c / "eval.json").read_text()))
                      for c in cases if (E2E / set_ / "r5" / arm / c / "eval.json").is_file()}
            ent = {"scored": len(rows), "failed": len(fails), "fail_errors": [r.get("error") for r in fails]}
            cells = []
            for k in ROWS:
                a = np.array([r[k]["mean"] for r in rows]) if rows else np.array([])
                w = max((r[k]["max"] for r in rows), default=float("nan"))
                ent[k] = {"mean": float(a.mean()) if len(a) else None, "max": w}
                cells.append(f"{a.mean():.2f} / {w:.2f}" if len(a) else "-")
            diffs = []
            for k in ROWS:
                pair = [(r[k]["mean"], r5rows[r["case"]][k]["mean"]) for r in rows if r["case"] in r5rows]
                if len(pair) >= 2:
                    d = np.array([a - b for a, b in pair])
                    lo, hi = boot_ci(d)
                    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                    ent[f"diff_{k}"] = {"n": len(d), "mean": float(d.mean()), "ci95": [lo, hi], "wilcoxon_p": p}
                    diffs.append(f"{d.mean():+.2f} ({lo:+.2f}, {hi:+.2f}) p={p:.2g}")
                else:
                    diffs.append("-")
            t = [r.get("total_s") for r in rows if r.get("total_s")]
            ent["s_per_head"] = float(np.mean(t)) if t else None
            res[f"{comp}/{arm}"] = ent
            md.append(f"| {comp} | {arm} | {len(rows)} / {len(fails)} | " + " | ".join(cells) + " | "
                      + " | ".join(diffs) + f" | {np.mean(t):.0f} |" if t else
                      f"| {comp} | {arm} | {len(rows)} / {len(fails)} | " + " | ".join(cells) + " | "
                      + " | ".join(diffs) + " | - |")
    md += ["", "R5 on the same heads: runs/r5/e2e/" + set_ + "/summary.md."]
    (OUT / set_).mkdir(parents=True, exist_ok=True)
    (OUT / set_ / "summary.json").write_text(json.dumps(res, indent=1))
    (OUT / set_ / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return res


def run(set_: str, jobs: int) -> int:
    cases = set_cases(set_)
    work = [(comp, arm, c) for c in cases for comp, arms in JOBS.items() for arm in arms]
    # the long charm jobs first, ANTs fills the gaps
    work.sort(key=lambda w: (w[0] == "c3", w[2]))
    print(f"{time.strftime('%F %T')} {set_}: {len(work)} jobs, {jobs} at a time on CPUs {CPUS}", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        list(ex.map(lambda w: launch(set_, *w), work))
    summary(set_)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("run", "one", "summary"))
    ap.add_argument("--set", required=True, choices=("pilot", "test"))
    ap.add_argument("--comp", choices=tuple(JOBS))
    ap.add_argument("--arm")
    ap.add_argument("--case")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    if a.cmd == "one":
        r = one(a.set, a.comp, a.arm, a.case)
        return 0 if r.get("ok") else 1
    if a.cmd == "summary":
        summary(a.set)
        return 0
    return run(a.set, a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
