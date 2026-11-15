"""One T1 MRI (or several thick series of one visit) -> a SimNIBS-format m2m folder, with every step timed.

Runs src/headmodel/navprep.py's stages in their envs (combine + segment: nnunet; warp + cortex: rtms-gpu) and
writes <m2m>/navprep.json: provisional (always, until charm confirms), the input checks, each step's
seconds and GPU peak, and each stage's wall time including Python, CUDA and model start-up (the cold cost a
warm GPU worker would save). Standard library only, so any program can call it from any env.

One prep at a time per machine: both stages run under an exclusive lock (HM_NAVPREP_LOCK), so two
patients never share one 8 GB card; the wait is recorded as queue_wait_s.

Several MRI series of one visit (e.g. 5 mm sagittal + axial T1) are combined first (stage `combine`).
navprep.json's `tier` says what the input was:
  1mm             one series, slices <= THICK_MM (2.5 mm);
  thick           one series with thicker slices;
  thick-combined  two or more series, combined.
Which points may be navigated per tier is set by the R5 scoring rule; until then all are provisional.

Usage:
  python scripts/navprep.py --modality mr IMAGE [IMAGE ...] M2M_DIR [--work DIR] [--cap-gb 7.5] [--no-cortex]
Exit codes: 0 done, 3 unusable input (message on stderr), 1 any other failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# A release (scripts/make_release.sh) names its models in RELEASE.json "env" (paths relative to the release, folds as
# given); an environment variable that is already set still wins. The stages inherit os.environ.
_REL = REPO / "RELEASE.json"
if _REL.is_file():
    for _k, _v in json.loads(_REL.read_text()).get("env", {}).items():
        os.environ.setdefault(_k, _v if _k.endswith("_FOLDS") else str(REPO / _v))
ENVS = Path(os.environ.get("MAMBA_ROOT_PREFIX", Path.home() / ".local/share/mamba")) / "envs"
SEGMENT_PY = os.environ.get("HM_NAVPREP_SEGMENT_PY", str(ENVS / "nnunet/bin/python"))
WARP_PY = os.environ.get("HM_NAVPREP_WARP_PY", str(ENVS / "rtms-gpu/bin/python"))
VERSION = 2   # 2: several MRI series (combine stage), tier
THICK_MM = 2.5   # = navprep.THICK_MM (this script imports nothing from headmodel)
LOCK = Path(os.environ.get("HM_NAVPREP_LOCK", "/dev/shm/headmodel_navprep.lock"))


def _stage(py: str, args: list[str], log) -> tuple[int, float]:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    t0 = time.perf_counter()
    rc = subprocess.run([py, "-m", "headmodel.navprep", *args], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return rc, round(time.perf_counter() - t0, 3)


def tier(images: list[Path], voxel_mm: list[float]) -> str:
    if len(images) > 1:
        return "thick-combined"
    return "1mm" if max(voxel_mm) <= THICK_MM else "thick"


def run(images: list[Path], modality: str, m2m: Path, work: Path, cap_gb: float | None, cortex: bool, log,
        queue_wait_s: float = 0.0) -> int:
    t0 = time.perf_counter()
    common = ["--cap-gb", str(cap_gb)] if cap_gb else []
    walls, comb = {}, None
    image = images[0]
    if len(images) > 1:
        if modality != "mr":
            print("navprep: several series are combined for MRI only", file=log, flush=True)
            return 3
        rc, walls["combine"] = _stage(SEGMENT_PY, ["combine", *map(str, images), "--work", str(work), *common], log)
        if rc != 0:
            return rc
        comb = json.loads((work / "combine.json").read_text())
        image = work / "combined.nii.gz"
    rc, seg_s = _stage(SEGMENT_PY, ["segment", str(image), "--modality", modality, "--work", str(work), *common], log)
    if rc != 0:
        return rc
    rc, warp_s = _stage(WARP_PY, ["warp", "--work", str(work), "--m2m", str(m2m), *common,
                                  *([] if cortex else ["--no-cortex"])], log)
    if rc != 0:
        return rc
    seg, wrp = (json.loads((work / f).read_text()) for f in ("segment.json", "warp.json"))
    checks = seg["checks"]
    if comb:
        thick = max(x["spacing_mm"] for x in comb["series"])
        checks = {**checks, "warnings": [f"combined from {len(images)} series (slices up to {thick:.1f} mm): "
                                         "targets less certain", *checks.get("warnings", [])]}
    info = {
        "version": VERSION,
        "provisional": True,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modality": modality,
        "input": ", ".join(i.name for i in images),
        "tier": tier(images, seg["checks"].get("voxel_mm", [0.0]) if not comb else [0.0]),
        "checks": checks,
        "combine": comb,
        "models": {"segment": {"model": Path(seg["model"]).parent.name, "folds": seg["folds"]},
                   "warp": "WP3b round 4 (register.adopted_setting)", "cortex": "TopoFit" if cortex else None},
        "cap_gb": cap_gb,
        "steps": (comb["steps"] if comb else []) + seg["steps"] + wrp["steps"],
        "stage_wall_s": {**walls, "segment": seg_s, "warp": warp_s},
        "total_s": round(time.perf_counter() - t0, 3),
        "queue_wait_s": round(queue_wait_s, 3),
    }
    tmp = m2m / "navprep.json.part"
    tmp.write_text(json.dumps(info, indent=1))
    os.replace(tmp, m2m / "navprep.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", type=Path, nargs="+", metavar="IMAGE")
    ap.add_argument("m2m", type=Path)
    ap.add_argument("--modality", choices=("mr",), required=True)
    ap.add_argument("--work", type=Path, default=None, help="kept when given; a temporary folder otherwise")
    ap.add_argument("--cap-gb", type=float, default=None)
    ap.add_argument("--no-cortex", action="store_true")
    a = ap.parse_args(argv)
    a.m2m.mkdir(parents=True, exist_ok=True)
    work = a.work or Path(tempfile.mkdtemp(prefix="navprep_"))
    try:
        t0 = time.perf_counter()
        with open(LOCK, "a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("navprep: waiting for another prep to finish", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
            rc = run([i.resolve() for i in a.images], a.modality, a.m2m.resolve(), work.resolve(), a.cap_gb, not a.no_cortex,
                     sys.stdout, time.perf_counter() - t0)
    finally:
        if a.work is None:
            shutil.rmtree(work, ignore_errors=True)
    if rc == 0:
        print(json.dumps(json.loads((a.m2m / "navprep.json").read_text())["steps"]))
    return rc if rc in (0, 3) else 1


if __name__ == "__main__":
    sys.exit(main())
