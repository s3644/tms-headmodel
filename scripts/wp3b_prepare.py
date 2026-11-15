"""WP3b: shared inputs for model selection on a VALIDATION split of the train cases (no test case).

  runs/wp3b/val_cases.json                     the 30 train cases with the smallest sha1(case id), sorted, on the
                                               134-case WP3 train snapshot (fixed file, authoritative: splits.val_cases)
  runs/wp3b/labels_cache/<case>.npz            charm's labels block-majority'd to the T1 grid ("lab")
  runs/wp3b/template_trainminusval/            label template + affine calibration from train - val
  runs/wp3b/charm_ref/<case>.json              charm's mni2subject_coords of the 10 targets + EEG cap

Everything is resumable and written atomically (tmp file / dir, then rename), so parallel explorers can
share it. Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_prepare.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from headmodel import dataset, register, splits  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3b"
VAL_FILE = splits.VAL_FILE
LABELS = OUT / "labels_cache"
TEMPLATE = OUT / "template_trainminusval"
CHARM_REF = OUT / "charm_ref"


def load_manifest() -> dict:
    f = REPO / "data/manifests/round1.json"
    return json.loads(f.read_text()) if f.is_file() else dataset.manifest(CT_ROOT)


def val_cases(man: dict) -> list[str]:
    """runs/wp3b/val_cases.json is authoritative (headmodel.splits.val_cases): it was made by the sha1
    rule on the 134-case WP3 train snapshot (val_cases_provenance.json); the grown manifest only warns."""
    return splits.val_cases(man)


def _atomic_text(f: Path, text: str):
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, f)


def labels(case: str, m2m: str) -> np.ndarray:
    f = LABELS / f"{case}.npz"
    if f.is_file():
        return np.load(f)["lab"]
    m2m = Path(m2m)
    t1, up = nib.load(m2m / "T1.nii.gz"), nib.load(m2m / dataset.SEG_LABELS)
    lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    LABELS.mkdir(parents=True, exist_ok=True)
    tmp = LABELS / f".{case}.{os.getpid()}.npz"
    np.savez_compressed(tmp, lab=lab)
    os.replace(tmp, f)
    return lab


def _labels_job(args):
    case, m2m = args
    t0 = time.time()
    labels(case, m2m)
    return case, time.time() - t0


def build_template_minus_val(train: dict, val: list[str], device: str):
    if (TEMPLATE / register.CALIBRATION_FILE).is_file():
        return
    import torch

    fit = {c: i for c, i in train.items() if c not in set(val)}
    tmp = OUT / f".template_trainminusval.{os.getpid()}"
    register.build_template(fit, tmp)
    params = register.Params(svf_stages=(), calibrate=False, frame="labels")
    pairs = {}
    for case, info in sorted(fit.items()):
        m2m = Path(info["m2m"])
        mat = m2m / "segmentation/coregistrationMatrices.mat"
        if not mat.is_file():
            continue
        t1 = nib.load(m2m / "T1.nii.gz")
        lab = labels(case, info["m2m"])
        r = register.register(t1, nib.Nifti1Image(lab, t1.affine), None, device=device, params=params,
                              template_dir=tmp, write=False)
        pairs[case] = (np.asarray(r["affine"]), scipy.io.loadmat(mat)["worldToWorldTransformMatrix"])
        torch.cuda.empty_cache()
    cal = register.fit_calibration(pairs, meta={
        "what": "ridge regression: our affine -> charm's worldToWorldTransformMatrix frame (matrix logs)",
        "split": "train minus the 30 validation cases (runs/wp3b/val_cases.json)",
        "licence": "research only: derived from CC BY-NC 4.0 data", "built": time.strftime("%Y-%m-%d %H:%M")})
    (tmp / register.CALIBRATION_FILE).write_text(json.dumps(cal, indent=1))
    if TEMPLATE.exists():   # someone else finished first
        shutil.rmtree(tmp)
    else:
        os.replace(tmp, TEMPLATE)
    print("calibration LOO (mm):", cal["loo_mm"], flush=True)


def charm_ref(case: str, m2m: str) -> dict:
    f = CHARM_REF / f"{case}.json"
    if f.is_file():
        return json.loads(f.read_text())
    import simnibs
    from eval_register import TARGETS, eeg_cap

    names, eeg = eeg_cap()
    pts = np.vstack([np.array(list(TARGETS.values()), float), eeg])
    ref = simnibs.mni2subject_coords(pts, str(m2m))
    r = {"case": case, "mni": pts.tolist(), "charm": np.asarray(ref).tolist(), "n_targets": len(TARGETS),
         "target_names": list(TARGETS), "eeg_names": names}
    _atomic_text(f, json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(4)
    man = load_manifest()
    train = {c: i for c, i in man["cases"].items() if i["split"] == "train"}
    val = val_cases(man)
    assert set(val) <= set(train)
    todo = [(c, i["m2m"]) for c, i in sorted(train.items(), key=lambda kv: kv[0] not in val)
            if not (LABELS / f"{c}.npz").is_file()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, args.workers)) as ex:
        for case, dt in ex.map(_labels_job, todo):
            print(f"labels {case} {dt:.0f} s", flush=True)
    print(f"labels cache: {len(todo)} new in {time.time() - t0:.0f} s", flush=True)
    for c in val:
        charm_ref(c, train[c]["m2m"])
    print("charm reference coordinates done", flush=True)
    build_template_minus_val(train, val, args.device)
    print("done", flush=True)


if __name__ == "__main__":
    main()
