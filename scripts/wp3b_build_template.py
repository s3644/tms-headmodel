"""WP3b Combine: the final label template, affine calibration and ridge frame from ALL train cases.

The train set is the one WP3's template was built from (runs/wp3/template/template.json, 134 cases;
the manifest keeps growing while charm queues run, so it is not re-read for the case list). No test
case is read. Writes runs/wp3b/template/ (runs/wp3/template/ is left as it is):

  tissues_MNI_prob.nii.gz + template.json   mean of charm's final_tissues_MNI over the train cases
  affine_calibration.json                   our label affine -> charm's frame (Params(frame="labels"))
  frame_ridge.json                          [label affine, brain-only label affine] -> charm's frame
                                            (Params(frame="ridge"), headmodel.frame.fit_ridge)
  affines.json                              the per-case affines the two were fitted from

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/wp3b_build_template.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import dataset, frame, register, splits  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
OUT = REPO / "runs/wp3b/template"
LABELS = REPO / "runs/wp3b/labels_cache"
WP3_TEMPLATE = REPO / "runs/wp3/template/template.json"


def main():
    torch.set_num_threads(4)
    man = dataset.manifest(CT_ROOT)
    train_ids = sorted(c["case"] for c in json.loads(WP3_TEMPLATE.read_text())["cases"])
    test_ids = set(splits.wp3_test_cases())
    assert not (set(train_ids) & test_ids), "train list contains test cases"
    assert all(man["cases"][c]["split"] == "train" for c in train_ids)
    train = {c: man["cases"][c] for c in train_ids}
    tmp = OUT.with_name(f".template.{os.getpid()}")
    register.build_template(train, tmp)
    p_raw = register.Params(svf_stages=(), calibrate=False, frame="labels")
    p_brain = register.Params(svf_stages=(), calibrate=False, frame="labels", classes=dict(frame.BRAIN_CLASSES))
    rows, t0 = {}, time.time()
    for case in train_ids:
        m2m = Path(train[case]["m2m"])
        mat = m2m / "segmentation/coregistrationMatrices.mat"
        if not mat.is_file():
            continue
        t1 = nib.load(m2m / "T1.nii.gz")
        lab = nib.Nifti1Image(np.load(LABELS / f"{case}.npz")["lab"], t1.affine)
        a = register.register(t1, lab, None, params=p_raw, template_dir=tmp, write=False)["affine_raw"]
        b = register.register(t1, lab, None, params=p_brain, template_dir=tmp, write=False)["affine_raw"]
        rows[case] = {"raw": np.asarray(a).tolist(), "brain": np.asarray(b).tolist(),
                      "W": scipy.io.loadmat(mat)["worldToWorldTransformMatrix"].tolist()}
        torch.cuda.empty_cache()
        print(case, f"{time.time() - t0:.0f} s", flush=True)
    meta = {"split": f"all {len(rows)} train cases (runs/wp3/template/template.json)",
            "licence": "research only: derived from CC BY-NC 4.0 data", "built": time.strftime("%Y-%m-%d %H:%M")}
    cal = register.fit_calibration({c: (np.asarray(r["raw"]), np.asarray(r["W"])) for c, r in rows.items()}, meta={
        "what": "ridge regression: our affine -> charm's worldToWorldTransformMatrix frame (matrix logs)", **meta})
    (tmp / register.CALIBRATION_FILE).write_text(json.dumps(cal, indent=1))
    ridge = frame.fit_ridge([[np.asarray(r["raw"]), np.asarray(r["brain"])] for r in rows.values()],
                            [np.asarray(r["W"]) for r in rows.values()],
                            meta={"features": ["raw", "labels_brainonly"],
                                  "what": "ridge: [label affine, brain-only label affine] -> charm's frame", **meta},
                            cases=list(rows))
    (tmp / frame.RIDGE_FILE).write_text(json.dumps(ridge, indent=1))
    (tmp / "affines.json").write_text(json.dumps(rows))
    if OUT.exists():
        shutil.rmtree(OUT)
    os.replace(tmp, OUT)
    print("calibration LOO (mm):", cal["loo_mm"], " ridge LOO box (mm):", ridge["loo_box_mm"], "lam", ridge["lam"],
          " cases", len(rows), flush=True)


if __name__ == "__main__":
    main()
