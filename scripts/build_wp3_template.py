"""WP3: build the label template and the affine calibration from TRAIN cases only.

  runs/wp3/template/tissues_MNI_prob.nii.gz + template.json   mean of charm's final_tissues_MNI
  runs/wp3/template/affine_calibration.json                   our affine -> charm's affine frame

The calibration needs our affine for every train case (GPU, ~2 s each) from charm's labels
block-majority'd to the T1 grid (CPU, ~10 s each; --labels-cache DIR reuses <case>.npz files with a
"lab" array made by headmodel.dataset.block_majority).

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/build_wp3_template.py [--rebuild] [--labels-cache DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import dataset, register  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="rebuild the label template too")
    ap.add_argument("--labels-cache", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_num_threads(4)
    f = REPO / "data/manifests/round1.json"
    man = json.loads(f.read_text()) if f.is_file() else dataset.manifest(CT_ROOT)
    train = {c: i for c, i in man["cases"].items() if i["split"] == "train"}
    out = register.WP3_TEMPLATE_DIR   # WP3's template; the WP3b one is scripts/wp3b_build_template.py
    if args.rebuild or not (out / register.TEMPLATE_FILE).is_file():
        register.build_template(train, out)
        print("template:", json.loads((out / "template.json").read_text())["n_cases"], "cases", flush=True)
    params = register.Params(svf_stages=(), calibrate=False, frame="labels")
    pairs = {}
    t0 = time.time()
    for case, info in sorted(train.items()):
        m2m = Path(info["m2m"])
        mat = m2m / "segmentation/coregistrationMatrices.mat"
        if not mat.is_file():
            continue
        t1 = nib.load(m2m / "T1.nii.gz")
        cache = args.labels_cache / f"{case}.npz" if args.labels_cache else None
        if cache is not None and cache.is_file():
            lab = np.load(cache)["lab"]
        else:
            up = nib.load(m2m / dataset.SEG_LABELS)
            lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
        r = register.register(t1, nib.Nifti1Image(lab, t1.affine), None, device=args.device, params=params,
                              template_dir=out, write=False)
        pairs[case] = (np.asarray(r["affine"]), scipy.io.loadmat(mat)["worldToWorldTransformMatrix"])
    print(f"affines of {len(pairs)} train cases in {time.time() - t0:.0f} s", flush=True)
    cal = register.fit_calibration(pairs, meta={
        "what": "ridge regression: our affine -> charm's worldToWorldTransformMatrix frame (matrix logs)",
        "licence": "research only: derived from CC BY-NC 4.0 data",
        "built": time.strftime("%Y-%m-%d %H:%M")})
    (out / register.CALIBRATION_FILE).write_text(json.dumps(cal, indent=1))
    print("calibration leave-one-out (mm, head box):", cal["loo_mm"])


if __name__ == "__main__":
    main()
