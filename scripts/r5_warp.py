"""R5 warp net (PLAN 10.8; rules r5_train_plan, r5_warp_train): the WP3b warp net retrained on the labels it
really sees, the R5 segmenter's out-of-fold predictions, with the IXI heads added and the gated points weighted up.

  oof      (nnunet env) every head of Dataset602's 5-fold split, predicted by the one fold that held it out,
           through navprep's own segment stage (LPS 1 mm grid, crop, nnU-Net with mirroring, largest part, charm
           labels): runs/r5/oof/<case>/{t1,labels}.nii.gz + segment.json. One model load per fold.
  cache    (rtms-gpu) warpnet.cache_case of every warp train and validation head on those labels and that T1 (our
           affine from the predicted labels, as navprep computes it), the target being charm's own map:
           runs/r5/warp/cache/<case>.npz. <= 4 CPU workers sample charm's maps.
  train    warpnet.train with the rule's settings: runs/r5/warp/e<epochs>/{last,best}.pt (resumable).
  score    the epoch pilot: validate() of a checkpoint on the first 5 (sorted) validation heads; prints
           SCORE <score_gate> (targets, rTMS EEG set, fiducials; unprojected).
  finalise store the training support in the checkpoint (warpnet.net_support reads it), so a release needs
           no cache.

Case sets (r5_train_plan): train = round 1's train cases minus the 30 WP3b validation cases + the IXI train
heads; validation = the 30 WP3b validation cases + the 30 IXI validation heads (all out-of-fold labels). Test
heads (round 1's test cases, the IXI test heads) are never read here.

Usage:
  ~/.local/share/mamba/envs/nnunet/bin/python scripts/r5_warp.py oof
  PYTHONPATH=src ~/.local/share/mamba/envs/rtms-gpu/bin/python scripts/r5_warp.py cache|train|score|finalise ...
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

import multiprocessing as mp
import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import navprep  # noqa: E402

R5 = Path(os.environ.get("HM_R5_RUNS", REPO / "runs/r5"))
OOF = R5 / "oof"
WARP = R5 / "warp"
CACHE = WARP / "cache"
MANIFEST = Path(os.environ.get("HM_R5_MANIFEST", REPO / "data/manifests/r5.json"))
NN_ROOT = Path(os.environ.get("HM_R5_NNUNET", REPO / "data/nnunet"))
DS = os.environ.get("HM_R5_DATASET", "Dataset602_MRHeadR5")
TRAINER_DIR = os.environ.get("HM_R5_TRAINER_DIR", "nnUNetTrainerHMThick__nnUNetPlans__3d_fullres")   # (R6: its trainer)
WP3B_VAL = REPO / "runs/wp3b/val_cases.json"
N_PILOT = 5
# r5_warp_train (sealed): point-loss weight of the 27 gated points, worst-points term as round 3
W_GATE = 4.0
W_TAIL = 1.0


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())["cases"]


def sets() -> tuple[list[str], list[str]]:
    """(warp train heads, warp validation heads) of the frozen R5 manifest; never a test head."""
    cases = manifest()
    wp3b_val = set(json.loads(WP3B_VAL.read_text()))
    trainable = {c for c, i in cases.items() if i["split"] == "train"}
    val = sorted(c for c in trainable if c in wp3b_val or cases[c].get("r5_set") == "val")
    train = sorted(trainable - set(val))
    if set(val) - trainable:
        raise SystemExit("a validation head is not in Dataset602's training set (no out-of-fold label)")
    return train, val


def folds() -> list[dict]:
    return json.loads((NN_ROOT / "preprocessed" / DS / "splits_final.json").read_text())


# ------------------------------------------------------------------------------------------- oof

def cmd_oof(a) -> int:
    """navprep.segment with fold k's model on fold k's held-out heads."""
    cases = manifest()
    want = set(sum(sets(), []))
    model = NN_ROOT / "results" / DS / TRAINER_DIR
    done = 0
    for k, split in enumerate(folds()):
        todo = [c for c in split["val"] if c in want and not (OOF / c / "labels.nii.gz").is_file()]
        if not todo:
            continue
        if not (model / f"fold_{k}" / "checkpoint_final.pth").is_file():
            raise SystemExit(f"fold {k} is not trained yet ({model / f'fold_{k}'})")
        pred = navprep.load_predictor(model, (k,), a.device)
        for c in todo:
            t = time.time()
            work = OOF / c / "work"
            image = Path(cases[c]["m2m"]) / "T1.nii.gz"
            out = navprep.segment(image, "mr", work, cap_gb=a.cap_gb, device=a.device, predictor=pred)
            for f in ("t1.nii.gz", "labels.nii.gz"):
                os.replace(work / f, OOF / c / f)
            (OOF / c / "segment.json").write_text(json.dumps({**out, "fold": k, "model": str(model)}, indent=1))
            shutil.rmtree(work)
            done += 1
            print(f"{c}: fold {k} {time.time() - t:.1f} s", flush=True)
    missing = sorted(c for c in want if not (OOF / c / "labels.nii.gz").is_file())
    print(f"oof: {done} new, {len(want) - len(missing)}/{len(want)} done", flush=True)
    return 1 if missing else 0


# ------------------------------------------------------------------------------------------- cache

def _charm_map(args):
    """Worker: charm's map of a head sampled on the net grid (CPU only)."""
    import torch

    from headmodel import warpnet

    case, m2m = args
    torch.set_num_threads(1)
    pts, _ = warpnet.mni_points()
    return case, warpnet.sample_charm_map(Path(m2m), warpnet.GridSpec(), pts)


def cmd_cache(a) -> int:
    import torch

    from headmodel import warpnet

    torch.set_num_threads(4)
    cases = manifest()
    train, val = sets()
    todo = [(c, cases[c]["m2m"]) for c in val + train if not (CACHE / f"{c}.npz").is_file()]
    miss = [c for c, _ in todo if not (OOF / c / "labels.nii.gz").is_file()]
    if miss:
        raise SystemExit(f"no out-of-fold labels for {len(miss)} heads (e.g. {miss[:3]}): run `oof` first")
    print(f"cache: {len(todo)} to do, {len(val) + len(train) - len(todo)} done", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, a.workers), mp_context=mp.get_context("spawn")) as ex:
        for case, charm in ex.map(_charm_map, todo):
            t = time.time()
            t1 = nib.load(OOF / case / "t1.nii.gz")
            lab = np.asarray(nib.load(OOF / case / "labels.nii.gz").dataobj).astype(np.uint8)
            warpnet.cache_case(case, t1, lab, None, a.device, out_dir=CACHE, template_dir=warpnet.TEMPLATE,
                               charm=charm)
            torch.cuda.empty_cache()
            print(f"{case}: valid {charm[1].mean():.2f}, {time.time() - t:.1f} s GPU+write", flush=True)
    print(f"cache done in {time.time() - t0:.0f} s", flush=True)
    return 0


def load(cases: list[str]) -> list[dict]:
    from headmodel import warpnet

    miss = [c for c in cases if not (CACHE / f"{c}.npz").is_file()]
    if miss:
        raise SystemExit(f"cache incomplete ({len(miss)} missing, e.g. {miss[:3]}): run `cache` first")
    return [warpnet.load_cached(CACHE / f"{c}.npz") for c in cases]


# ------------------------------------------------------------------------------------------- train / score

def train_config(epochs: int, max_hours: float):
    from headmodel import warpnet

    return warpnet.TrainConfig(epochs=epochs, max_hours=max_hours, val_every=20, w_tail=W_TAIL, w_gate=W_GATE)


def cmd_train(a) -> int:
    import torch

    from headmodel import warpnet

    torch.set_num_threads(4)
    reason = warpnet.yield_reason()
    if reason:
        print(f"not starting: {reason}", flush=True)
        return 0
    train, val = sets()
    t0 = time.time()
    tr, va = load(train), load(val)
    print(f"loaded {len(tr)} train + {len(va)} val heads in {time.time() - t0:.0f} s", flush=True)
    out = WARP / f"e{a.epochs}"
    res = warpnet.train(tr, va, out, warpnet.NetConfig(), train_config(a.epochs, a.max_hours), a.device,
                        log=lambda m: print(time.strftime("%F %T"), m, flush=True))
    (out / "status.json").write_text(json.dumps({**res, "time": time.strftime("%F %T")}, indent=1))
    return 0


def cmd_score(a) -> int:
    import torch

    from headmodel import warpnet

    _, val = sets()
    cases = sorted(val)[:N_PILOT]
    dev = torch.device(a.device)
    model, cfg = warpnet.load_net(a.ckpt, dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if set(ck.get("train_cases", [])) & set(cases):
        raise SystemExit("checkpoint was trained on some of these heads")
    res = warpnet.validate(model, load(cases), cfg.spec, cfg.spec.velocity_grid(dev), cfg.steps, dev)
    res.update({"ckpt": str(a.ckpt), "epoch": ck.get("epoch"), "cases_scored": cases})
    Path(a.json).write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(f"{a.ckpt} (epoch {ck.get('epoch')}) on {cases}: gate score {s['score_gate']:.4f} targets "
          f"{s['targets_mean']:.3f}/{s['targets_max']:.2f} rTMS EEG {s['rtms_mean']:.3f}/{s['rtms_max']:.2f} "
          f"fiducials {s['fid_mean']:.3f}/{s['fid_max']:.2f}", flush=True)
    print(f"SCORE {s['score_gate']:.6f}")
    return 0


def cmd_finalise(a) -> int:
    import torch

    from headmodel import warpnet

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if "support" not in ck:
        ck["support"] = warpnet.support_mask(sorted(ck["train_cases"]), CACHE)
        warpnet._atomic_save(ck, Path(a.ckpt))
    print(f"{a.ckpt}: support {int(np.asarray(ck['support']).sum())} voxels from {len(ck['train_cases'])} heads")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof")
    o.add_argument("--device", default="cuda")
    o.add_argument("--cap-gb", type=float, default=None)
    c = sub.add_parser("cache")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, required=True)
    t.add_argument("--max-hours", type=float, default=30.0)
    t.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--json", required=True)
    s.add_argument("--device", default="cuda")
    f = sub.add_parser("finalise")
    f.add_argument("--ckpt", required=True)
    a = ap.parse_args(argv)
    return {"oof": cmd_oof, "cache": cmd_cache, "train": cmd_train, "score": cmd_score, "finalise": cmd_finalise}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
