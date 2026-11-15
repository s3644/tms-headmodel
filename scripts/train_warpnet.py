"""WP3b: train the supervised warp network (headmodel.warpnet) on charm's warps of the train cases.

  cache   runs/wp3b/warpnet/cache/<case>.npz for every train case (validation cases included, they are
          scored, never trained on). Needs runs/wp3b/template_trainminusval (scripts/wp3b_prepare.py;
          waited for) and uses runs/wp3b/labels_cache when present. <= 4 CPU worker processes read
          charm's files; the GPU part (our raw affine, the input resampling) runs here.
  train   train on train - validation cases, validate on runs/wp3b/val_cases.json, checkpoints in
          runs/wp3b/warpnet/{last,best}.pt, log lines on stdout, history.jsonl. Resumable; yields the
          GPU to nnU-Net jobs (warpnet.yield_reason). The 57 test cases are never read.
  score   validate() of a checkpoint on the first N (sorted) validation cases: the epoch pilot.

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/train_warpnet.py cache [--workers 4]
                       PYTHONPATH=src python scripts/train_warpnet.py train [--epochs 300] [--max-hours 4] [--w-tail 1]
                       PYTHONPATH=src python scripts/train_warpnet.py score --ckpt <pt> --json <out> [--n-val 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import multiprocessing as mp
import nibabel as nib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import dataset, register, warpnet  # noqa: E402
from headmodel import splits as case_splits  # noqa: E402

CT_ROOT = Path.home() / "Projects/ct-headmodel"
LABELS_CACHE = REPO / "runs/wp3b/labels_cache"


def splits() -> tuple[dict, list[str], list[str]]:
    """(manifest cases, train-minus-val cases, val cases); test cases = the frozen 57 WP3 cases."""
    f = REPO / "data/manifests/round1.json"
    man = json.loads(f.read_text()) if f.is_file() else dataset.manifest(CT_ROOT)
    test = set(case_splits.wp3_test_cases())
    train = sorted(c for c, i in man["cases"].items() if i["split"] == "train" and c not in test)
    val = json.loads(warpnet.VAL_FILE.read_text())
    if set(val) - set(train) or set(val) & test:
        raise SystemExit("validation list is not a subset of the train cases")
    fit = [c for c in train if c not in set(val)]
    return man["cases"], fit, val


def _load_cpu(args):
    """Worker: T1, labels on the T1 grid, charm's map sampled on the net grid (CPU only)."""
    case, m2m = args
    torch.set_num_threads(1)
    m2m = Path(m2m)
    t1 = nib.load(m2m / "T1.nii.gz")
    f = LABELS_CACHE / f"{case}.npz"
    if f.is_file():
        lab = np.load(f)["lab"]
    else:
        up = nib.load(m2m / dataset.SEG_LABELS)
        lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    pts, _ = warpnet.mni_points()
    charm = warpnet.sample_charm_map(m2m, warpnet.GridSpec(), pts)
    return case, np.asarray(dataset._vol(t1), np.float32), np.asarray(t1.affine), lab, charm


def cmd_cache(args):
    torch.set_num_threads(4)
    cases, fit, val = splits()
    tpl = warpnet.TEMPLATE
    while not (tpl / register.TEMPLATE_FILE).is_file():
        print(f"waiting for {tpl} (scripts/wp3b_prepare.py)", flush=True)
        time.sleep(60)
    used = {r["case"] for r in json.loads((tpl / "template.json").read_text())["cases"]}
    if used & set(val):
        raise SystemExit("template_trainminusval contains validation cases")
    todo = [(c, cases[c]["m2m"]) for c in val + fit if not (warpnet.CACHE / f"{c}.npz").is_file()]
    print(f"cache: {len(todo)} cases to do, {len(val) + len(fit) - len(todo)} done", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(4, args.workers), mp_context=mp.get_context("spawn")) as ex:
        for case, t1d, t1a, lab, charm in ex.map(_load_cpu, todo):
            t = time.time()
            warpnet.cache_case(case, nib.Nifti1Image(t1d, t1a), lab, None, args.device, template_dir=tpl,
                               charm=charm)
            torch.cuda.empty_cache()
            v = charm[1].mean()
            print(f"{case}: valid {v:.2f}, {time.time() - t:.1f} s GPU+write", flush=True)
    print(f"cache done in {time.time() - t0:.0f} s", flush=True)


def load_sets(fit: list[str], val: list[str], n_train: int = 0, n_val: int = 0):
    t0 = time.time()
    fit = fit[:n_train] if n_train else fit
    val = val[:n_val] if n_val else val
    missing = [c for c in fit + val if not (warpnet.CACHE / f"{c}.npz").is_file()]
    if missing:
        raise SystemExit(f"cache incomplete ({len(missing)} missing, e.g. {missing[:3]}): run `cache` first")
    tr = [warpnet.load_cached(warpnet.CACHE / f"{c}.npz") for c in fit]
    va = [warpnet.load_cached(warpnet.CACHE / f"{c}.npz") for c in val]
    print(f"loaded {len(tr)} train + {len(va)} val samples in {time.time() - t0:.0f} s", flush=True)
    return tr, va


def cmd_train(args):
    torch.set_num_threads(4)
    reason = warpnet.yield_reason()
    if reason:
        print(f"not starting: {reason}", flush=True)
        return
    _, fit, val = splits()
    tr, va = load_sets(fit, val, args.n_train, args.n_val)
    d = warpnet.TrainConfig()
    tcfg = warpnet.TrainConfig(epochs=args.epochs, max_hours=args.max_hours, batch=args.batch,
                               val_every=args.val_every, lr=args.lr, w_tail=args.w_tail, tail_frac=args.tail_frac,
                               aug={k: v * args.aug_scale for k, v in d.aug.items()})
    out = Path(args.out)
    res = warpnet.train(tr, va, out, warpnet.NetConfig(), tcfg, args.device,
                        max_epochs_this_run=args.max_epochs_this_run or None,
                        log=lambda m: print(time.strftime("%F %T"), m, flush=True))
    (out / "status.json").write_text(json.dumps({**res, "time": time.strftime("%F %T")}, indent=1))


def cmd_score(args):
    """Epoch pilot (user rule 2026-10-28): validate() of a checkpoint on the first N validation cases
    (sorted), written to --json. Never test cases."""
    _, _, val = splits()
    cases = sorted(val)[: args.n_val]
    dev = torch.device(args.device)
    model, cfg = warpnet.load_net(args.ckpt, dev)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if set(ck.get("train_cases", [])) & set(cases):
        raise SystemExit("checkpoint was trained on some of these cases")
    va = [warpnet.load_cached(warpnet.CACHE / f"{c}.npz") for c in cases]
    res = warpnet.validate(model, va, cfg.spec, cfg.spec.velocity_grid(dev), cfg.steps, dev)
    res.update({"ckpt": str(args.ckpt), "epoch": ck.get("epoch"), "cases_scored": cases})
    Path(args.json).write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(f"{args.ckpt} (epoch {ck.get('epoch')}) on {cases}: score {s['score']:.4f} targets {s['targets_mean']:.3f}/"
          f"{s['targets_max']:.2f} EEG {s['eeg_mean']:.3f}/{s['eeg_max']:.2f}", flush=True)
    print(f"SCORE {s['score']:.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--device", default="cuda")
    t = sub.add_parser("train")
    t.add_argument("--out", default=str(warpnet.OUT))
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--max-hours", type=float, default=4.0)
    t.add_argument("--max-epochs-this-run", type=int, default=0)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--val-every", type=int, default=5)
    t.add_argument("--lr", type=float, default=warpnet.TrainConfig.lr)
    t.add_argument("--aug-scale", type=float, default=1.0, help="multiplies TrainConfig.aug (rotation, scale, ...)")
    t.add_argument("--n-train", type=int, default=0, help="first N train cases only (smoke runs)")
    t.add_argument("--n-val", type=int, default=0)
    t.add_argument("--w-tail", type=float, default=0.0, help="TrainConfig.w_tail (worst-points term; 0: off)")
    t.add_argument("--tail-frac", type=float, default=warpnet.TrainConfig.tail_frac)
    t.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--n-val", type=int, default=5)
    s.add_argument("--json", required=True)
    s.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"cache": cmd_cache, "train": cmd_train, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
