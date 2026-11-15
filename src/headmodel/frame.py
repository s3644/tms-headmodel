"""WP3b: an intensity-based affine frame, MNI mm -> subject scanner-RAS mm.

Charm's nonlinear warp inherits the frame of its initial affine (segmentation/coregistrationMatrices.mat,
worldToWorldTransformMatrix). This module estimates an affine from the subject T1 INTENSITIES, as an
alternative to the tissue-label affine of headmodel.register, implemented from published methods only:

  * mutual information with Parzen-window joint histograms (Viola and Wells 1997; Mattes et al. 2003),
    between the subject T1 and either charm's atlas template labelling (template.nii in
    charm_atlas_mni_v1-1: a crisp labelling, 12 values, on charm's atlas grid, which is MNI mm) or the
    MNI152 T1 intensities;
  * the label-conditional Gaussian log-likelihood (a correlation-ratio type criterion, Roche et al.
    1998; the same idea as fitting the atlas with a Gaussian intensity model per class, as in
    Van Leemput et al. 1999 / Puonti et al. 2016): sum_k p_k log sigma_k of the T1 inside each class;
  * optionally plus the soft Dice of the subject labels vs the label template (register.py's term);
  * a 12-dof affine optimised by Adam on a template-side pyramid (8 -> 4 -> 2 mm), restricted to the
    whole template grid ("head", neck included), the template above z_cut ("noneck"), or the dilated
    template brain ("brain"). Template points mapped outside the T1 field of view do not count.

Nothing here reads, imports or translates SimNIBS/samseg code; only the atlas template image is read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F

from . import register as R

# charm atlas template.nii values (found by overlap with charm's tissues in MNI space):
# 0 background, 1 WM, 2 GM, 3 CSF, 4 eyes, 5 scalp / soft tissue, 6 bone, 7 vessels and muscle,
# 8-10 deep brain structures, 11 brain stem / cord
ATLAS_N = 12
ATLAS_BRAIN = (1, 2, 8, 9, 10, 11)


@dataclass
class FrameParams:
    metric: str = "mi"            # "mi" (Parzen mutual information) or "gauss" (label-conditional Gaussian)
    template: str = "atlas"       # "atlas" (charm's template labelling) or "mni" (MNI152 T1, metric "mi")
    region: str = "head"          # "head" (whole template grid), "noneck" (z >= z_cut), "brain"
    z_cut: float = -60.0          # MNI mm, "noneck"
    brain_dilate_mm: float = 10.0  # "brain": template brain dilated by this much
    bins: int = 32                # Parzen bins of the subject (and MNI152) T1
    stages: tuple = ((8.0, 120), (4.0, 100), (2.0, 60))   # (template grid mm, iterations)
    lr: float = 0.004             # Adam step of the 3x3 part; translation x 10 mm
    w_label: float = 0.0          # weight of the label soft Dice (register.py's term, needs labels)


def _t1_norm(t1: np.ndarray) -> np.ndarray:
    """Robust [0, 1] scaling of a T1: 0 at zero, 1 at the 99.5th percentile of the non-zero voxels."""
    v = t1[t1 > 0]
    hi = np.percentile(v, 99.5) if v.size else 1.0
    return np.clip(t1 / max(hi, 1e-6), 0, 1).astype(np.float32)


def _parzen(x: torch.Tensor, bins: int) -> torch.Tensor:
    """Soft (Gaussian, sigma = 1 bin) membership of values in [0, 1]: [bins, N], rows sum to 1."""
    c = torch.linspace(0, 1, bins, device=x.device)
    w = torch.exp(-0.5 * ((x[None] - c[:, None]) * (bins - 1)) ** 2)
    return w / w.sum(0, keepdim=True).clamp_min(1e-12)


def mutual_information(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """MI of two soft memberships a [K, N], b [B, N] weighted by m [N] (nats)."""
    j = (a * m) @ b.T
    j = j / j.sum().clamp_min(1e-12)
    pa, pb = j.sum(1, keepdim=True), j.sum(0, keepdim=True)
    return (j * (torch.log(j + 1e-12) - torch.log(pa @ pb + 1e-12))).sum()


def gauss_cost(t: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """sum_k p_k log sigma_k of the values x [N] in the soft classes t [K, N], weights m [N]."""
    w = t * m
    n = w.sum(1).clamp_min(1e-6)
    mu = (w @ x) / n
    var = (w @ (x * x)) / n - mu ** 2
    p = n / n.sum()
    return (p * 0.5 * torch.log(var.clamp_min(1e-6) + 1e-4)).sum()


def _template(p: FrameParams, dev, label_template_dir=R.TEMPLATE_DIR):
    """Template channels [C, ...] on its grid, the region mask, and the affine (MNI mm)."""
    if p.template == "atlas":
        img = nib.load(R.atlas_template())
        lab = np.asarray(img.dataobj).astype(np.int64).reshape(img.shape[:3])
        chans = np.stack([lab == k for k in range(ATLAS_N)]).astype(np.float32)
        brain = np.isin(lab, ATLAS_BRAIN)
    elif p.template == "mni":
        img = nib.load(R.mni_t1())
        t = np.asarray(img.dataobj, np.float32).reshape(img.shape[:3])
        chans = _t1_norm(t)[None]
        brain = t > np.percentile(t[t > 0], 60)   # rough MNI152 brain: replaced by the label template below
        try:
            prob = np.asarray(nib.load(Path(label_template_dir) / R.TEMPLATE_FILE).dataobj, np.float32)
            brain = prob[..., 1:3].sum(-1) > 0.5
        except FileNotFoundError:
            pass
    else:
        raise ValueError(p.template)
    aff = img.affine
    vs = np.linalg.norm(aff[:3, :3], axis=0)
    if p.region == "head":
        mask = np.ones(chans.shape[1:], np.float32)
    elif p.region == "noneck":
        z = nib.affines.apply_affine(aff, np.c_[np.zeros((chans.shape[3], 2)), np.arange(chans.shape[3])])[:, 2]
        mask = np.broadcast_to((z >= p.z_cut)[None, None], chans.shape[1:]).astype(np.float32)
    elif p.region == "brain":
        d = ndi.distance_transform_edt(~brain, sampling=vs)
        mask = (d <= p.brain_dilate_mm).astype(np.float32)
    else:
        raise ValueError(p.region)
    return R.Grid(torch.as_tensor(np.concatenate([chans, mask[None]]), device=dev), aff)


def intensity_affine(t1_img, A0: np.ndarray, p: FrameParams | None = None, device: str = "cuda",
                     labels: np.ndarray | None = None, label_template_dir=R.TEMPLATE_DIR,
                     classes: dict | None = None) -> dict:
    """Affine MNI mm -> subject mm maximising the intensity similarity, started from A0 (e.g. the
    label affine). Returns {"affine": 4x4 list, "loss": float, "peak_mem_gb": float}."""
    p = p or FrameParams()
    dev = torch.device(device)
    if p.template == "mni" and p.metric != "mi":
        raise ValueError("the MNI152 T1 template needs metric 'mi'")
    t1_aff = np.asarray(t1_img.affine, float)
    t1 = _t1_norm(R._vol(t1_img).astype(np.float32))
    sub = R.Grid(torch.as_tensor(np.stack([t1, np.ones_like(t1)]), device=dev), t1_aff)   # T1, field of view
    tpl = _template(p, dev, label_template_dir)
    nt = tpl.data.shape[0] - 1
    lab_sub = lab_tpl = None
    if p.w_label > 0:
        if labels is None:
            raise ValueError("w_label > 0 needs the subject labels")
        classes = classes or dict(R.MERGED)
        lab_sub = R.Grid(torch.cat([R._onehot(labels, classes, dev), torch.ones((1,) + labels.shape, device=dev)]),
                         t1_aff)
        lab_tpl = R.load_template(classes, dev, label_template_dir)
        ncl = len(classes)

    A0 = np.asarray(A0, float)
    c = torch.as_tensor(np.array([0.0, -20.0, 0.0]), dtype=torch.float32, device=dev)   # rotate about mid-head
    L = torch.as_tensor(A0[:3, :3], dtype=torch.float32, device=dev).requires_grad_()
    tr = torch.as_tensor((A0[:3, :3] @ c.cpu().numpy() + A0[:3, 3]) / 10, dtype=torch.float32,
                         device=dev).requires_grad_()
    opt = torch.optim.Adam([{"params": [L], "lr": p.lr}, {"params": [tr], "lr": p.lr * 10}])

    def amap(x):
        return (x - c) @ L.T + tr * 10

    loss = torch.zeros(())
    for res, iters in p.stages:
        f, eaff, eshape = R._eval_grid(tpl.affine, tpl.data.shape[1:], res)
        tl = tpl.pooled(f).data
        x = R._points(eshape, eaff, dev)
        sl = R._sub_pyramid(sub, res)
        keep = tl[nt] > 0.5
        xk, tk = x[keep], tl[:nt][:, keep]
        if p.template == "mni":
            tk = _parzen(tk[0], p.bins)
        if p.w_label > 0:
            ll = lab_tpl.pooled(R._pool_factor(lab_tpl.affine, res))
            lsub = R._sub_pyramid(lab_sub, res)
            fl, laff, lshape = R._eval_grid(lab_tpl.affine, lab_tpl.data.shape[1:], res)
            xl = R._points(lshape, laff, dev)
        for _ in range(iters):
            opt.zero_grad()
            s = sl.sample(amap(xk)[None, None])   # [2, 1, 1, N]
            val, fov = s[0].reshape(-1), s[1].reshape(-1)
            if p.metric == "mi":
                loss = -mutual_information(tk, _parzen(val, p.bins), fov)
            elif p.metric == "gauss":
                loss = gauss_cost(tk, val, fov)
            else:
                raise ValueError(p.metric)
            if p.w_label > 0:
                sd = lsub.sample(amap(xl))
                loss = loss + p.w_label * R.soft_dice(sd[:ncl], ll.data[:ncl] * sd[ncl])
            loss.backward()
            opt.step()
    A = np.eye(4)
    Ld = L.detach().cpu().numpy().astype(float)
    A[:3, :3] = Ld
    A[:3, 3] = tr.detach().cpu().numpy() * 10 - Ld @ c.cpu().numpy()
    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else 0.0
    return {"affine": A.tolist(), "loss": float(loss.item()), "peak_mem_gb": peak}


# ---------------------------------------------------------------------------- learned frame (ridge)
# Charm's worldToWorldTransformMatrix is the output of charm's own initialiser (a trained network),
# which no anatomical fit reproduces (the least-squares affine of charm's own nonlinear warp is
# ~5 mm from it). A ridge regression from our label affines predicts it best (runs/wp3b/frame).

RIDGE_FILE = R.RIDGE_FILE   # "frame_ridge.json" in the template dir
BRAIN_CLASSES = {k: R.MERGED[k] for k in ("brain", "wm", "csf")}


# lambda grid of fit_ridge. WP3b first used (0.01 .. 3.0); its smallest value won in both templates, so
# the grid was extended downward: the leave-one-out minimum is then interior (0.003 on the train cases)
RIDGE_LAMS = (1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def fit_ridge(feats: list[list[np.ndarray]], W: list[np.ndarray], lams=RIDGE_LAMS,
              meta: dict | None = None, cases: list[str] | None = None) -> dict:
    """Ridge regression (standardised matrix logs of the feature affines, feats[i] = [A_base, ...]) of
    the correction D = log(inv(A_base) W); lambda by leave-one-out mean distance over a head-sized box.
    cases (one id per row, optional) are stored in the model as its provenance. lam_at_grid_edge flags a
    leave-one-out minimum at an end of lams (widen the grid)."""
    if cases is not None and len(cases) != len(feats):
        raise ValueError(f"{len(cases)} case ids for {len(feats)} rows")
    lg = R._log_affine
    X = np.array([np.concatenate([lg(np.asarray(a)) for a in f]) for f in feats])
    Y = np.array([lg(np.linalg.inv(np.asarray(f[0])) @ np.asarray(w)) for f, w in zip(feats, W)])
    box = np.array([[x, y, z] for x in (-70, 0, 70) for y in (-100, 0, 70) for z in (-40, 30, 90)], float)

    def fit(m, lam):
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        Xs, ym = (X[m] - mu) / sd, Y[m].mean(0)
        B = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(X.shape[1]), Xs.T @ (Y[m] - ym))
        return {"mu": mu.tolist(), "sd": sd.tolist(), "B": B.tolist(), "y_mean": ym.tolist()}

    loo = {}
    for lam in lams:
        d = []
        for i in range(len(X)):
            m = np.arange(len(X)) != i
            A = apply_ridge(feats[i], fit(m, lam))
            d.append(np.linalg.norm(nib.affines.apply_affine(A, box) - nib.affines.apply_affine(np.asarray(W[i]), box), axis=1).mean())
        loo[lam] = float(np.mean(d))
    lam = min(loo, key=loo.get)
    out = {**fit(np.ones(len(X), bool), lam), "lam": lam, "loo_box_mm": loo[lam], "n_cases": len(X),
           "loo_all": {str(k): v for k, v in loo.items()}, "lams": [float(x) for x in lams],
           "lam_at_grid_edge": bool(len(lams) > 1 and lam in (min(lams), max(lams))), **(meta or {})}
    if cases is not None:
        out["cases"] = list(cases)
    return out


def apply_ridge(feats: list[np.ndarray], model: dict) -> np.ndarray:
    x = np.concatenate([R._log_affine(np.asarray(a)) for a in feats])
    y = ((x - np.asarray(model["mu"])) / np.asarray(model["sd"])) @ np.asarray(model["B"]) + np.asarray(model["y_mean"])
    return np.asarray(feats[0], float) @ R._exp_affine(y)
