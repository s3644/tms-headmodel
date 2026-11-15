"""WP3: GPU registration of a head to MNI space, written as charm's toMNI warps.

Inputs: the subject T1 (scanner RAS mm) and its tissue labels in charm's label values on the T1 grid
(charm's own labels block-majority'd to the T1 grid for now, the segmenter's prediction later).
Template side: the MNI152 T1 and a label template, the per-label probability maps of charm's
toMNI/final_tissues_MNI.nii.gz averaged over training cases (build_template).

Method, implemented from the published methods only:
  (a) affine, 12 dof, MNI mm -> subject mm, initialised from the brain centroids, volumes and
      principal axes; multi-resolution Adam on soft Dice over merged tissue classes (LNCC of T1 vs
      the MNI152 T1 is available, weight 0 by default: on train cases it moved us away from charm).
  (a') affine calibration: charm's nonlinear warp inherits its own affine frame wherever the labels
      leave the correspondence free (tangential on the scalp), so a ridge regression fitted on train
      cases (fit_calibration) moves our affine into charm's frame; the SVF starts from that frame.
  (b) a stationary velocity field (SVF) v in MNI mm on a coarse grid (16 -> 8 -> 4 mm), exponentiated
      by scaling and squaring (Arsigny et al. 2006; Ashburner 2007; as in VoxelMorph, Dalca et al.
      2019); the same soft Dice plus a diffusion penalty on v, 10x stiffer inside the template brain,
      and a penalty on Jacobian determinants below 0.3. Forward map exp(+v), inverse exp(-v), both
      exponentiated on a 2 mm grid for the written maps.
  The full map is   subject(x) = A( phi(x) ),   phi = exp(v),  x in MNI mm,
  and its inverse   mni(y)     = phi^-1( A^-1 y ),  phi^-1 = exp(-v).
  Weights and resolutions were chosen on train cases only (distance to charm's warps).
  WP3b defaults (chosen on a 30-case validation split of the train cases, runs/wp3b/frame_full):
  the SVF starts from the ridge frame (headmodel.frame: our label affine and a brain-only label
  affine regressed onto charm's frame), and the brain class is off in the SVF soft Dice (WM kept).
  Params(frame="labels", svf_class_weights=None) with WP3_TEMPLATE_DIR is the WP3 behaviour.

Output, the same files, grids and conventions as charm (checked against charm's writer and the
simnibs readers):
  toMNI/MNI2Conform_nonl.nii.gz  charm's atlas grid (182x238x282, 1 mm, diag(-1,1,1), origin
                                 (90,-126,-172)), float64, x*y*z*3: subject scanner-RAS mm of each
                                 MNI voxel (read by mni2subject_coords);
  toMNI/Conform2MNI_nonl.nii.gz  the T1 grid, float32, x*y*z*3: MNI mm of each T1 voxel
                                 (read by subject2mni_coords);
  both completely filled: the velocity grid covers both domains, and beyond it the displacement is
  continued from its border (charm leaves zeros, which simnibs reads as real coordinates);
  segmentation/coregistrationMatrices.mat  worldToWorldTransformMatrix = A (MNI RAS mm -> subject
                                 RAS mm, as brain_surface._load_mni_transform and
                                 charm_utils._cut_and_combine_labels read it) and
                                 imageToImageTransformMatrix = inv(T1 affine) @ A @ atlas affine
                                 (atlas voxel -> T1 voxel, as charm saves it; no reader needs it).

Licence: the label template and the calibration are derived from CC BY-NC 4.0 data (SynthRAD; Sun Yat-sen is CC BY 4.0), so it and
any output made with it are research only (PLAN.md, section 6).
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io
import scipy.linalg
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
# WP3b: label template, calibration and ridge frame (frame.RIDGE_FILE) from all train cases
# (scripts/wp3b_build_template.py); WP3's template (no ridge frame) is kept in WP3_TEMPLATE_DIR
TEMPLATE_DIR = REPO / "runs/wp3b/template"
# WP3b round 4, adopted 2026-10-29 under the sealed rule wp3b_r4_adopt (validation): round 3's warp net + its
# template, the last SVF stage only (brain weight 0) and the fold repair. adopted_setting() returns it; Params()
# keeps the net-free WP3b defaults. Test run (57 cases): runs/wp3b/warpnet_r4/eval_test, gate not met.
ADOPTED_NET = Path(os.environ.get("HM_WARP_NET", REPO / "runs/wp3b/warpnet_r3/e1000/last.pt"))   # R5 scoring: its net
ADOPTED_TEMPLATE_DIR = REPO / "runs/wp3b/template_trainminusval"
WP3_TEMPLATE_DIR = REPO / "runs/wp3/template"
TEMPLATE_FILE = "tissues_MNI_prob.nii.gz"
CALIBRATION_FILE = "affine_calibration.json"
RIDGE_FILE = "frame_ridge.json"   # = headmodel.frame.RIDGE_FILE (Params(frame="ridge"))
N_LABELS = 11   # background + charm labels 1..10


def simnibs_dir() -> Path:
    spec = importlib.util.find_spec("simnibs")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("simnibs is not installed in this environment")
    return Path(list(spec.submodule_search_locations)[0])


def atlas_template() -> Path:
    """The grid charm writes MNI2Conform_nonl on (its samseg atlas template)."""
    return simnibs_dir() / "segmentation/atlases/charm_atlas_mni_v1-1/template.nii"


def mni_t1() -> Path:
    return simnibs_dir() / "resources/templates/MNI152_T1_1mm.nii.gz"


# merged classes (charm label values) for the similarity terms; brain = WM + GM etc.
MERGED = {"brain": (1, 2), "wm": (1,), "csf": (3,), "bone": (4, 7, 8), "scalp": (5, 10), "eyes": (6,)}


@dataclass
class Params:
    classes: dict = field(default_factory=lambda: dict(MERGED))
    w_ncc: float = 0.0                 # LNCC weight (soft Dice weight is 1); 0: it moved us away from charm
    class_weights: dict = field(default_factory=dict)   # soft Dice weight per class (default 1)
    w_ssd: float = 0.0                 # weight of the mean squared difference of the class maps
    ssd_classes: tuple = ()            # ... of these classes (all when empty), in the SVF stage only
    ncc_win: int = 7                   # LNCC window, voxels of the evaluation grid
    affine_stages: tuple = ((4.0, 150), (2.0, 100))                 # (eval grid mm, iterations)
    svf_stages: tuple = ((4.0, 16.0, 100), (2.0, 8.0, 150), (2.0, 4.0, 150))   # (eval mm, velocity mm, iters)
    lr_affine: float = 0.01            # Adam step for the 3x3 part (unitless); translation x 10 mm
    lr_svf: float = 0.2                # Adam step for v, mm
    w_diff: float = 1.0                # diffusion penalty on v
    w_bend: float = 0.0                # bending penalty on v
    stiff_brain: float = 10.0          # relative stiffness inside the template brain (1: uniform)
    w_jac: float = 1.0                 # penalty on Jacobian determinants below jac_floor (folding guard)
    jac_floor: float = 0.3
    calibrate: bool = True             # move the affine into charm's frame (CALIBRATION_FILE, if present)
    w_mag: float = 1e-4                # tiny |v|^2 penalty, so v decays far from the head
    steps: int = 7                     # scaling and squaring
    out_spacing: float = 2.0           # grid of the final exp(+-v) that the written maps sample
    margin_mm: float = 24.0            # velocity grid margin around both output domains
    # WP3b terms (0: off, the WP3 behaviour)
    w_ncc_brain: float = 0.0           # LNCC of T1 vs the MNI152 T1 inside the template brain, last SVF stage
    w_sdt: float = 0.0                 # SSD of clipped signed distance maps of class boundaries (SVF stages)
    sdt_classes: tuple = ("brain", "wm")   # ... of these merged classes (pial and GM/WM boundaries)
    sdt_clip: float = 6.0              # mm; distances are clipped to +-sdt_clip and scaled by it
    # soft Dice weights in the SVF stages (None: class_weights). WP3b default: the brain (WM + GM) class
    # off in the SVF, WM kept (chosen on the validation split; it removed a 15 mm regional slip)
    svf_class_weights: dict | None = field(default_factory=lambda: {"brain": 0.0})
    svf_reg_scale: tuple = ()          # per SVF stage multiplier of the diffusion + bending penalty (missing: 1)
    # WP3b task B: the affine frame the SVF starts from. "labels": the label affine (+ calibration);
    # "intensity": headmodel.frame.intensity_affine (T1 intensities vs charm's atlas template or the
    # MNI152 T1) started from the label affine, not calibrated; "ridge": headmodel.frame.apply_ridge of
    # the label affine and a brain-only (WM, GM, CSF) label affine, model frame.RIDGE_FILE in template_dir;
    # "auto" (WP3b default, chosen on the validation split): "ridge" if template_dir has frame.RIDGE_FILE,
    # else "labels" (and "labels" with init_warp, which replaces the frame anyway)
    frame: str = "auto"
    frame_params: dict = field(default_factory=dict)   # headmodel.frame.FrameParams fields
    ridge_file: str = RIDGE_FILE       # ridge model in template_dir (frame="ridge" / "auto"); older fits are
                                       # kept beside it (e.g. frame_ridge_lam0.01.json, the first WP3b grid)
    # WP3b task A: a supervised warp network (headmodel.warpnet checkpoint path). Its frame (our raw
    # label affine, with template_dir = the net's template) replaces A and its velocity v_net starts
    # the SVF: the stages optimise v = v_net + dv, regularised on dv only (+ w_init |dv|^2);
    # svf_stages=() writes the net's warp alone. None: off (the WP3 behaviour)
    init_warp: str | None = None
    w_init: float = 0.0
    # WP3b round 3: v_net outside the net's training support (warpnet.net_support) replaced by its
    # harmonic extension (warpnet.harmonic_fill); False: the raw net output (task N, folds below the head)
    net_fill: bool = True
    # WP3 round 4: repair of the written maps where their Jacobian determinant is <= 0 (_unfold: local 3x3x3
    # smoothing of the folded voxels and their neighbours until none is left). exp(-v) folds a few voxels
    # where the forward map compresses strongly (PIT30, forward J 0.13); a finer exp grid also fixes it but
    # needs 12 GB, and a fixed-point inverse does not converge there. It only acts where a written map folds. 0: off
    unfold_iters: int = 50


# ---------------------------------------------------------------------------- template

def build_template(cases: dict, out_dir: Path = TEMPLATE_DIR) -> Path:
    """Per-label probability maps (background + charm labels 1..10) on the MNI152 grid, averaged over
    the given cases' toMNI/final_tissues_MNI.nii.gz. cases: {case: {"m2m":..., "dataset":...,
    "licence":..., "split": "train"}} (headmodel.dataset.manifest rows). Writes TEMPLATE_FILE (uint8,
    scaled 1/255) and template.json (cases, licences) into out_dir."""
    bad = [c for c, i in cases.items() if i.get("split", "train") != "train"]
    if bad:
        raise ValueError(f"template must be built from train cases only: {bad[:5]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc, ref, used = None, None, []
    for case, info in sorted(cases.items()):
        f = Path(info["m2m"]) / "toMNI/final_tissues_MNI.nii.gz"
        if not f.is_file():
            continue
        img = nib.load(f)
        lab = np.asarray(img.dataobj).reshape(img.shape[:3]).astype(np.int64)
        if ref is None:
            ref = img
            acc = np.zeros(img.shape[:3] + (N_LABELS,), np.float32)
        elif img.shape[:3] != ref.shape[:3] or not np.allclose(img.affine, ref.affine):
            continue
        lab[(lab < 0) | (lab >= N_LABELS)] = 0
        for k in range(N_LABELS):
            acc[..., k] += lab == k
        used.append({"case": case, "dataset": info.get("dataset"), "licence": info.get("licence")})
    if not used:
        raise ValueError("no final_tissues_MNI.nii.gz found")
    prob = np.round(acc / len(used) * 255).astype(np.uint8)
    out = nib.Nifti1Image(prob, ref.affine)
    out.header.set_slope_inter(1 / 255, 0)
    nib.save(out, out_dir / TEMPLATE_FILE)
    meta = {"file": TEMPLATE_FILE, "channels": ["background"] + [f"charm label {k}" for k in range(1, N_LABELS)],
            "source": "toMNI/final_tissues_MNI.nii.gz of charm runs (training heads)", "n_cases": len(used),
            "licence": "research only: derived from CC BY-NC 4.0 data", "cases": used,
            "built": time.strftime("%Y-%m-%d %H:%M")}
    (out_dir / "template.json").write_text(json.dumps(meta, indent=1))
    return out_dir / TEMPLATE_FILE


# ---------------------------------------------------------------------------- affine calibration

def _log_affine(A: np.ndarray) -> np.ndarray:
    return scipy.linalg.logm(np.asarray(A, float)).real[:3].ravel()


def _exp_affine(y: np.ndarray) -> np.ndarray:
    return scipy.linalg.expm(np.vstack([np.reshape(y, (3, 4)), np.zeros(4)]))


def fit_calibration(pairs: dict, lam: float = 0.1, meta: dict | None = None) -> dict:
    """Ridge regression (standardised inputs) from our affine A (MNI -> subject) to the correction D
    with charm's worldToWorldTransformMatrix W = A @ D, both in matrix-log coordinates.
    pairs: {case: (A, W)} of train cases. Charm's nonlinear warp inherits its affine frame where the
    labels leave the correspondence free (tangential on the scalp), so the SVF starts from A @ D."""
    cases = sorted(pairs)
    X = np.array([_log_affine(pairs[c][0]) for c in cases])
    Y = np.array([_log_affine(np.linalg.inv(pairs[c][0]) @ np.asarray(pairs[c][1])) for c in cases])

    def fit(m):
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        Xs, ym = (X[m] - mu) / sd, Y[m].mean(0)
        B = np.linalg.solve(Xs.T @ Xs + lam * len(Xs) * np.eye(X.shape[1]), Xs.T @ (Y[m] - ym))
        return {"mu": mu.tolist(), "sd": sd.tolist(), "B": B.tolist(), "y_mean": ym.tolist()}

    pts = np.array([[x, y, z] for x in (-70, 0, 70) for y in (-100, 0, 70) for z in (-40, 30, 90)], float)
    raw, loo = [], []
    for i in range(len(cases)):   # leave-one-out check, distances on a head-sized box of points (mm)
        m = np.ones(len(cases), bool)
        m[i] = False
        A, W = (np.asarray(a, float) for a in pairs[cases[i]])
        Ac = calibrate(A, fit(m))
        raw.append(np.linalg.norm(nib.affines.apply_affine(A, pts) - nib.affines.apply_affine(W, pts), axis=1).mean())
        loo.append(np.linalg.norm(nib.affines.apply_affine(Ac, pts) - nib.affines.apply_affine(W, pts), axis=1).mean())
    cal = fit(np.ones(len(cases), bool))
    cal.update({"lam": lam, "n_cases": len(cases), "cases": cases,
                "loo_mm": {"raw_mean": float(np.mean(raw)), "calibrated_mean": float(np.mean(loo)),
                           "calibrated_max": float(np.max(loo))}, **(meta or {})})
    return cal


def calibrate(A: np.ndarray, cal: dict) -> np.ndarray:
    y = ((_log_affine(A) - np.asarray(cal["mu"])) / np.asarray(cal["sd"])) @ np.asarray(cal["B"]) + np.asarray(cal["y_mean"])
    return np.asarray(A, float) @ _exp_affine(y)


def same_template(trained: str | Path, given: str | Path) -> bool:
    """Is `given` the template a warp net was trained with? The same folder, or the same repo-relative path
    under another repo copy (a navprep release: make_release.sh copies runs/... with sha256 in RELEASE.json)."""
    a, b = Path(trained), Path(given)
    if a.resolve() == b.resolve():
        return True
    rel = lambda q: q.parts[q.parts.index("runs"):] if "runs" in q.parts else None   # noqa: E731
    return rel(a) is not None and rel(a) == rel(b)


def adopted_setting() -> tuple["Params", Path]:
    """The adopted registration (WP3b round 4) as (Params, template_dir) for register()."""
    return Params(init_warp=str(ADOPTED_NET), svf_stages=((2.0, 4.0, 150),), svf_class_weights={"brain": 0.0},
                  w_init=0.0, net_fill=True, unfold_iters=50), ADOPTED_TEMPLATE_DIR


def load_calibration(template_dir: Path = TEMPLATE_DIR) -> dict | None:
    f = Path(template_dir) / CALIBRATION_FILE
    return json.loads(f.read_text()) if f.is_file() else None


# ---------------------------------------------------------------------------- torch helpers

def _t(a, dev, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=dev)


def _vol(img) -> np.ndarray:
    data = np.asanyarray(img.dataobj)
    return data.reshape(data.shape[:3]) if data.ndim == 4 and data.shape[3] == 1 else data


class Grid:
    """A volume [C, nx, ny, nz] with its voxel -> mm affine, sampled at mm points."""

    def __init__(self, data: torch.Tensor, affine: np.ndarray):
        self.data, self.affine = data, np.asarray(affine, float)
        dev = data.device
        inv = np.linalg.inv(self.affine)
        n = torch.tensor(data.shape[-3:], dtype=torch.float32, device=dev)
        # mm -> normalised grid_sample coordinates (align_corners=True), in (i, j, k) order
        self._m = _t(inv[:3, :3], dev) * (2 / (n - 1))[:, None]
        self._b = _t(inv[:3, 3], dev) * (2 / (n - 1)) - 1

    def sample(self, pts: torch.Tensor, padding: str = "zeros") -> torch.Tensor:
        """pts [..., 3] mm (3-D grid of points) -> [C, ...]."""
        g = (pts @ self._m.T + self._b).flip(-1)
        shp = pts.shape[:-1]
        g = g.reshape(1, *shp, 3) if len(shp) == 3 else g.reshape(1, 1, 1, -1, 3)
        out = F.grid_sample(self.data[None], g, mode="bilinear", padding_mode=padding, align_corners=True)[0]
        return out.reshape(out.shape[0], *shp)

    def pooled(self, f: tuple[int, int, int]) -> "Grid":
        if all(k == 1 for k in f):
            return self
        d = F.avg_pool3d(self.data[None], kernel_size=f, stride=f, ceil_mode=True, count_include_pad=False)[0]
        s = np.diag(list(f) + [1]).astype(float)
        s[:3, 3] = (np.asarray(f) - 1) / 2
        return Grid(d, self.affine @ s)


def _points(shape, affine, dev, z0=0, z1=None) -> torch.Tensor:
    """mm coordinates [nx, ny, nz(z0:z1), 3] of a voxel grid."""
    z1 = shape[2] if z1 is None else z1
    ii = [torch.arange(n, dtype=torch.float32, device=dev) for n in (shape[0], shape[1])]
    ii.append(torch.arange(z0, z1, dtype=torch.float32, device=dev))
    ijk = torch.stack(torch.meshgrid(*ii, indexing="ij"), -1)
    return ijk @ _t(affine[:3, :3], dev).T + _t(affine[:3, 3], dev)


def _eval_grid(tmpl_affine, tmpl_shape, res):
    """Template grid pooled to `res` mm: pooling factor and affine."""
    vs = np.linalg.norm(tmpl_affine[:3, :3], axis=0)
    f = tuple(max(1, int(round(res / v))) for v in vs)
    s = np.diag(list(f) + [1]).astype(float)
    s[:3, 3] = (np.asarray(f) - 1) / 2
    shape = tuple(int(np.ceil(n / k)) for n, k in zip(tmpl_shape, f))
    return f, tmpl_affine @ s, shape


def _pool_factor(affine, res):
    return tuple(max(1, int(round(res / v))) for v in np.linalg.norm(affine[:3, :3], axis=0))


def dice_per_class(p: torch.Tensor, q: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """2|p q| / (|p| + |q|) per class; p, q [C, ...] in [0, 1]."""
    dims = tuple(range(1, p.ndim))
    return (2 * (p * q).sum(dims) + eps).div(p.sum(dims) + q.sum(dims) + eps)


def soft_dice(p: torch.Tensor, q: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
    """1 - (weighted) mean over classes of the soft Dice."""
    d = dice_per_class(p, q)
    return 1 - (d.mean() if w is None else (d * w).sum() / w.sum())


def lncc(a: torch.Tensor, b: torch.Tensor, win: int, w: torch.Tensor | None = None) -> torch.Tensor:
    """1 - local squared normalised cross-correlation over cubic windows (a, b [X, Y, Z])."""
    def box(x):
        return F.avg_pool3d(x[None, None], win, stride=1, padding=win // 2, count_include_pad=False)[0, 0]
    ma, mb = box(a), box(b)
    va = (box(a * a) - ma * ma).clamp_min(0)
    vb = (box(b * b) - mb * mb).clamp_min(0)
    cab = box(a * b) - ma * mb
    cc = cab * cab / (va * vb + 1e-5)
    if w is None:
        return 1 - cc.mean()
    return 1 - (cc * w).sum() / w.sum().clamp_min(1e-6)


# ---------------------------------------------------------------------------- SVF

class VelocityGrid:
    """Axis-aligned grid in MNI mm (RAS, positive steps) carrying v [1, 3, nx, ny, nz] in mm."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, spacing: float, dev):
        self.spacing = float(spacing)
        self.shape = tuple(int(np.ceil((h - l) / spacing)) + 1 for l, h in zip(lo, hi))
        self.lo = np.asarray(lo, float)
        self.affine = np.diag([spacing] * 3 + [1.0])
        self.affine[:3, 3] = self.lo
        n = torch.tensor(self.shape, dtype=torch.float32, device=dev)
        ijk = torch.stack(torch.meshgrid(*[torch.arange(k, dtype=torch.float32, device=dev) for k in self.shape],
                                         indexing="ij"), -1)
        self._id = (2 * ijk / (n - 1) - 1).flip(-1)[None]          # identity, grid_sample order
        self._scale = (2 / ((n - 1) * spacing)).flip(0)             # mm -> normalised, grid order

    def compose_self(self, u: torch.Tensor) -> torch.Tensor:
        """u + u o (id + u), u [1, 3, nx, ny, nz] mm."""
        g = self._id + u.permute(0, 2, 3, 4, 1).flip(-1) * self._scale
        return u + F.grid_sample(u, g, mode="bilinear", padding_mode="border", align_corners=True)

    def exp(self, v: torch.Tensor, steps: int) -> torch.Tensor:
        u = v / (2 ** steps)
        for _ in range(steps):
            u = self.compose_self(u)
        return u

    def resample(self, v: torch.Tensor, other: "VelocityGrid") -> torch.Tensor:
        """v of `other` resampled onto this grid."""
        pts = _points(self.shape, self.affine, v.device)
        return Grid(v[0], other.affine).sample(pts, padding="border")[None]


def regulariser(v: torch.Tensor, h: float, w: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Diffusion (mean squared first differences) and bending (second differences) of v [1,3,...],
    optionally weighted per voxel by w [nx, ny, nz] (relative stiffness)."""
    def mean(x, d, n):
        if w is None:
            return x.mean()
        sl = [slice(None)] * 3
        sl[d - 2] = slice(0, w.shape[d - 2] - n)
        return (x * w[tuple(sl)]).mean()
    diff = sum(mean((v.diff(dim=d) / h) ** 2, d, 1) for d in (2, 3, 4))
    bend = sum(mean((v.diff(n=2, dim=d) / h ** 2) ** 2, d, 2) for d in (2, 3, 4)) * h ** 2
    return diff, bend


# ---------------------------------------------------------------------------- registration

def signed_distance(mask: np.ndarray, spacing, clip: float) -> np.ndarray:
    """Signed Euclidean distance (mm) to the boundary of a binary mask, negative inside, clipped to
    +-clip (distance-map matching as in Paragios et al. 2003). Voxel centres: a voxel next to the
    boundary is at +-0.5 voxel."""
    import scipy.ndimage as ndi

    mask = np.asarray(mask, bool)
    if not mask.any() or mask.all():
        return np.full(mask.shape, clip if not mask.any() else -clip, np.float32)
    sp = np.asarray(spacing, float)
    out = ndi.distance_transform_edt(~mask, sampling=sp) - ndi.distance_transform_edt(mask, sampling=sp)
    out = out - np.sign(out) * 0.5 * sp.min()
    return np.clip(out, -clip, clip).astype(np.float32)


def _onehot(labels: np.ndarray, classes: dict, dev) -> torch.Tensor:
    lab = torch.as_tensor(labels.astype(np.int64), device=dev)
    return torch.stack([torch.isin(lab, torch.tensor(v, device=dev)).float() for v in classes.values()])


def _norm_t1(t1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vals = t1[mask]
    if vals.numel() > 2_000_000:
        vals = vals[torch.randperm(vals.numel(), device=vals.device, generator=None)[:2_000_000]]
    lo, hi = torch.quantile(vals, 0.01), torch.quantile(vals, 0.99)
    return ((t1 - lo) / (hi - lo).clamp_min(1e-6)).clamp(-0.5, 1.5)


def load_template(classes: dict, dev, template_dir: Path = TEMPLATE_DIR):
    img = nib.load(Path(template_dir) / TEMPLATE_FILE)
    prob = torch.as_tensor(np.asarray(img.dataobj), device=dev)
    prob = prob.float() / 255 if prob.dtype == torch.uint8 else prob.float()
    merged = torch.stack([prob[..., list(v)].sum(-1) for v in classes.values()])
    head = 1 - prob[..., 0]
    mni = nib.load(mni_t1())
    if mni.shape[:3] != img.shape[:3] or not np.allclose(mni.affine, img.affine):
        raise ValueError("label template and MNI152 T1 are on different grids")
    t1 = _t(_vol(mni).astype(np.float32), dev)
    t1 = _norm_t1(t1, prob[..., 1:3].sum(-1) > 0.5)
    return Grid(torch.cat([merged, t1[None], head[None]]), img.affine)


def _principal_init(sub_brain: np.ndarray, sub_aff, tpl_brain: np.ndarray, tpl_aff):
    """MNI -> subject affine from brain centroids, volumes and principal axes (rotation kept only
    when it is small, so swapped or flipped axes of a near-spherical brain cannot mislead it)."""
    def moments(mask, aff):
        ijk = np.argwhere(mask > 0.5)
        x = ijk @ aff[:3, :3].T + aff[:3, 3]
        vol = len(ijk) * abs(np.linalg.det(aff[:3, :3]))
        return x.mean(0), np.cov(x.T), vol
    cs, Cs, vs = moments(sub_brain, sub_aff)
    ct, Ct, vt = moments(tpl_brain, tpl_aff)
    s = (vs / vt) ** (1 / 3)
    es, Vs = np.linalg.eigh(Cs)
    et, Vt = np.linalg.eigh(Ct)
    Vs = Vs * np.sign((Vs * Vt).sum(0))[None]
    R = Vs @ Vt.T
    if np.linalg.det(R) < 0 or np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) > 20:
        R = np.eye(3)
    L = s * R
    A = np.eye(4)
    A[:3, :3] = L
    A[:3, 3] = cs - L @ ct
    return A


def _sub_pyramid(sub: Grid, res: float) -> Grid:
    return sub.pooled(_pool_factor(sub.affine, res))


def register(t1_img, labels_img, out_m2m: Path, device: str = "cuda", params: Params | None = None,
             template_dir: Path = TEMPLATE_DIR, write: bool = True, init_affine: np.ndarray | None = None) -> dict:
    """Register a subject (T1 + charm-valued labels on the T1 grid) to MNI and write charm's toMNI
    warps and coregistrationMatrices.mat into out_m2m. Returns timings (s), losses and peak GPU
    memory (GB); with write=False the transform is returned as "transform" instead of written."""
    p = params or Params()
    dev = torch.device(device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    timings, losses = {}, {}

    t1_aff = np.asarray(t1_img.affine, float)
    labels = _vol(labels_img)
    if labels.shape[:3] != t1_img.shape[:3] or not np.allclose(labels_img.affine, t1_aff, atol=1e-3):
        raise ValueError("labels are not on the T1 grid")
    tpl = load_template(p.classes, dev, template_dir)
    nc = len(p.classes)
    oh = _onehot(labels, p.classes, dev)
    t1 = _t(_vol(t1_img).astype(np.float32), dev)
    t1 = _norm_t1(t1, oh[0] > 0.5)
    sub = Grid(torch.cat([oh, t1[None], torch.ones_like(t1)[None]]), t1_aff)   # last: field of view
    del oh, t1
    sdt_idx = []
    if p.w_sdt > 0:   # signed distance channels appended after the field of view (sub) / head (tpl)
        sp_s = np.linalg.norm(t1_aff[:3, :3], axis=0)
        sp_t = np.linalg.norm(tpl.affine[:3, :3], axis=0)
        ds, dt = [], []
        for k in p.sdt_classes:
            ds.append(signed_distance(np.isin(labels, p.classes[k]), sp_s, p.sdt_clip))
            dt.append(signed_distance(tpl.data[list(p.classes).index(k)].cpu().numpy() > 0.5, sp_t, p.sdt_clip))
        sub = Grid(torch.cat([sub.data, _t(np.stack(ds), dev)]), t1_aff)
        tpl = Grid(torch.cat([tpl.data, _t(np.stack(dt), dev)]), tpl.affine)
        sdt_idx = list(range(nc + 2, nc + 2 + len(ds)))
        del ds, dt
    timings["load"] = time.perf_counter() - t0

    # --- affine, initialised from the brains' moments
    sub_brain = np.isin(labels, p.classes["brain"])
    tpl_brain = tpl.data[0].cpu().numpy()
    A0 = _principal_init(sub_brain, t1_aff, tpl_brain, tpl.affine) if init_affine is None else np.asarray(init_affine)
    c = _t(tpl.affine[:3, :3] @ (np.asarray(tpl.data.shape[1:]) / 2) + tpl.affine[:3, 3], dev)   # template centre
    L = _t(A0[:3, :3], dev).requires_grad_()
    tr = ((_t(A0[:3, :3] @ c.cpu().numpy() + A0[:3, 3], dev)) / 10).requires_grad_()   # subject(c) / 10 mm

    def affine_map(x):   # MNI mm -> subject mm
        return (x - c) @ L.T + tr * 10

    names = list(p.classes)
    ssd_idx = [names.index(k) for k in p.ssd_classes] if p.ssd_classes else list(range(nc))

    cw = _t([p.class_weights.get(k, 1.0) for k in names], dev)
    cw_svf = cw if p.svf_class_weights is None else _t([p.svf_class_weights.get(k, 1.0) for k in names], dev)

    def similarity(ys, sub_l, tpl_l, svf=False, last=False):
        s = sub_l.sample(ys)
        fov = s[nc + 1]   # template voxels that map outside the T1 field of view do not count
        dice = soft_dice(s[:nc], tpl_l[:nc] * fov, cw_svf if svf else cw)
        ncc = (lncc(s[nc], tpl_l[nc], p.ncc_win, tpl_l[nc + 1] * fov) if p.w_ncc > 0
               else torch.zeros((), device=dev))
        if svf and p.w_ssd > 0:   # every boundary voxel counts the same, whatever the size of its class
            dice = dice + p.w_ssd * ((s[ssd_idx] - tpl_l[ssd_idx]) ** 2 * fov).sum(0).mean() * 100
        if svf and sdt_idx:   # boundary distances: mean over template voxels within the clip band
            ds, dt = s[sdt_idx], tpl_l[sdt_idx]
            band = (dt.abs() < p.sdt_clip).float().sum().clamp_min(1)
            dice = dice + p.w_sdt * (((ds - dt) / p.sdt_clip) ** 2 * fov).sum() / band
        if svf and last and p.w_ncc_brain > 0:
            brain = tpl_l[names.index("brain")] * fov
            ncc_b = lncc(s[nc], tpl_l[nc], p.ncc_win, brain)
            return dice + p.w_ncc * ncc + p.w_ncc_brain * ncc_b, dice, ncc + ncc_b
        return dice + p.w_ncc * ncc, dice, ncc

    t_a = time.perf_counter()
    opt = torch.optim.Adam([{"params": [L], "lr": p.lr_affine}, {"params": [tr], "lr": p.lr_affine * 10}])
    for res, iters in p.affine_stages:
        f, eaff, eshape = _eval_grid(tpl.affine, tpl.data.shape[1:], res)
        tpl_l = tpl.pooled(f).data
        sub_l = _sub_pyramid(sub, res)
        x = _points(eshape, eaff, dev)
        for _ in range(iters):
            opt.zero_grad()
            loss, dice, ncc = similarity(affine_map(x), sub_l, tpl_l)
            loss.backward()
            opt.step()
    if p.affine_stages:
        losses["affine"] = {"total": loss.item(), "dice": dice.item(), "ncc": ncc.item()}
    A = np.eye(4)
    Ld = L.detach().cpu().numpy().astype(float)
    A[:3, :3] = Ld
    A[:3, 3] = tr.detach().cpu().numpy() * 10 - Ld @ c.cpu().numpy()
    A_raw = A.copy()
    kind = p.frame
    if kind == "auto":
        kind = "ridge" if not p.init_warp and (Path(template_dir) / p.ridge_file).is_file() else "labels"
    cal = load_calibration(template_dir) if p.calibrate and kind == "labels" else None
    if cal is not None:
        A = calibrate(A, cal)
    if kind == "intensity":
        from . import frame as _frame   # frame imports this module
        fr = _frame.intensity_affine(t1_img, A_raw, _frame.FrameParams(**p.frame_params), device=device,
                                     labels=labels, label_template_dir=template_dir, classes=p.classes)
        A = np.asarray(fr["affine"], float)
        losses["frame"] = fr["loss"]
    elif kind == "ridge":
        from . import frame as _frame
        model = json.loads((Path(template_dir) / p.ridge_file).read_text())
        pb = Params(classes=dict(_frame.BRAIN_CLASSES), svf_stages=(), calibrate=False, frame="labels",
                    affine_stages=p.affine_stages, lr_affine=p.lr_affine)
        A_brain = register(t1_img, labels_img, None, device=device, params=pb, template_dir=template_dir,
                           write=False)["affine_raw"]
        A = _frame.apply_ridge([A_raw, np.asarray(A_brain)], model)
        losses["frame_brain_affine"] = A_brain
    elif kind != "labels":
        raise ValueError(f"unknown frame {p.frame!r}")
    losses["frame_kind"] = kind
    v_net = vg_net = None
    if p.init_warp:
        from . import warpnet as _warpnet   # warpnet imports this module
        net = _warpnet.load_net(p.init_warp, dev)
        if net[1].frame != "affine_raw" or not same_template(net[1].template_dir, template_dir):
            raise ValueError(f"{p.init_warp} was trained on the raw affine of {net[1].template_dir}; "
                             f"pass that as template_dir")
        A, cal = A_raw.copy(), None
        support = _warpnet.net_support(p.init_warp, dev) if p.net_fill else None
        v_net, vg_net = _warpnet.predict_velocity(net, t1_img, labels, A, dev, support)
        del net
    if cuda:
        torch.cuda.synchronize(dev)
    timings["affine"] = time.perf_counter() - t_a

    # --- SVF in MNI space, on a grid that covers the atlas grid and the T1 grid mapped to MNI
    t_n = time.perf_counter()
    atlas = nib.load(atlas_template())
    Ainv = np.linalg.inv(A)
    corners = np.array(np.meshgrid(*[[0, n - 1] for n in t1_img.shape[:3]], indexing="ij")).reshape(3, -1).T
    acorners = np.array(np.meshgrid(*[[0, n - 1] for n in atlas.shape[:3]], indexing="ij")).reshape(3, -1).T
    pts = np.vstack([nib.affines.apply_affine(Ainv @ t1_aff, corners), nib.affines.apply_affine(atlas.affine, acorners)])
    lo, hi = pts.min(0) - p.margin_mm, pts.max(0) + p.margin_mm
    A_t = _t(A, dev)

    def full_map(x, u, vg):   # MNI mm -> subject mm
        disp = Grid(u[0], vg.affine).sample(x, padding="border")
        return (x + disp.permute(*range(1, x.ndim), 0)) @ A_t[:3, :3].T + A_t[:3, 3]

    vg, v = None, None
    for si, (res, spacing, iters) in enumerate(p.svf_stages):
        last = si == len(p.svf_stages) - 1
        new = VelocityGrid(lo, hi, spacing, dev)
        v = torch.zeros((1, 3) + new.shape, device=dev) if vg is None else new.resample(v.detach(), vg)
        vg = new
        v.requires_grad_()
        base = None if v_net is None else new.resample(v_net, vg_net)   # v below is dv on top of it
        stiff = None
        if p.stiff_brain != 1.0:   # template brain probability on the velocity grid, smoothed
            bp = Grid(F.avg_pool3d(tpl.data[:1][None], 5, 1, 2, count_include_pad=False)[0], tpl.affine)
            stiff = 1 + (p.stiff_brain - 1) * bp.sample(_points(vg.shape, vg.affine, dev))[0]
        opt = torch.optim.Adam([v], lr=p.lr_svf)
        f, eaff, eshape = _eval_grid(tpl.affine, tpl.data.shape[1:], res)
        tpl_l = tpl.pooled(f).data
        sub_l = _sub_pyramid(sub, res)
        x = _points(eshape, eaff, dev)
        for _ in range(iters):
            opt.zero_grad()
            u = vg.exp(v if base is None else base + v, p.steps)
            loss_s, dice, ncc = similarity(full_map(x, u, vg), sub_l, tpl_l, svf=True, last=last)
            rd, rb = regulariser(v, vg.spacing, stiff)
            rs = p.svf_reg_scale[si] if si < len(p.svf_reg_scale) else 1.0
            loss = loss_s + rs * (p.w_diff * rd + p.w_bend * rb) + p.w_mag * (v ** 2).mean()
            if base is not None and p.w_init > 0:
                loss = loss + p.w_init * (v ** 2).mean()
            if p.w_jac > 0:
                loss = loss + p.w_jac * F.relu(p.jac_floor - _jacobian_det(u, vg.spacing)).pow(2).mean() * 1000
            loss.backward()
            opt.step()
    if vg is None:   # affine only, or the net's warp alone
        vg = VelocityGrid(lo, hi, 8.0 if v_net is None else min(p.out_spacing, vg_net.spacing), dev)
        v = torch.zeros((1, 3) + vg.shape, device=dev) if v_net is None else vg.resample(v_net, vg_net)
        f, eaff, eshape = _eval_grid(tpl.affine, tpl.data.shape[1:], 2.0)
        tpl_l, sub_l, x = tpl.pooled(f).data, _sub_pyramid(sub, 2.0), _points(eshape, eaff, dev)
    else:
        losses["svf"] = {"total": loss.item(), "dice": dice.item(), "ncc": ncc.item(), "diffusion": rd.item(),
                         "bending": rb.item()}
        if v_net is not None:   # v_net at its own resolution plus the refinement dv
            fine = VelocityGrid(lo, hi, min(p.out_spacing, vg_net.spacing), dev)
            v, vg = fine.resample(v_net, vg_net) + fine.resample(v.detach(), vg), fine
    v = v.detach()
    with torch.no_grad():
        if p.out_spacing < vg.spacing:   # exponentiate on a finer grid: closer inverse consistency
            fine = VelocityGrid(lo, hi, p.out_spacing, dev)
            v, vg = fine.resample(v, vg), fine
        u_fwd, u_inv = vg.exp(v, p.steps), vg.exp(-v, p.steps)
        # the warp's Jacobian determinant on the velocity grid (folding check)
        J = _jacobian_det(u_fwd, vg.spacing)
        losses["jacobian_min"] = float(J.min())
        s = sub_l.sample(full_map(x, u_fwd, vg))
        losses["dice_svf"] = dict(zip(names, dice_per_class(s[:nc], tpl_l[:nc] * s[nc + 1]).tolist()))
        s = sub_l.sample(x @ A_t[:3, :3].T + A_t[:3, 3])
        losses["dice_affine"] = dict(zip(names, dice_per_class(s[:nc], tpl_l[:nc] * s[nc + 1]).tolist()))
    if cuda:
        torch.cuda.synchronize(dev)
    timings["svf"] = time.perf_counter() - t_n

    out = {"timings": timings, "losses": losses, "affine": A.tolist(), "affine_raw": A_raw.tolist(),
           "calibrated": cal is not None}
    t_w = time.perf_counter()
    with torch.no_grad():
        m2c = _map_volume(atlas.shape[:3], atlas.affine, lambda x: full_map(x, u_fwd, vg), dev)
        Ainv_t = _t(Ainv, dev)

        def inv_map(y):   # subject mm -> MNI mm
            xa = y @ Ainv_t[:3, :3].T + Ainv_t[:3, 3]
            disp = Grid(u_inv[0], vg.affine).sample(xa, padding="border")
            return xa + disp.permute(*range(1, xa.ndim), 0)
        c2m = _map_volume(t1_img.shape[:3], t1_aff, inv_map, dev)
    if p.unfold_iters:
        t_u = time.perf_counter()
        losses["unfold"] = {"MNI2Conform_nonl": _unfold(m2c, atlas.affine, p.unfold_iters, dev),
                            "Conform2MNI_nonl": _unfold(c2m, t1_aff, p.unfold_iters, dev)}
        timings["unfold"] = time.perf_counter() - t_u
    if cuda:
        torch.cuda.synchronize(dev)
        out["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
    else:
        out["peak_mem_gb"] = 0.0
    timings["maps"] = time.perf_counter() - t_w
    timings["gpu_total"] = time.perf_counter() - t0
    if write:
        t_f = time.perf_counter()
        write_m2m(Path(out_m2m), m2c, atlas.affine, c2m, t1_aff, A)
        timings["write"] = time.perf_counter() - t_f
    else:
        out["transform"] = {"mni2conform": m2c, "atlas_affine": atlas.affine, "conform2mni": c2m,
                            "t1_affine": t1_aff, "A": A}   # write_m2m's arguments
    timings["total"] = time.perf_counter() - t0
    return out


def _jacobian_det(u: torch.Tensor, h: float) -> torch.Tensor:
    """det(I + grad u) by central differences, u [1, 3, nx, ny, nz] mm."""
    g = torch.stack([torch.gradient(u[0, i], spacing=h)[j] for i in range(3) for j in range(3)]).reshape(3, 3, *u.shape[2:])
    g = g + torch.eye(3, device=u.device)[:, :, None, None, None]
    return torch.linalg.det(g.permute(2, 3, 4, 0, 1))


def _map_volume(shape, affine, fn, dev, chunk: int = 32) -> np.ndarray:
    """fn applied to every voxel centre of a grid (mm in, mm out), in z slabs: [nx, ny, nz, 3]."""
    out = np.empty(tuple(shape) + (3,), np.float32)
    for z0 in range(0, shape[2], chunk):
        z1 = min(z0 + chunk, shape[2])
        out[:, :, z0:z1] = fn(_points(shape, affine, dev, z0, z1)).cpu().numpy()
    return out


def _map_det(m: np.ndarray, affine, chunk: int = 32, dev="cpu") -> np.ndarray:
    """Jacobian determinant of a coordinate map [nx, ny, nz, 3] (world mm per voxel) with respect to its
    grid's world coordinates, as eval_register.map_jacobian (central differences, one-sided at the volume
    border), in z slabs on `dev`."""
    Minv = torch.as_tensor(np.linalg.inv(np.asarray(affine, float)[:3, :3]), dtype=torch.float64, device=dev)
    nz = m.shape[2]
    out = np.empty(m.shape[:3], np.float32)
    for z0 in range(0, nz, chunk):
        z1 = min(z0 + chunk, nz)
        a, b = max(z0 - 1, 0), min(z1 + 1, nz)
        s = torch.as_tensor(np.ascontiguousarray(m[:, :, a:b]), device=dev).double()
        dims = [d for d in range(3) if s.shape[d] > 1]
        G = torch.zeros(s.shape[:3] + (3, 3), dtype=s.dtype, device=dev)
        for i in range(3):
            for d, g in zip(dims, torch.gradient(s[..., i], dim=dims)):
                G[..., i, d] = g
        out[:, :, z0:z1] = torch.linalg.det(G[:, :, z0 - a:z0 - a + z1 - z0] @ Minv).float().cpu().numpy()
    return out


def _unfold(m: np.ndarray, affine, iters: int, dev="cpu") -> dict:
    """Repair a written coordinate map in place where its Jacobian determinant is <= 0: the folded voxels
    and their 26 neighbours are replaced by their 3x3x3 mean, then the determinant is recomputed around
    them, at most `iters` times. Returns the folded voxel count before, the repaired voxel count, the
    rounds used and the min determinant after (> 0: no folding left)."""
    from scipy.ndimage import binary_dilation, uniform_filter

    bad = _map_det(m, affine, dev=dev) <= 0
    n0, fixed, it = int(bad.sum()), np.zeros(m.shape[:3], bool), 0
    while bad.any() and it < iters:
        it += 1
        idx = np.argwhere(bad)
        lo, hi = np.maximum(idx.min(0) - 3, 0), np.minimum(idx.max(0) + 4, m.shape[:3])
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        sub = m[sl]   # a view: edits m
        fix = binary_dilation(bad[sl], np.ones((3, 3, 3), bool))
        sub[fix] = uniform_filter(sub, size=(3, 3, 3, 1), mode="nearest")[fix]
        fixed[sl] |= fix
        # determinants of the box; its own faces are one-sided there, so judge only its interior
        # (and the true volume border, where the full-volume determinant is one-sided too)
        d = _map_det(m[sl], affine, dev=dev)
        inner = np.ones(d.shape, bool)
        for ax in range(3):
            if lo[ax] > 0:
                inner[(slice(None),) * ax + (0,)] = False
            if hi[ax] < m.shape[ax]:
                inner[(slice(None),) * ax + (-1,)] = False
        bad = np.zeros(m.shape[:3], bool)
        bad[sl] = (d <= 0) & inner
    return {"folded": n0, "repaired_voxels": int(fixed.sum()), "rounds": it,
            "min_det": float(_map_det(m, affine, dev=dev).min()) if n0 else None}


def write_m2m(out_m2m: Path, m2c: np.ndarray, atlas_affine, c2m: np.ndarray, t1_affine, A: np.ndarray):
    """charm's file formats: MNI2Conform float64 on the atlas grid, Conform2MNI float32 on the T1 grid
    (both sform code 2, as nib writes charm's), coregistrationMatrices.mat."""
    (out_m2m / "toMNI").mkdir(parents=True, exist_ok=True)
    (out_m2m / "segmentation").mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(m2c.astype(np.float64), atlas_affine), out_m2m / "toMNI/MNI2Conform_nonl.nii.gz")
    nib.save(nib.Nifti1Image(c2m.astype(np.float32), t1_affine), out_m2m / "toMNI/Conform2MNI_nonl.nii.gz")
    scipy.io.savemat(out_m2m / "segmentation/coregistrationMatrices.mat", {
        "imageToImageTransformMatrix": np.linalg.inv(t1_affine) @ A @ np.asarray(atlas_affine),
        "worldToWorldTransformMatrix": A})
