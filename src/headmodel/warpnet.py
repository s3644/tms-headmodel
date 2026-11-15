"""WP3b: a supervised warp network that reproduces charm's MNI warp from charm's own warps.

Why: label matching (register.py) leaves the correspondence free where the labels carry no
information (tangential on the scalp, within a gyrus); there charm's warp follows its own affine
frame and atlas stiffness. A network trained on charm's warps of the train cases learns both.

Frame. For a subject with our affine A (MNI mm -> subject mm, register()'s affine stage, raw, i.e. not
calibrated), everything lives on a fixed 2 mm MNI grid (GRID_*; it covers charm's atlas grid, and its
points are atlas voxel centres):
  input   x_in(x) = subject channels sampled at A(x): fractions of charm labels LABELS (label 4 counts
          as 7), the normalised T1 (register._norm_t1), the T1 field of view; plus 3 coordinate
          channels (CoordConv, Liu et al. 2018) added on the fly
  target  r(x) = A^-1(charm(x)) - x, charm(x) = charm's toMNI/MNI2Conform_nonl at x (subject mm);
          masked where charm left zeros or x is off its atlas grid
  output  a stationary velocity field v (mm) on the grid; u = exp(v) by scaling and squaring
          (Arsigny et al. 2006; Ashburner 2007; as in VoxelMorph, Dalca et al. 2019), so the map
          subject(x) = A(x + u(x)) is diffeomorphic and its inverse is exp(-v) -- register.py's
          representation, so register() can start its SVF stage from v (Params.init_warp)
Model: a 3-D U-Net (Ronneberger et al. 2015; Cicek et al. 2016), instance norm, leaky ReLU, 4
downsamplings (2 -> 32 mm). Loss: weighted end-point error (mm, brain and skin band weighted up) +
end-point error at the 10 targets and the 80 EEG10-10 / fiducial MNI positions + a small diffusion
penalty on v (round 5: the gated points weighted up, TrainConfig.w_gate). Augmentation: a random
small affine T (MNI -> MNI), applied consistently: the input becomes x_in(T(x)) (what affine A o T would give) and the target T^-1(x + r(x)) - x. No flips (the
frame is lateralised). Model selection on the validation cases (runs/wp3b/val_cases.json, never test
cases); the label template behind A is runs/wp3b/template_trainminusval (no validation case in it).

Machine sharing: training yields the GPU (checkpoint, clean exit, resumable) whenever an nnU-Net job
runs or head-model's round 1 has its slot (yield_reason).

Licence: trained on charm runs of CC BY-NC 4.0 data, so weights and outputs are research only.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/wp3b/warpnet"
CACHE = OUT / "cache"
TEMPLATE = REPO / "runs/wp3b/template_trainminusval"
VAL_FILE = REPO / "runs/wp3b/val_cases.json"
ROUND1_LOG = REPO / "runs/logs/train_round1.log"

LABELS = (1, 2, 3, 5, 6, 7, 8, 9, 10)          # charm labels after the surface update (4 -> 7)
N_IN = len(LABELS) + 2                          # + T1 + field of view (cached); + 3 coordinates on the fly
# 2 mm MNI grid (RAS, positive steps), points on charm's atlas voxel centres; divisible by 16
GRID_LO = (-95.0, -134.0, -174.0)
GRID_SHAPE = (96, 128, 144)
GRID_SPACING = 2.0

# rTMS targets in MNI152 mm (literature presets): left targets, right = x-mirror (as eval_register)
_LEFT = {"dlpfc_sgacc": (-38, 44, 26), "dlpfc_5cm": (-41, 16, 54), "dlpfc_ba46": (-44, 40, 29),
         "dlpfc_ba9": (-36, 39, 43), "m1_hand": (-37, -21, 58)}
TARGETS = {f"{k}_{s}": (x if s == "l" else -x, y, z) for k, (x, y, z) in _LEFT.items() for s in "lr"}
CAP = "EEG10-10_UI_Jurak_2007"


# the points the gate judges (rule wp3_gate_rtms_set): the targets, these 13 cap electrodes and the 4 fiducials
GATE_EEG = ("F3", "F4", "Fz", "Fpz", "Fp1", "Fp2", "Cz", "C3", "C4", "FCz", "T7", "T8", "Oz")
GATE_FIDUCIALS = ("Nz", "Iz", "LPA", "RPA")


def point_kinds() -> list[tuple[str, str]]:
    """(kind, name) of each mni_points() row: ("target", key), or the cap's type ("Electrode",
    "ReferenceElectrode" for Cz, "Fiducial") and name."""
    import csv
    with open(register.simnibs_dir() / f"resources/ElectrodeCaps_MNI/{CAP}.csv") as fh:
        cap = [(row[0], row[4]) for row in csv.reader(fh) if len(row) >= 5]
    return [("target", k) for k in TARGETS] + cap


def gate_points() -> dict[str, np.ndarray]:
    """bool masks over mni_points() rows: targets, the rTMS EEG set, the fiducials (wp3_gate_rtms_set)."""
    kinds = point_kinds()
    return {"targets": np.array([k == "target" for k, _ in kinds]),
            "rtms": np.array([k in ("Electrode", "ReferenceElectrode") and n in GATE_EEG for k, n in kinds]),  # Cz: reference
            "fid": np.array([k == "Fiducial" and n in GATE_FIDUCIALS for k, n in kinds])}


def mni_points() -> tuple[np.ndarray, int]:
    """10 targets then the 80 positions of the EEG10-10_UI_Jurak_2007 cap (incl. 4 fiducials), MNI mm."""
    pts = [list(map(float, v)) for v in TARGETS.values()]
    import csv
    with open(register.simnibs_dir() / f"resources/ElectrodeCaps_MNI/{CAP}.csv") as fh:
        pts += [[float(v) for v in row[1:4]] for row in csv.reader(fh) if len(row) >= 5]
    return np.array(pts), len(TARGETS)


@dataclass
class GridSpec:
    lo: tuple = GRID_LO
    shape: tuple = GRID_SHAPE
    spacing: float = GRID_SPACING

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.lo) + self.spacing * (np.asarray(self.shape) - 1)

    @property
    def affine(self) -> np.ndarray:
        a = np.diag([self.spacing] * 3 + [1.0])
        a[:3, 3] = self.lo
        return a

    def velocity_grid(self, dev) -> register.VelocityGrid:
        vg = register.VelocityGrid(np.asarray(self.lo, float), self.hi, self.spacing, dev)
        assert tuple(vg.shape) == tuple(self.shape)
        return vg

    def points(self, dev) -> torch.Tensor:
        return register._points(self.shape, self.affine, dev)


# ---------------------------------------------------------------------------- inputs and targets

def subject_grid(t1_img, labels: np.ndarray, dev) -> register.Grid:
    """Subject channels on the T1 grid pooled to 2 mm: label fractions, normalised T1, field of view."""
    lab = torch.as_tensor(np.where(labels == 4, 7, labels).astype(np.int64), device=dev)
    oh = torch.stack([(lab == k).float() for k in LABELS])
    del lab
    t1 = register._t(register._vol(t1_img).astype(np.float32), dev)
    t1 = register._norm_t1(t1, (oh[0] + oh[1]) > 0.5)
    g = register.Grid(torch.cat([oh, t1[None], torch.ones_like(t1)[None]]), np.asarray(t1_img.affine, float))
    del oh, t1
    return register._sub_pyramid(g, GRID_SPACING)


def net_input(t1_img, labels: np.ndarray, A: np.ndarray, dev, spec: GridSpec | None = None) -> torch.Tensor:
    """[N_IN, *shape]: subject channels at A(x) for every grid point x."""
    spec = spec or GridSpec()
    sub = subject_grid(t1_img, labels, dev)
    x = spec.points(dev)
    A_t = register._t(A, dev)
    out = sub.sample(x @ A_t[:3, :3].T + A_t[:3, 3])
    del sub
    return out


def sample_charm_map(m2m: Path, spec: GridSpec | None = None, pts: np.ndarray | None = None):
    """charm's MNI2Conform_nonl (subject mm) at the grid points and at `pts` (trilinear, as the
    simnibs reader samples it), with validity (charm leaves zeros; off its grid counts as invalid).
    CPU. Returns (y [*shape, 3] float64, valid [*shape] bool, y_pts [P, 3], valid_pts [P])."""
    import scipy.ndimage as ndi

    spec = spec or GridSpec()
    img = nib.load(Path(m2m) / "toMNI/MNI2Conform_nonl.nii.gz")
    d = np.asarray(img.dataobj, np.float64).reshape(img.shape[:3] + (3,))
    ok = (d != 0).any(-1).astype(np.float32)
    inv = np.linalg.inv(img.affine)

    def at(xyz):
        ijk = nib.affines.apply_affine(inv, xyz.reshape(-1, 3)).T
        y = np.stack([ndi.map_coordinates(d[..., c], ijk, order=1, cval=0.0) for c in range(3)], -1)
        v = ndi.map_coordinates(ok, ijk, order=1, cval=0.0) > 0.999
        return y, v

    xg = nib.affines.apply_affine(spec.affine, np.indices(spec.shape).reshape(3, -1).T)
    y, v = at(xg)
    out = (y.reshape(*spec.shape, 3), v.reshape(spec.shape))
    if pts is None:
        return out + (None, None)
    yp, vp = at(np.asarray(pts, float))
    return out + (yp, vp)


def residual(y: np.ndarray, A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """r = A^-1(y) - x (mm), y subject mm, x MNI mm; any leading shape [..., 3]."""
    Ai = np.linalg.inv(A)
    return y @ Ai[:3, :3].T + Ai[:3, 3] - x


# ---------------------------------------------------------------------------- model

class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """Plain 3-D U-Net -> velocity field (mm). Spatial size must be divisible by 2**(len(ch)-1)."""

    def __init__(self, cin: int = N_IN + 3, ch: tuple = (16, 32, 64, 128, 192)):
        super().__init__()
        self.enc = nn.ModuleList([_Block(cin if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.down = nn.ModuleList([nn.Conv3d(ch[i], ch[i], 2, stride=2) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([_Block(ch[i] + ch[i + 1], ch[i]) for i in range(len(ch) - 1)])
        self.head = nn.Conv3d(ch[0], 3, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, size=skips[i].shape[2:], mode="trilinear", align_corners=False)
            x = self.dec[i](torch.cat([skips[i], x], 1))
        return self.head(x)


def coords(spec: GridSpec, dev, batch: int = 1) -> torch.Tensor:
    """[B, 3, *shape] grid coordinates scaled to about [-1, 1] (MNI mm / 100)."""
    return (spec.points(dev) / 100.0).permute(3, 0, 1, 2)[None].expand(batch, -1, -1, -1, -1)


@dataclass
class NetConfig:
    channels: tuple = (16, 32, 64, 128, 192)
    lo: tuple = GRID_LO
    shape: tuple = GRID_SHAPE
    spacing: float = GRID_SPACING
    steps: int = 7                      # scaling and squaring
    frame: str = "affine_raw"           # which affine of register() the input is resampled with
    template_dir: str = str(TEMPLATE)   # label template behind that affine

    @property
    def spec(self) -> GridSpec:
        return GridSpec(tuple(self.lo), tuple(self.shape), self.spacing)


def build_model(cfg: NetConfig) -> UNet3D:
    return UNet3D(N_IN + 3, tuple(cfg.channels))


def velocity(model: nn.Module, x_in: torch.Tensor, spec: GridSpec) -> torch.Tensor:
    """x_in [B, N_IN, *shape] -> v [B, 3, *shape] (mm, float32); bf16 autocast on CUDA."""
    inp = torch.cat([x_in.float(), coords(spec, x_in.device, x_in.shape[0])], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x_in.is_cuda):
        v = model(inp)
    return v.float()


def load_net(path: str | Path, dev) -> tuple[nn.Module, NetConfig]:
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = NetConfig(**ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), cfg


@torch.no_grad()
def predict_velocity(path_or_net, t1_img, labels: np.ndarray, A: np.ndarray, dev, support: torch.Tensor | None = None):
    """register() hook: v [1, 3, *shape] (mm) on the net's MNI grid and that grid (VelocityGrid).
    support (bool [*shape], net_support()): v outside it is replaced by harmonic_fill (folding fix)."""
    model, cfg = load_net(path_or_net, dev) if isinstance(path_or_net, (str, Path)) else path_or_net
    spec = cfg.spec
    x_in = net_input(t1_img, labels, A, dev, spec)[None]
    v = velocity(model, x_in, spec)
    if support is not None:
        v = harmonic_fill(v, support.to(v.device))
    return v, spec.velocity_grid(dev)


# ---------------------------------------------------------------------------- outside the training support

SUPPORT_FRAC = 0.5   # a voxel is supported when charm's map is valid there in at least this share of train cases


def support_mask(cases: list[str], cache: Path = CACHE, frac: float = SUPPORT_FRAC) -> np.ndarray:
    """bool [*shape]: grid voxels where charm's map was valid (so the net was supervised) in >= frac of
    the cached `cases`."""
    acc = None
    for c in cases:
        with np.load(Path(cache) / f"{c}.npz") as z:
            m = z["valid"].astype(np.float32)
        acc = m if acc is None else acc + m
    if acc is None:
        raise ValueError("no cases")
    return acc >= frac * len(cases)


def net_support(ckpt: str | Path, dev="cpu", cache: Path = CACHE) -> torch.Tensor | None:
    """The training support of a checkpoint: its own "support" entry, else support_mask() of its
    train_cases, cached beside it as <stem>_support.npz (with the case list it was made from); None
    for a checkpoint with neither (no fill)."""
    ckpt = Path(ckpt)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "support" in ck:
        return torch.as_tensor(np.asarray(ck["support"], bool), device=dev)
    if not ck.get("train_cases"):
        return None
    cases = sorted(ck["train_cases"])
    f = ckpt.with_name(f"{ckpt.stem}_support.npz")
    if f.is_file():
        with np.load(f) as z:
            if sorted(z["cases"].tolist()) == cases and float(z["frac"]) == SUPPORT_FRAC:
                return torch.as_tensor(z["support"], device=dev)
    m = support_mask(cases, cache)
    tmp = f.with_name(f".{f.stem}.{os.getpid()}.npz")
    np.savez_compressed(tmp, support=m, cases=np.array(cases), frac=SUPPORT_FRAC)
    os.replace(tmp, f)
    return torch.as_tensor(m, device=dev)


def _avg6(u: torch.Tensor) -> torch.Tensor:
    """Mean of the 6 face neighbours (replicate padding), u [B, C, X, Y, Z]."""
    p = F.pad(u, (1, 1, 1, 1, 1, 1), mode="replicate")
    return (p[:, :, 2:, 1:-1, 1:-1] + p[:, :, :-2, 1:-1, 1:-1] + p[:, :, 1:-1, 2:, 1:-1] + p[:, :, 1:-1, :-2, 1:-1]
            + p[:, :, 1:-1, 1:-1, 2:] + p[:, :, 1:-1, 1:-1, :-2]) / 6


def harmonic_fill(v: torch.Tensor, known: torch.Tensor, iters: int = 60) -> torch.Tensor:
    """v [B, C, X, Y, Z] kept where known (bool [X, Y, Z]), elsewhere the discrete harmonic extension
    of those values (Laplace equation, zero-flux at the grid border), solved coarse to fine (Jacobi
    sweeps per level). Why: the net's output is unconstrained where charm's map never supervised it
    (below the head, outside charm's atlas grid) and folds there (WP3b task N); a harmonic field is as
    smooth as its boundary values allow and has no interior extrema."""
    k = known.to(v.dtype)[None, None]
    if not bool(known.any()):
        return torch.zeros_like(v)
    if min(v.shape[2:]) > 4:   # coarse solution as the start: masked 2x pooling
        num = F.avg_pool3d(v * k, 2, ceil_mode=True)
        den = F.avg_pool3d(k.expand_as(v[:, :1]), 2, ceil_mode=True)
        kc = den[0, 0] > 0
        coarse = harmonic_fill(num / den.clamp_min(1e-6), kc, iters)
        init = F.interpolate(coarse, size=v.shape[2:], mode="trilinear", align_corners=False)
    else:
        init = ((v * k).sum((2, 3, 4), keepdim=True) / k.sum()).expand_as(v)
    u = torch.where(k > 0, v, init)
    for _ in range(iters):
        u = torch.where(k > 0, v, _avg6(u))
    return u


# ---------------------------------------------------------------------------- cache

def cache_case(case: str, t1_img, labels: np.ndarray, charm_m2m: Path, dev, out_dir: Path = CACHE,
               template_dir: Path = TEMPLATE, spec: GridSpec | None = None, A: np.ndarray | None = None,
               charm=None) -> Path:
    """One training sample as <out_dir>/<case>.npz (float16 x, r; uint8 valid; float32 points).
    A: our raw affine (computed with template_dir when None). charm: sample_charm_map() output."""
    spec = spec or GridSpec()
    pts, nt = mni_points()
    if A is None:
        res = register.register(t1_img, nib.Nifti1Image(labels, t1_img.affine), None, device=str(dev), write=False,
                                params=register.Params(svf_stages=(), calibrate=False, frame="labels"), template_dir=template_dir)
        A = np.asarray(res["affine_raw"])
    y, valid, yp, vp = charm if charm is not None else sample_charm_map(charm_m2m, spec, pts)
    xg = nib.affines.apply_affine(spec.affine, np.indices(spec.shape).reshape(3, -1).T).reshape(*spec.shape, 3)
    r = residual(y, A, xg)
    r[~valid] = 0
    rp = residual(yp, A, pts)
    x = net_input(t1_img, labels, A, dev, spec).cpu().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{case}.npz"
    tmp = out_dir / f".{case}.{os.getpid()}.npz"
    np.savez_compressed(tmp, x=x.astype(np.float16), r=np.moveaxis(r, -1, 0).astype(np.float16),
                        valid=valid.astype(np.uint8), pts=pts.astype(np.float32), r_pts=rp.astype(np.float32),
                        valid_pts=vp.astype(np.uint8), n_targets=nt, A=A, lo=np.asarray(spec.lo),
                        shape=np.asarray(spec.shape), spacing=spec.spacing)
    os.replace(tmp, f)
    return f


def load_cached(f: Path) -> dict:
    z = np.load(f)
    return {"case": Path(f).stem, "x": torch.from_numpy(z["x"]), "r": torch.from_numpy(z["r"]),
            "valid": torch.from_numpy(z["valid"].astype(bool)), "pts": torch.from_numpy(z["pts"]),
            "r_pts": torch.from_numpy(z["r_pts"]), "valid_pts": torch.from_numpy(z["valid_pts"].astype(bool)),
            "n_targets": int(z["n_targets"]), "A": torch.from_numpy(z["A"].astype(np.float32))}


# ---------------------------------------------------------------------------- augmentation and loss

def random_affine(rng: np.random.Generator, rot_deg=4.0, scale=0.04, shear=0.02, trans=4.0,
                  centre=(0.0, -18.0, 10.0)) -> np.ndarray:
    """Small random MNI -> MNI affine about `centre` (no flips: det > 0)."""
    a = np.radians(rng.uniform(-rot_deg, rot_deg, 3))
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    R = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    S = np.diag(np.exp(rng.uniform(-scale, scale, 3)))
    H = np.eye(3) + np.triu(rng.uniform(-shear, shear, (3, 3)), 1)
    M = R @ S @ H
    T = np.eye(4)
    T[:3, :3] = M
    c = np.asarray(centre)
    T[:3, 3] = c - M @ c + rng.uniform(-trans, trans, 3)
    return T


def augment(x_in: torch.Tensor, r: torch.Tensor, r_pts: torch.Tensor, pts: torch.Tensor, T: np.ndarray,
            spec: GridSpec):
    """Consistent augmentation by T (MNI -> MNI) for one sample: as if our affine were A o T.
    x_in [C, *shape], r [3, *shape], pts / r_pts [P, 3] (MNI mm). Returns the new x_in, r, r_pts."""
    dev = x_in.device
    T_t = register._t(T, dev)
    Ti = register._t(np.linalg.inv(T), dev)
    xg = spec.points(dev)
    g = register.Grid(x_in.float(), spec.affine)
    x_new = g.sample(xg @ T_t[:3, :3].T + T_t[:3, 3])              # x_in(T(x)), zeros off the grid
    y = xg + r.float().permute(1, 2, 3, 0)                         # x + r(x)
    r_new = (y @ Ti[:3, :3].T + Ti[:3, 3] - xg).permute(3, 0, 1, 2)
    yp = pts + r_pts
    rp_new = yp @ Ti[:3, :3].T + Ti[:3, 3] - pts
    return x_new, r_new, rp_new


def weights(x_in: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per voxel EPE weight [B, *shape]: valid * (0.2 + head + 2 brain + 3 skin/head boundary band)."""
    lab = x_in[:, :len(LABELS)].float()
    head = lab.sum(1, keepdim=True).clamp(0, 1)
    brain = lab[:, 0:2].sum(1, keepdim=True)
    band = (F.max_pool3d(head, 5, 1, 2) + F.max_pool3d(-head, 5, 1, 2)).clamp_min(0)   # dilation - erosion
    w = 0.2 + head + 2 * brain + 3 * (band > 0.05).float()
    return w[:, 0] * valid.float()


def diffusion(v: torch.Tensor, h: float) -> torch.Tensor:
    return sum(((v.diff(dim=d) / h) ** 2).mean() for d in (2, 3, 4))


def point_errors(u: torch.Tensor, vg: register.VelocityGrid, pts: torch.Tensor, r_pts: torch.Tensor,
                 L: torch.Tensor | None = None) -> torch.Tensor:
    """|u(p) - r_p| per point (MNI mm, or subject mm with L = A[:3, :3]); u [3, *shape]."""
    up = register.Grid(u, vg.affine).sample(pts, padding="border").T
    d = up - r_pts
    if L is not None:
        d = d @ L.T
    return d.norm(dim=1)


@dataclass
class TrainConfig:
    epochs: int = 300
    batch: int = 2
    lr: float = 1e-3                   # smoke runs (validation cases): 1e-3 with aug x0.5 > 3e-4 with aug x1
    lr_min_frac: float = 0.05
    warmup_epochs: int = 2
    weight_decay: float = 1e-5
    w_pts: float = 1.0
    w_tail: float = 0.0                # + w_tail x mean of the worst tail_frac point errors of the batch (0: off,
    tail_frac: float = 0.1             # task N); round 3: the gate is on worst points, the mean alone misses them
    w_gate: float = 1.0                # point-loss weight of the gated points (gate_points; 1: all equal, round 3)
    w_diff: float = 0.02
    p_aug: float = 0.8
    aug: dict = field(default_factory=lambda: {"rot_deg": 2.0, "scale": 0.02, "shear": 0.01, "trans": 2.0})
    t1_jitter: float = 0.1             # T1 channel: random gain / offset / gamma of this size
    val_every: int = 5
    max_hours: float = 4.0             # total training wall clock over all resumes
    seed: int = 0


def lr_at(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / (cfg.warmup_epochs + 1)
    t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return cfg.lr * (cfg.lr_min_frac + (1 - cfg.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


_PW: dict = {}


def _point_weight(n: int, w_gate: float, dev) -> torch.Tensor:
    """Per point weight [n]: w_gate on the gated points (gate_points), 1 elsewhere."""
    key = (n, w_gate, str(dev))
    if key not in _PW:
        g = gate_points()
        gate = g["targets"] | g["rtms"] | g["fid"]
        if len(gate) != n:
            raise ValueError(f"{n} points, the gate masks have {len(gate)}")
        _PW[key] = torch.where(torch.as_tensor(gate, device=dev), float(w_gate), 1.0)
    return _PW[key]


def losses_for(model, batch: list[dict], spec: GridSpec, vg, tcfg: TrainConfig, steps: int, dev,
               rng: np.random.Generator | None = None) -> tuple[torch.Tensor, dict, torch.Tensor, list]:
    """Loss of a batch of cached samples (augmented when rng is given). Returns the loss, its parts,
    u [B, 3, *shape] and the per sample (x_in, r, r_pts) actually used."""
    xs, rs, rps = [], [], []
    for s in batch:
        x_in, r = s["x"].to(dev, non_blocking=True).float(), s["r"].to(dev, non_blocking=True).float()
        pts, r_pts = s["pts"].to(dev), s["r_pts"].to(dev)
        if rng is not None and rng.random() < tcfg.p_aug:
            x_in, r, r_pts = augment(x_in, r, r_pts, pts, random_affine(rng, **tcfg.aug), spec)
        if rng is not None and tcfg.t1_jitter > 0:
            j = tcfg.t1_jitter
            k = len(LABELS)
            t1 = x_in[k].clamp(0, 1.5)
            t1 = (t1 / 1.5) ** float(np.exp(rng.uniform(-j, j))) * 1.5
            x_in = x_in.clone()
            x_in[k] = t1 * float(1 + rng.uniform(-j, j)) + float(rng.uniform(-j, j) / 2)
        xs.append(x_in)
        rs.append(r)
        rps.append(r_pts)
    x_in = torch.stack(xs)
    r = torch.stack(rs)
    valid = torch.stack([s["valid"].to(dev) for s in batch])
    v = velocity(model, x_in, spec)
    u = vg.exp(v, steps)
    w = weights(x_in, valid)
    epe = (u - r).norm(dim=1)
    l_field = (epe * w).sum() / w.sum().clamp_min(1)
    pe = [point_errors(u[b], vg, s["pts"].to(dev), rps[b])[s["valid_pts"].to(dev)] for b, s in enumerate(batch)]
    pe = torch.cat(pe)
    if tcfg.w_gate == 1.0:   # round 3: every point equal
        l_pts = pe.mean()
    else:
        pw = torch.cat([_point_weight(len(s["pts"]), tcfg.w_gate, dev)[s["valid_pts"].to(dev)] for s in batch])
        l_pts = (pe * pw).sum() / pw.sum()
    l_diff = diffusion(v, spec.spacing)
    loss = l_field + tcfg.w_pts * l_pts + tcfg.w_diff * l_diff
    parts = {"field": l_field.item(), "pts": l_pts.item(), "diff": l_diff.item()}
    if tcfg.w_tail > 0:
        l_tail = pe.topk(max(1, math.ceil(tcfg.tail_frac * pe.numel()))).values.mean()
        loss = loss + tcfg.w_tail * l_tail
        parts["tail"] = l_tail.item()
    return loss, parts, u, list(zip(xs, rs, rps))


@torch.no_grad()
def validate(model, val: list[dict], spec: GridSpec, vg, steps: int, dev) -> dict:
    """Distances (subject mm) at the 10 targets and the 80 cap positions to charm's map, per case,
    and the field EPE (MNI mm) over valid head voxels. Score = mean over cases of
    (targets mean + cap mean) / 2 (lower is better)."""
    model.eval()
    rows = []
    gate = gate_points()
    for s in val:
        dev_s = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in s.items()}
        v = velocity(model, dev_s["x"][None].float(), spec)
        u = vg.exp(v, steps)
        L = dev_s["A"][:3, :3]
        e = point_errors(u[0], vg, dev_s["pts"], dev_s["r_pts"], L)
        ok = dev_s["valid_pts"]
        nt = s["n_targets"]
        et, ee = e[:nt][ok[:nt]], e[nt:][ok[nt:]]
        head = (dev_s["x"][:len(LABELS)].float().sum(0) > 0.5) & dev_s["valid"]
        epe = (u[0] - dev_s["r"].float()).norm(dim=0)[head]
        if len(e) == len(gate["rtms"]):
            g = {k: torch.as_tensor(m, device=dev) & ok for k, m in gate.items() if k != "targets"}
            er, ef = e[g["rtms"]], e[g["fid"]]
        else:   # not the 90 mni_points() (synthetic tests): no gated rows
            er = ef = torch.full((1,), float("nan"), device=dev)
        rows.append({"case": s["case"], "targets_mean": et.mean().item(), "targets_max": et.max().item(),
                     "eeg_mean": ee.mean().item(), "eeg_max": ee.max().item(), "epe_head": epe.mean().item(),
                     "rtms_mean": er.mean().item(), "rtms_max": er.max().item(),
                     "fid_mean": ef.mean().item(), "fid_max": ef.max().item(),
                     "pass_targets": bool(et.mean() <= 2 and et.max() <= 5),
                     "pass_eeg": bool(ee.mean() <= 2 and ee.max() <= 5)})
    model.train()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("targets_mean", "eeg_mean", "epe_head", "rtms_mean",
                                                             "fid_mean")}
    agg.update({"targets_max": max(r["targets_max"] for r in rows), "eeg_max": max(r["eeg_max"] for r in rows),
                "rtms_max": max(r["rtms_max"] for r in rows), "fid_max": max(r["fid_max"] for r in rows),
                "pass_targets": sum(r["pass_targets"] for r in rows), "pass_eeg": sum(r["pass_eeg"] for r in rows),
                "n": len(rows)})
    agg["score"] = (agg["targets_mean"] + agg["eeg_mean"]) / 2
    # R5's pilot score: the gated points only (targets, rTMS EEG set, fiducials; unprojected)
    agg["score_gate"] = (agg["targets_mean"] + agg["rtms_mean"] + agg["fid_mean"]) / 3
    return {"summary": agg, "cases": rows}


# ---------------------------------------------------------------------------- machine sharing

NNUNET_PATTERN = r"nnUNetv2_(train|plan_and_preprocess|predict)"
SLOT_FREE = re.compile(r"^[0-9-]+ [0-9:]+ slot free", re.M)


SHARE_MIN_FREE_GB = 6.0


def yield_reason(log: Path = ROUND1_LOG) -> str | None:
    """Why training must not use the GPU now (None: free). An nnU-Net job of either project has
    priority, and head-model's round 1 once its log says it has the slot. HM_WARPNET_IGNORE_ROUND1_LOG=1
    drops the log rule (for the orchestrator, after round 1 has finished). HM_WARPNET_SHARE_GPU=1 (one
    machine only, user 2026-10-28): train beside them and yield only when the GPU has less than
    SHARE_MIN_FREE_GB free."""
    if os.environ.get("HM_WARPNET_SHARE_GPU") == "1":
        if torch.cuda.is_available():
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            if free < SHARE_MIN_FREE_GB:
                return f"GPU shared and only {free:.1f} GB free"
        return None
    r = subprocess.run(["pgrep", "-f", NNUNET_PATTERN], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p.strip() and int(p) != os.getpid()]
    if pids:
        return f"nnU-Net process running (pids {' '.join(pids)})"
    if os.environ.get("HM_WARPNET_IGNORE_ROUND1_LOG") != "1" and Path(log).is_file():
        if SLOT_FREE.search(Path(log).read_text(errors="ignore")):
            return f"round 1 has the GPU slot ({log})"
    return None


# ---------------------------------------------------------------------------- training

def _atomic_save(obj, f: Path):
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, f)


def train(train_set: list[dict], val_set: list[dict], out_dir: Path = OUT, ncfg: NetConfig | None = None,
          tcfg: TrainConfig | None = None, dev="cuda", max_epochs_this_run: int | None = None,
          guard=yield_reason, log=print) -> dict:
    """Resumable training: <out_dir>/last.pt each epoch, best.pt on the best validation score.
    Stops (checkpoint first) when guard() gives a reason, after tcfg.epochs, or after tcfg.max_hours
    of training in total. Returns {"stopped": reason, "epoch": ..., "best": ...}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(dev)
    last = out_dir / "last.pt"
    ck = torch.load(last, map_location="cpu", weights_only=False) if last.is_file() else None
    if ck is not None:   # the checkpoint's configs win; only the budget (epochs, hours) may change
        saved = TrainConfig(**ck["train_config"])
        if tcfg is not None:
            saved.epochs, saved.max_hours = tcfg.epochs, tcfg.max_hours
        ncfg, tcfg = NetConfig(**ck["config"]), saved
    ncfg, tcfg = ncfg or NetConfig(), tcfg or TrainConfig()
    spec = ncfg.spec
    vg = spec.velocity_grid(dev)
    torch.manual_seed(tcfg.seed)
    model = build_model(ncfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    epoch, elapsed, best, history = 0, 0.0, float("inf"), []
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        epoch, elapsed, best, history = ck["epoch"], ck["elapsed_s"], ck["best_score"], ck["history"]
        log(f"resumed at epoch {epoch} ({elapsed / 3600:.2f} h trained, best score {best:.3f})")
    reason = guard()
    if reason:
        log(f"not starting: {reason}")
        return {"stopped": reason, "epoch": epoch, "best": best}
    rng = np.random.default_rng(tcfg.seed + epoch)

    def save(f, extra=None):
        _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "elapsed_s": elapsed,
                      "best_score": best, "history": history, "config": asdict(ncfg), "train_config": asdict(tcfg),
                      "train_cases": [s["case"] for s in train_set], "val_cases": [s["case"] for s in val_set],
                      **(extra or {})}, f)

    run_epochs, stopped = 0, None
    model.train()
    while True:
        if epoch >= tcfg.epochs:
            stopped = "epochs done"
            break
        if elapsed >= tcfg.max_hours * 3600:
            stopped = f"time cap {tcfg.max_hours} h"
            break
        if max_epochs_this_run is not None and run_epochs >= max_epochs_this_run:
            stopped = "max epochs for this run"
            break
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch, tcfg)
        order = rng.permutation(len(train_set))
        parts = []
        for i in range(0, len(order) - tcfg.batch + 1, tcfg.batch):
            batch = [train_set[j] for j in order[i:i + tcfg.batch]]
            opt.zero_grad(set_to_none=True)
            loss, part, _, _ = losses_for(model, batch, spec, vg, tcfg, ncfg.steps, dev, rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            parts.append({"loss": loss.item(), **part})
        epoch += 1
        run_epochs += 1
        rec = {"epoch": epoch, "lr": lr_at(epoch - 1, tcfg),
               **{k: float(np.mean([p[k] for p in parts])) for k in parts[0]}}
        val_res = None
        if val_set and (epoch % tcfg.val_every == 0 or epoch == tcfg.epochs):
            val_res = validate(model, val_set, spec, vg, ncfg.steps, dev)
            rec["val"] = val_res["summary"]
        elapsed += time.time() - t0
        rec["epoch_s"] = time.time() - t0
        rec["elapsed_h"] = elapsed / 3600
        rec["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30 if dev.type == "cuda" else 0.0
        history.append(rec)
        if val_res is not None and val_res["summary"]["score"] < best:
            best = val_res["summary"]["score"]
            save(out_dir / "best.pt", {"val": val_res})
            (out_dir / "best_val.json").write_text(json.dumps({"epoch": epoch, **val_res}, indent=1))
        save(last)
        msg = (f"epoch {epoch} loss {rec['loss']:.3f} field {rec['field']:.3f} pts {rec['pts']:.3f} "
               f"diff {rec['diff']:.4f} lr {rec['lr']:.2e} {rec['epoch_s']:.0f} s {rec['peak_gb']:.1f} GB")
        if val_res is not None:
            v = val_res["summary"]
            msg += (f" | val targets {v['targets_mean']:.2f}/{v['targets_max']:.1f} EEG {v['eeg_mean']:.2f}/"
                    f"{v['eeg_max']:.1f} pass {v['pass_targets']}/{v['pass_eeg']} of {v['n']} epe {v['epe_head']:.2f}"
                    f" score {v['score']:.3f}{' *' if v['score'] <= best else ''}")
        log(msg)
        with open(out_dir / "history.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        reason = guard()
        if reason:
            stopped = f"yielded: {reason}"
            break
    log(f"stopped: {stopped} (epoch {epoch}, {elapsed / 3600:.2f} h, best score {best:.3f})")
    return {"stopped": stopped, "epoch": epoch, "best": best}
