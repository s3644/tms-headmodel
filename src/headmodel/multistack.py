"""Thick-slice MRI stacks (e.g. a 5 mm axial and a 5 mm sagittal T1 of the same visit) -> one ~1 mm volume.

Each stack is sharp in its own slice plane and coarse across its slices; orthogonal stacks are coarse in
different directions, so together they sample every direction finely (super-resolution reconstruction,
Plenge 2012, PMID 22298247; SynthSR does it with a CNN, PMID 34048902). On the GPU:
1. rigid registration of every stack to the first (the patient moves between series);
2. x = argmin sum_k |F_k x - y_k|^2 + lam TV(x), where F_k is stack k's acquisition: resample x along the
   stack's slices and average over the slice thickness (a box profile; slice gaps simply leave no data).

Voxel i covers [i - 0.5, i + 0.5] around its centre (nibabel convention). Affines map voxel index -> mm.
navprep's `combine` stage uses combine() for several MRI series of one visit; the result is provisional until
the multi-series arms of the R5 scoring rule are scored (PLAN 10.5).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn.functional import grid_sample

N_SUB = 8   # samples across one slice's thickness


@dataclass
class Stack:
    data: torch.Tensor        # [X, Y, Z] float, on the device
    affine: torch.Tensor      # [4, 4] voxel -> mm (float64)
    axis: int                 # the slice axis of `data` (0-2)
    thick: float              # slice thickness, mm


def rigid(params: torch.Tensor) -> torch.Tensor:
    """6 params (rotation vector rad, translation mm) -> [4, 4], rotation about the origin."""
    r, t = params[:3], params[3:]
    th = torch.sqrt((r * r).sum() + 1e-12)
    k = r / th
    K = torch.zeros(3, 3, dtype=params.dtype, device=params.device)
    K[0, 1], K[0, 2], K[1, 2] = -k[2], k[1], -k[0]
    K = K - K.T
    R = torch.eye(3, dtype=params.dtype, device=params.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K
    top = torch.cat([R, t[:, None]], 1)
    return torch.cat([top, torch.tensor([[0, 0, 0, 1]], dtype=params.dtype, device=params.device)], 0)


def _sample(vol: torch.Tensor, vol_affine: torch.Tensor, pts_mm: torch.Tensor) -> torch.Tensor:
    """Trilinear values of vol [X, Y, Z] at mm points [..., 3]; zero outside."""
    inv = torch.linalg.inv(vol_affine.double()).to(pts_mm.device, pts_mm.dtype)
    ijk = pts_mm @ inv[:3, :3].T + inv[:3, 3]
    n = torch.tensor(vol.shape, dtype=ijk.dtype, device=ijk.device)
    g = (2 * ijk / (n - 1) - 1).flip(-1)                       # grid_sample wants (z, y, x) order
    out = grid_sample(vol[None, None], g.reshape(1, 1, 1, -1, 3).to(vol.dtype), mode="bilinear",
                      padding_mode="zeros", align_corners=True)
    return out.reshape(pts_mm.shape[:-1])


def slab_points(stack_shape, stack_affine: torch.Tensor, axis: int, thick: float, spacing: float,
                motion: torch.Tensor | None = None, device="cuda") -> torch.Tensor:
    """mm points [X, Y, Z, N_SUB, 3]: N_SUB positions across each voxel's slice thickness, mapped by the
    stack's affine and then by `motion` (head pose at this stack's acquisition, relative to x's frame)."""
    idx = [torch.arange(s, device=device, dtype=torch.float64) for s in stack_shape]
    I, J, K = torch.meshgrid(*idx, indexing="ij")
    base = torch.stack([I, J, K], -1)[..., None, :].expand(*stack_shape, N_SUB, 3).clone()
    off = (torch.arange(N_SUB, device=device, dtype=torch.float64) + 0.5) / N_SUB - 0.5   # [-0.5, 0.5)
    base[..., axis] = base[..., axis] + off * (thick / spacing)
    A = stack_affine.to(device)
    if motion is not None:
        A = motion.to(device) @ A
    return base @ A[:3, :3].T + A[:3, 3]


def acquire(vol: torch.Tensor, vol_affine: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """The forward model: the slab mean of vol at slab_points -> the stack's voxel values."""
    return _sample(vol, vol_affine, pts).mean(-1)


def tv(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    d = [torch.diff(x, dim=i) for i in range(3)]
    return torch.sqrt(d[0][:, :-1, :-1] ** 2 + d[1][:-1, :, :-1] ** 2 + d[2][:-1, :-1, :] ** 2 + eps ** 2).mean()


def interp_to(stack: Stack, grid_shape, grid_affine: torch.Tensor, motion=None) -> torch.Tensor:
    """Plain trilinear resampling of a stack onto the grid (the no-reconstruction baseline)."""
    idx = [torch.arange(s, device=stack.data.device, dtype=torch.float64) for s in grid_shape]
    pts = torch.stack(torch.meshgrid(*idx, indexing="ij"), -1) @ grid_affine[:3, :3].T.to(stack.data.device) \
        + grid_affine[:3, 3].to(stack.data.device)
    A = stack.affine if motion is None else motion.to(stack.affine.device) @ stack.affine
    return _sample(stack.data, A, pts)


def register_stacks(stacks: list[Stack], grid_shape, grid_affine, iters: int = 150, lr: float = 0.01,
                    ref: torch.Tensor | None = None, init=None):
    """Rigid pose of each stack k > 0 relative to stack 0, by fitting stack k's forward model to `ref` (default:
    stack 0's plain interpolation, which is smooth across stack 0's slices; pass a reconstruction to refine, see
    register_and_reconstruct). Rotation is about the grid centre; the step size decays to zero (cosine).
    Returns [motion_k] (4x4, identity for stack 0)."""
    dev = stacks[0].data.device
    if ref is None:
        ref = interp_to(stacks[0], grid_shape, grid_affine).float()
    c = grid_affine[:3, :3].to(dev) @ (torch.tensor(grid_shape, device=dev, dtype=torch.float64) - 1) / 2 \
        + grid_affine[:3, 3].to(dev)
    C, Ci = torch.eye(4, dtype=torch.float64, device=dev), torch.eye(4, dtype=torch.float64, device=dev)
    C[:3, 3], Ci[:3, 3] = c, -c
    scale = torch.tensor([1, 1, 1, 100, 100, 100], dtype=torch.float64, device=dev)   # rad vs mm
    out = [torch.eye(4, dtype=torch.float64, device=dev)]
    for k, s in enumerate(stacks[1:], 1):
        M0 = torch.eye(4, dtype=torch.float64, device=dev) if init is None else init[k].to(dev)
        p = torch.zeros(6, dtype=torch.float64, device=dev, requires_grad=True)
        opt = torch.optim.Adam([p], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        sp = _spacing(s)
        for it in range(iters):
            M = C @ rigid(p * scale) @ Ci @ M0
            pts = slab_points(s.data.shape, s.affine, s.axis, s.thick, sp, M, dev)
            pred = acquire(ref, grid_affine.to(dev), pts.float())
            m = (pred > 0) & (s.data > 0)
            loss = -_ncc(pred[m], s.data[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        out.append(C @ rigid(p.detach() * scale) @ Ci @ M0)
    return out


def register_and_reconstruct(stacks: list[Stack], grid_shape, grid_affine, rounds: int = 2, **kw):
    """Alternate (as slice-to-volume reconstruction does): poses -> volume -> poses against the volume -> volume."""
    mot = register_stacks(stacks, grid_shape, grid_affine)
    x = reconstruct(stacks, mot, grid_shape, grid_affine, **kw)
    for _ in range(rounds - 1):
        mot = register_stacks(stacks, grid_shape, grid_affine, ref=x, init=mot)
        x = reconstruct(stacks, mot, grid_shape, grid_affine, **kw)
    return x, mot


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


def _spacing(s: Stack) -> float:
    return float(torch.linalg.norm(s.affine[:3, s.axis]))


def reconstruct(stacks: list[Stack], motions, grid_shape, grid_affine, lam: float = 0.02,
                iters: int = 200, lr: float = 0.02) -> torch.Tensor:
    """Super-resolution volume on the grid from the registered stacks (data term + TV)."""
    dev = stacks[0].data.device
    ga = grid_affine.to(dev)
    init = [interp_to(s, grid_shape, ga, m).float() for s, m in zip(stacks, motions)]
    cover = [(i > 0).float() for i in init]
    x = (sum(init) / torch.clamp(sum(cover), min=1)).clone().requires_grad_(True)
    pts = [slab_points(s.data.shape, s.affine, s.axis, s.thick, _spacing(s), m, dev).float()
           for s, m in zip(stacks, motions)]
    opt = torch.optim.Adam([x], lr=lr)
    for it in range(iters):
        loss = sum(((acquire(x, ga, p) - s.data) ** 2).mean() for s, p in zip(stacks, pts)) + lam * tv(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return x.detach()


def simulate(vol: torch.Tensor, vol_affine: torch.Tensor, axis: int, thick: float, gap: float,
             inplane: float = 1.0, motion=None, noise: float = 0.0, seed: int = 0) -> Stack:
    """A thick-slice stack of vol: slices `thick` mm, `gap` mm apart, along world axis `axis` of vol's grid,
    in-plane `inplane` mm, covering vol's field of view. `motion` = the head's pose during this series."""
    dev = vol.device
    zooms = torch.linalg.norm(vol_affine[:3, :3], dim=0)
    fov = torch.tensor(vol.shape, dtype=torch.float64) * zooms.cpu()
    sp = torch.full((3,), inplane, dtype=torch.float64)
    sp[axis] = thick + gap
    shape = tuple(int(n) for n in torch.floor(fov / sp))
    A = vol_affine.clone()
    A[:3, :3] = vol_affine[:3, :3] / zooms * sp.to(vol_affine)
    corner = vol_affine[:3, 3] - vol_affine[:3, :3] @ torch.full((3,), 0.5, dtype=torch.float64)
    A[:3, 3] = corner + A[:3, :3] @ torch.full((3,), 0.5, dtype=torch.float64)
    pts = slab_points(shape, A, axis, thick, float(sp[axis]), motion, dev).float()
    y = acquire(vol, vol_affine.to(dev), pts)
    if noise:
        g = torch.Generator(device=dev).manual_seed(seed)
        y = torch.sqrt((y + noise * torch.randn(y.shape, generator=g, device=dev)) ** 2
                       + (noise * torch.randn(y.shape, generator=g, device=dev)) ** 2)   # Rician
    return Stack(y, A, axis, thick)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# ------------------------------------------------------------------------------------------ files

MIN_NCC = 0.5   # below this after registration, the series are not of the same head and visit


def slice_thickness(path: Path, spacing: float) -> float:
    """dcm2niix's sidecar (<name>.json, SliceThickness) when present; otherwise the slice spacing (no gap)."""
    side = Path(str(path).removesuffix(".gz").removesuffix(".nii") + ".json")
    if side.is_file():
        t = json.loads(side.read_text()).get("SliceThickness")
        if t:
            return min(float(t), spacing)
    return spacing


def load_series(path: Path, device: str = "cuda") -> tuple[Stack, float]:
    """A NIfTI series -> (Stack scaled to its foreground 99th percentile, that scale). Slice axis = the
    coarsest voxel axis."""
    img = nib.load(str(path))
    v = np.asarray(img.dataobj, dtype=np.float32)
    if v.ndim == 4:
        v = v[..., 0]
    if v.ndim != 3:
        raise ValueError(f"{path.name}: not a 3-D image")
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    fg = v[v > np.percentile(v, 50)]
    scale = float(np.percentile(fg, 99)) if fg.size else 1.0
    axis = int(np.argmax(zooms))
    stack = Stack(torch.from_numpy(v / max(scale, 1e-6)).to(device), torch.from_numpy(img.affine).double(),
                  axis, slice_thickness(path, float(zooms[axis])))
    return stack, scale


def union_grid(stacks: list[Stack], mm: float = 1.0) -> tuple[tuple[int, int, int], torch.Tensor]:
    """An LPS-aligned grid of mm voxels covering every series' field of view (world corners)."""
    pts = []
    for s in stacks:
        n = torch.tensor(s.data.shape, dtype=torch.float64)
        c = torch.stack(torch.meshgrid(*[torch.tensor([-0.5, float(k) - 0.5], dtype=torch.float64) for k in n],
                                       indexing="ij"), -1).reshape(-1, 3)
        pts.append(c @ s.affine[:3, :3].T + s.affine[:3, 3])
    p = torch.cat(pts)
    lps = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))   # world RAS -> LPS axes
    q = p @ lps.T
    lo, hi = q.min(0).values, q.max(0).values
    shape = tuple(int(k) for k in torch.ceil((hi - lo) / mm))
    A = torch.eye(4, dtype=torch.float64)
    A[:3, :3] = lps * mm
    A[:3, 3] = lps @ (lo + mm / 2)
    return shape, A


def combine(paths: list[Path], out: Path, device: str = "cuda") -> dict:
    """Several thick series of one visit -> out (1 mm, LPS, the reference series' intensity scale) + a summary.
    The reference is the series with the thinnest slices (ties: the larger field of view, then the first given)."""
    loaded = [load_series(Path(p), device) for p in paths]

    def fov(s: Stack) -> float:
        return float(torch.linalg.norm(s.affine[:3, :3], dim=0).prod()) * s.data.numel()

    ref = min(range(len(loaded)), key=lambda i: (round(_spacing(loaded[i][0]), 2), -fov(loaded[i][0]), i))
    order = [ref] + [i for i in range(len(loaded)) if i != ref]
    stacks = [loaded[i][0] for i in order]
    shape, A = union_grid(stacks)
    x, mot = register_and_reconstruct(stacks, shape, A)
    ncc = []   # each series against the reference series alone (the reconstruction would absorb any series)
    ref_only = interp_to(stacks[0], shape, A).float()
    for s, m in zip(stacks[1:], mot[1:]):
        pts = slab_points(s.data.shape, s.affine, s.axis, s.thick, _spacing(s), m, device).float()
        pred = acquire(ref_only, A.to(device), pts)
        keep = (pred > 0) & (s.data > 0)
        ncc.append(round(float(_ncc(pred[keep], s.data[keep])), 3))
    if ncc and min(ncc) < MIN_NCC:
        raise ValueError(f"the series do not match after registration (NCC {ncc}): same patient and visit?")
    vol = (x.clamp(min=0) * loaded[ref][1]).cpu().numpy().astype(np.float32)
    nib.save(nib.Nifti1Image(vol, A.numpy()), str(out))
    series = []
    for k, i in enumerate(order):
        s = loaded[i][0]
        M = mot[k].cpu()
        series.append({"input": Path(paths[i]).name, "shape": list(s.data.shape),
                       "slice_axis": s.axis, "spacing_mm": round(_spacing(s), 3), "thickness_mm": round(s.thick, 3),
                       "motion_deg": round(float(torch.rad2deg(torch.arccos(((M[:3, :3].trace() - 1) / 2).clamp(-1, 1)))), 2),
                       "motion_mm": round(float(M[:3, 3].norm()), 2)})
    return {"reference": Path(paths[ref]).name, "series": series, "ncc": ncc, "grid": list(shape)}
