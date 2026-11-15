"""Scan-rescan floor of charm (TASK F): rigid alignment of two T1s of one head and the WP3 point metrics.

Own code (PyTorch + numpy/scipy). The rigid fit maximises the global normalised cross-correlation (NCC)
of the two T1s inside a brain mask of the fixed image, coarse to fine (Gaussian smoothing per level),
6 dof (rotation vector about the mask centroid + translation), L-BFGS. Conventions:

- M_ab (4x4) maps fixed (A) world mm to moving (B) world mm: the pull-back used to resample B on A.
- Points delivered in B's space go to A's space with inv(M_ab).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
import torch

from .register import Grid

GATE_MEAN, GATE_MAX = 2.0, 5.0
LEVELS = ((4.0, 4), (2.0, 2), (1.0, 1), (0.0, 1))   # (Gaussian sigma mm, point stride in voxels)


def rotvec_matrix(r: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, differentiable at 0."""
    th2 = (r * r).sum()
    K = torch.zeros(3, 3, dtype=r.dtype, device=r.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -r[2], r[1], r[2], -r[0], -r[1], r[0]
    if float(th2.detach()) < 1e-12:
        a, b = 1 - th2 / 6, 0.5 - th2 / 24
    else:
        th = torch.sqrt(th2)
        a, b = torch.sin(th) / th, (1 - torch.cos(th)) / th2
    return torch.eye(3, dtype=r.dtype, device=r.device) + a * K + b * (K @ K)


def rigid(rotvec, t, centre=(0.0, 0.0, 0.0)) -> np.ndarray:
    """4x4: x -> R (x - c) + c + t."""
    R = rotvec_matrix(torch.as_tensor(np.asarray(rotvec, float), dtype=torch.float64)).numpy()
    c = np.asarray(centre, float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c - R @ c + np.asarray(t, float)
    return M


def apply(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ M[:3, :3].T + M[:3, 3]


def polar_rotation(L: np.ndarray) -> np.ndarray:
    """Nearest rotation to a 3x3 matrix (polar decomposition; det +1)."""
    U, _, Vt = np.linalg.svd(L)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def init_from_affines(W_a: np.ndarray, W_b: np.ndarray, c_a: np.ndarray, c_b: np.ndarray) -> np.ndarray:
    """Start for M_ab from two MNI -> subject affines (charm's worldToWorldTransformMatrix): the rotation
    of W_b inv(W_a), with the translation that maps A's brain centroid c_a onto B's c_b."""
    R = polar_rotation((W_b @ np.linalg.inv(W_a))[:3, :3])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(c_b, float) - R @ np.asarray(c_a, float)
    return M


def labels_on_grid(lab: np.ndarray, lab_aff: np.ndarray, shape, aff: np.ndarray) -> np.ndarray:
    """Nearest-neighbour labels at the voxel centres of another grid (world-based; any grids)."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape[:3]], indexing="ij"), -1).reshape(-1, 3)
    vox = np.rint(apply(np.linalg.inv(lab_aff) @ aff, ijk)).astype(np.int64)
    ok = ((vox >= 0) & (vox < np.array(lab.shape[:3]))).all(1)
    out = np.zeros(len(ijk), lab.dtype)
    out[ok] = lab[tuple(vox[ok].T)]
    return out.reshape(shape[:3])


def ncc_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a - a.mean(), b - b.mean()
    return (a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum() + 1e-12)


def register_rigid(fixed: np.ndarray, fixed_aff: np.ndarray, moving: np.ndarray, moving_aff: np.ndarray,
                   mask: np.ndarray, init: np.ndarray | None = None, levels=LEVELS, device: str = "cpu",
                   iters: int = 40) -> dict:
    """6-dof M_ab (fixed world -> moving world) maximising the NCC of the two images over the fixed
    mask voxels. Returns {"M", "ncc_init", "ncc", "centre"}."""
    fixed, moving = np.asarray(fixed, np.float32), np.asarray(moving, np.float32)
    M0 = np.eye(4) if init is None else np.asarray(init, float)
    zf = np.sqrt((fixed_aff[:3, :3] ** 2).sum(0))
    zm = np.sqrt((moving_aff[:3, :3] ** 2).sum(0))
    ijk_all = np.argwhere(mask)
    if len(ijk_all) < 100:
        raise ValueError("mask has < 100 voxels")
    c = apply(fixed_aff, ijk_all).mean(0)
    p = torch.zeros(6, dtype=torch.float64, device=device, requires_grad=True)
    M0_t = torch.as_tensor(M0, dtype=torch.float64, device=device)
    c_t = torch.as_tensor(c, dtype=torch.float64, device=device)

    def transform(x):   # fixed world -> moving world, x [N, 3] float64
        R = rotvec_matrix(p[:3])
        y = (x - c_t) @ R.T + c_t + p[3:] * 10   # translation in cm: rotations and shifts of similar size
        return y @ M0_t[:3, :3].T + M0_t[:3, 3]

    ncc_init = ncc_final = None
    for sigma, stride in levels:
        f_s = ndi.gaussian_filter(fixed, sigma / zf) if sigma > 0 else fixed
        m_s = ndi.gaussian_filter(moving, sigma / zm) if sigma > 0 else moving
        sel = ijk_all[(ijk_all % stride == 0).all(1)] if stride > 1 else ijk_all
        x = torch.as_tensor(apply(fixed_aff, sel), dtype=torch.float64, device=device)
        fv = torch.as_tensor(f_s[tuple(sel.T)], dtype=torch.float32, device=device)
        g = Grid(torch.as_tensor(m_s, device=device)[None], moving_aff)
        opt = torch.optim.LBFGS([p], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            v = g.sample(transform(x).float())[0]
            loss = 1 - ncc_t(fv, v)
            loss.backward()
            return loss

        with torch.no_grad():
            if ncc_init is None:
                ncc_init = float(ncc_t(fv, g.sample(transform(x).float())[0]))
        opt.step(closure)
        with torch.no_grad():
            ncc_final = float(ncc_t(fv, g.sample(transform(x).float())[0]))
    pp = p.detach().cpu().numpy()
    M = M0 @ rigid(pp[:3], pp[3:] * 10, c)
    return {"M": M, "ncc_init": ncc_init, "ncc": ncc_final, "centre": c.tolist()}


def pair_errors(pts_a: dict, pts_b: dict, M_ab: np.ndarray) -> dict:
    """Per group: distances (mm) between A's points and B's points mapped into A's space (inv(M_ab))."""
    Minv = np.linalg.inv(M_ab)
    out = {}
    for g, pa in pts_a.items():
        pa, pb = np.asarray(pa, float), np.asarray(pts_b[g], float)
        if pa.shape != pb.shape:
            raise ValueError(f"{g}: {pa.shape} vs {pb.shape}")
        out[g] = np.linalg.norm(apply(Minv, pb) - pa, axis=1)
    return out


def summarise(pairs: list[dict], groups=None, gate_mean: float = GATE_MEAN, gate_max: float = GATE_MAX) -> dict:
    """pairs: [{"errors": {group: per point mm}}]. Per group: mean of pair means, SD, worst pair mean,
    worst point, p95 of all points, points over gate_max, pairs passing (mean <= gate_mean and max <= gate_max)."""
    if not pairs:
        return {"n_pairs": 0}
    groups = groups or list(pairs[0]["errors"])
    s = {"n_pairs": len(pairs)}
    for g in groups:
        per = [np.asarray(p["errors"][g], float) for p in pairs]
        means = np.array([e.mean() for e in per])
        maxs = np.array([e.max() for e in per])
        pts = np.concatenate(per)
        s[g] = {"mean": float(means.mean()), "sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "max_pair_mean": float(means.max()), "max": float(pts.max()), "p95": float(np.percentile(pts, 95)),
                "points_over_max": int((pts > gate_max).sum()), "n_points": int(len(pts)),
                "pairs_pass": int(((means <= gate_mean) & (maxs <= gate_max)).sum())}
    return s


def labels_majority(lab: np.ndarray, lab_aff: np.ndarray, shape, aff: np.ndarray, slab: int = 16) -> np.ndarray:
    """Labels on a coarser grid by majority vote of the fine voxels whose centres fall in each coarse
    voxel (any grids, e.g. charm's 0.5 mm labels on a 1 x 1 x 1.25 mm T1); ties go to the lower label;
    coarse voxels without any fine voxel get the nearest fine label."""
    lab = np.asarray(lab)
    n_lab = int(lab.max()) + 1
    N = int(np.prod(shape[:3]))
    counts = np.zeros((N, n_lab), np.uint16)
    M = np.linalg.inv(aff) @ lab_aff
    ii, jj = np.meshgrid(np.arange(lab.shape[0]), np.arange(lab.shape[1]), indexing="ij")
    for k0 in range(0, lab.shape[2], slab):
        k1 = min(lab.shape[2], k0 + slab)
        ijk = np.stack([np.repeat(ii[..., None], k1 - k0, -1), np.repeat(jj[..., None], k1 - k0, -1),
                        np.broadcast_to(np.arange(k0, k1), ii.shape + (k1 - k0,))], -1).reshape(-1, 3)
        v = np.rint(apply(M, ijk)).astype(np.int64)
        ok = ((v >= 0) & (v < np.array(shape[:3]))).all(1)
        flat = np.ravel_multi_index(tuple(v[ok].T), shape[:3])
        lv = lab[:, :, k0:k1].reshape(-1)[ok].astype(np.int64)
        np.add.at(counts, (flat, lv), 1)
    out = counts.argmax(1).astype(lab.dtype)
    empty = counts.sum(1) == 0
    if empty.any():
        out[empty] = labels_on_grid(lab, lab_aff, shape, aff).reshape(-1)[empty]
    return out.reshape(shape[:3])
