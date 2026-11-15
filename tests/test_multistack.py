"""Thick-slice stacks -> one volume: two orthogonal stacks beat one, and the rigid pose between them is found."""
from __future__ import annotations

import pytest
import torch

from headmodel import multistack as ms

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")


def _vol(n=48):
    """Smooth blobs 3-7 mm wide (structures, not texture: a 5 mm slab must still see where they are)."""
    g = torch.Generator().manual_seed(0)
    ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32)] * 3, indexing="ij"), -1)
    v = torch.zeros(n, n, n)
    for _ in range(40):
        c = torch.rand(3, generator=g) * (n - 12) + 6
        s = 3 + 4 * float(torch.rand(1, generator=g))
        v += float(torch.rand(1, generator=g) + 0.5) * torch.exp(-((ijk - c) ** 2).sum(-1) / (2 * s * s))
    return (v / v.max()).cuda(), torch.eye(4, dtype=torch.float64)


@cuda
def test_constant_volume_is_acquired_unchanged():
    v = torch.ones(32, 32, 32, device="cuda")
    s = ms.simulate(v, torch.eye(4, dtype=torch.float64), 2, 4.0, 1.0)
    inner = s.data[2:-2, 2:-2, 1:-1]
    assert torch.allclose(inner, torch.ones_like(inner), atol=1e-5)


@cuda
def test_two_orthogonal_stacks_beat_one_and_pose_is_found():
    v, a = _vol()
    c = torch.eye(4, dtype=torch.float64)
    c[:3, 3] = 23.5
    ci = torch.linalg.inv(c)
    motion = c @ ms.rigid(torch.tensor([0.02, 0.0, -0.02, 1.0, -0.5, 0.5], dtype=torch.float64)) @ ci
    s1 = ms.simulate(v, a, 2, 4.0, 1.0)
    s2 = ms.simulate(v, a, 0, 4.0, 1.0, motion=motion)
    two, mot = ms.register_and_reconstruct([s1, s2], v.shape, a)
    pts = torch.tensor([[0.0, 0, 0], [47, 47, 47], [0, 47, 23]], dtype=torch.float64)
    est = pts @ mot[1].cpu()[:3, :3].T + mot[1].cpu()[:3, 3]
    true = pts @ motion[:3, :3].T + motion[:3, 3]
    assert float((est - true).norm(dim=1).max()) < 0.5
    one = ms.reconstruct([s1], mot[:1], v.shape, a)
    inner = (slice(4, -4),) * 3
    err = lambda x: float(((x - v)[inner] ** 2).mean().sqrt())   # noqa: E731
    assert err(two) < 0.8 * err(one)


@cuda
def test_combine_files_registers_reconstructs_and_rejects_a_stranger(tmp_path):
    import json

    import nibabel as nib

    v, a = _vol()
    world = torch.tensor([[1.0, 0, 0, -24], [0, 1, 0, -30], [0, 0, 1, -20], [0, 0, 0, 1]], dtype=torch.float64)
    a = world   # a RAS volume placed in the world
    c = torch.eye(4, dtype=torch.float64)
    c[:3, 3] = world[:3, :3] @ torch.full((3,), 23.5, dtype=torch.float64) + world[:3, 3]
    motion = c @ ms.rigid(torch.tensor([0.0, 0.02, 0.0, 0.5, 1.0, 0.0], dtype=torch.float64)) @ torch.linalg.inv(c)
    paths = []
    for name, axis, mot in (("ax", 2, None), ("sag", 0, motion)):
        s = ms.simulate(v, a, axis, 4.0, 1.0, motion=mot)
        f = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(s.data.cpu().numpy(), s.affine.numpy()), str(f))
        (tmp_path / f"{name}.json").write_text(json.dumps({"SliceThickness": 4.0}))
        paths.append(f)
    out = tmp_path / "combined.nii.gz"
    info = ms.combine(paths, out)
    img = nib.load(str(out))
    assert nib.aff2axcodes(img.affine) == ("L", "P", "S") and min(info["ncc"]) > 0.9
    assert {x["thickness_mm"] for x in info["series"]} == {4.0} and info["series"][0]["spacing_mm"] == 5.0
    assert info["series"][1]["motion_mm"] > 0.3
    # a different head as the second series: refused
    g = torch.Generator().manual_seed(5)
    other = ms.simulate(torch.rand(48, 48, 48, generator=g).cuda(), a, 0, 4.0, 1.0)
    nib.save(nib.Nifti1Image(other.data.cpu().numpy(), other.affine.numpy()), str(tmp_path / "other.nii.gz"))
    with pytest.raises(ValueError, match="do not match"):
        ms.combine([paths[0], tmp_path / "other.nii.gz"], tmp_path / "bad.nii.gz")
