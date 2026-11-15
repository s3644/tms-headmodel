"""R5 thick-slice augmentation: slab means along one axis (with or without a gap), same size, other axes untouched."""
from __future__ import annotations

import pytest
import torch

from headmodel.nnunet_r5 import thick_slices


def test_thick_slices_averages_one_axis_only():
    img = torch.zeros(1, 30, 20, 10)
    img[:, 15] = 1.0                                   # a one-voxel sheet across axis 0
    out = thick_slices(img, axis=0, k=5)
    assert out.shape == img.shape
    line = out[0, :, 0, 0]
    assert line.max() < 0.5 and float(line.sum()) == pytest.approx(1.0, abs=0.2)   # spread, mass kept
    assert torch.allclose(thick_slices(img, axis=1, k=5), img)   # constant along axis 1: unchanged
    assert torch.equal(thick_slices(img, axis=0, k=1), img)


def test_k1_no_gap_is_identity_and_ramps_survive():
    img = torch.randn(2, 12, 9, 7)
    assert torch.equal(thick_slices(img, axis=2, k=1.0), img)
    ramp = torch.arange(40.0).view(1, 40, 1, 1).expand(1, 40, 3, 3)
    out = thick_slices(ramp, axis=0, k=5, gap=1.5)   # a linear ramp stays linear away from the ends
    assert torch.allclose(out[0, 5:-8, 0, 0], ramp[0, 5:-8, 0, 0], atol=1e-4)


def test_a_sheet_in_the_gap_is_lost():
    img = torch.zeros(1, 40, 4, 4)
    img[:, 5] = 1.0                                    # slabs [0,5) and [6.5,11.5): voxel 5 lies in the gap
    assert float(thick_slices(img, axis=0, k=5, gap=1.5).abs().max()) < 1e-6
    img2 = torch.zeros(1, 40, 4, 4)
    img2[:, 2] = 1.0                                   # inside the first slab: kept, spread
    assert float(thick_slices(img2, axis=0, k=5, gap=1.5).max()) > 0.1


def test_trainer_inserts_the_transform():
    nn = pytest.importorskip("nnunetv2")   # nnunet env only
    from headmodel.nnunet_r5 import ThickSliceTransform, nnUNetTrainerHMThick

    tr = nnUNetTrainerHMThick.get_training_transforms((64, 64, 64), (0, 0), None, (0, 1, 2), False)
    names = [type(getattr(t, "transform", t)).__name__ for t in tr.transforms]
    i = names.index("ThickSliceTransform")
    assert names[i - 1] == "GammaTransform" and nn is not None
    out = tr(**{"image": torch.randn(1, 64, 64, 64), "segmentation": torch.zeros(1, 64, 64, 64)})
    assert out["image"].shape == (1, 64, 64, 64)
    p = ThickSliceTransform().get_parameters(image=torch.zeros(1, 8, 8, 8))
    assert set(p) == {"axis", "k", "gap"} and 0 <= p["gap"] <= 0.3 * p["k"]
