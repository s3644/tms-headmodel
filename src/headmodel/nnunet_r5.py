"""R5 segmenter (PLAN.md 10.8): nnU-Net's default trainer, 250 epochs, plus thick-slice augmentation.

The clinic's old T1s (stroke patients especially) are 5 mm thick, often with a gap between slices (PLAN 10.5).
ThickSliceTransform makes a training patch look like a thick-slice acquisition along one random axis:
- slabs k voxels thick (a rectangular slice profile, k ~ U[2, 6] voxels of the ~1 mm grid), one every k + gap
  voxels; half the draws have no gap, the other half a gap ~ U[0, 0.3] k (amendment r5_train_plan_amend1);
- linear interpolation between the slab centres back to the patch size.
The labels stay at full resolution, so the network learns 1 mm labels from thick input. It is applied after
nnU-Net's own intensity and low-resolution augmentations, with probability P_THICK.

nnU-Net finds trainers by class name inside its own package. `envs/README.md` describes the one-line shim placed in
nnunetv2/training/nnUNetTrainer/variants/ that imports this class (it needs head-model's src on PYTHONPATH, which
navprep and the training scripts set).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.nn.functional import interpolate

P_THICK = 0.3
K_RANGE = (2.0, 6.0)
GAP_MAX = 0.3   # gap as a fraction of the slice thickness


def thick_slices(img: torch.Tensor, axis: int, k: float, gap: float = 0.0) -> torch.Tensor:
    """img [C, X, Y, Z] -> the same size, as if acquired with slices k voxels thick, `gap` voxels apart, along
    `axis` (0-2). Voxel i covers [i, i+1); the first slab starts at 0."""
    n = img.shape[1 + axis]
    s = k + gap
    if s >= n or (k <= 1 and gap == 0):
        return img
    x = img.float().movedim(1 + axis, -1)
    flat = x.reshape(-1, n)
    cum = torch.nn.functional.pad(flat.cumsum(-1), (1, 0))          # integral at the voxel edges 0..n

    def integral(p: torch.Tensor) -> torch.Tensor:                  # piecewise-linear between the edges
        p = p.clamp(0, n)
        i = p.floor().long().clamp(max=n - 1)
        return cum[:, i] + (p - i) * (cum[:, i + 1] - cum[:, i])

    m = int((n - k) // s) + 1
    centre = k / 2 + s * torch.arange(m, dtype=torch.float64)
    lo, hi = (centre - k / 2).float(), (centre + k / 2).clamp(max=n).float()
    slab = (integral(hi) - integral(lo)) / (hi - lo)                # [rows, m] slab means
    f = ((torch.arange(n) + 0.5 - k / 2) / s).clamp(0, m - 1)       # voxel centres in slab units
    j = f.floor().long().clamp(max=max(m - 2, 0))
    w = (f - j).float()
    out = slab[:, j] * (1 - w) + slab[:, (j + 1).clamp(max=m - 1)] * w
    return out.reshape(x.shape).movedim(-1, 1 + axis).to(img.dtype)


try:   # the transform and trainer need the nnunet env; thick_slices above does not
    from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
    from batchgeneratorsv2.transforms.utils.random import RandomTransform
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
except ImportError:   # pragma: no cover - other envs import thick_slices only
    ImageOnlyTransform = nnUNetTrainer = RandomTransform = None


if ImageOnlyTransform is not None:
    class ThickSliceTransform(ImageOnlyTransform):
        def __init__(self, k_range=K_RANGE):
            super().__init__()
            self.k_range = k_range

        def get_parameters(self, **data_dict) -> dict:
            img = data_dict["image"]
            k = float(np.random.uniform(*self.k_range))
            gap = 0.0 if np.random.rand() < 0.5 else float(np.random.uniform(0, GAP_MAX)) * k
            return {"axis": int(np.random.randint(img.ndim - 1)), "k": k, "gap": gap}

        def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
            if img.ndim != 4:   # 2-D configurations: left alone
                return img
            return thick_slices(img, params["axis"], params["k"], params["gap"])

    class nnUNetTrainerHMThick(nnUNetTrainer):
        """nnUNetTrainer, 250 epochs, + ThickSliceTransform after the intensity augmentations."""

        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device("cuda")):
            # explicit signature: nnUNetTrainer records its subclass's __init__ arguments by name
            super().__init__(plans, configuration, fold, dataset_json, device)
            self.num_epochs = 250
            if os.environ.get("HM_R5_SMOKE") == "1":   # scripts/train_r5.sh --smoke only: a few iterations
                self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 1, 5, 2

        @staticmethod
        def get_training_transforms(*args, **kwargs):
            tr = nnUNetTrainer.get_training_transforms(*args, **kwargs)
            ts = tr.transforms
            # after the last image-intensity transform (GammaTransform), before mirroring and the seg transforms
            names = [type(getattr(t, "transform", t)).__name__ for t in ts]
            at = max(i for i, n in enumerate(names) if n == "GammaTransform") + 1
            ts.insert(at, RandomTransform(ThickSliceTransform(), apply_probability=P_THICK))
            return tr

    class nnUNetTrainerHMThick430(nnUNetTrainerHMThick):
        """R6 (docs/r6_plan_draft.md): nnUNetTrainerHMThick with 430 epochs. nnU-Net's epoch is a fixed 250
        iterations, so with 947 training heads instead of R5's 548, 430 epochs (947/548 x 250 = 432) show each head
        as often as R5's 250 did. Everything else is R5's."""

        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                     device: torch.device = torch.device("cuda")):
            super().__init__(plans, configuration, fold, dataset_json, device)
            if os.environ.get("HM_R5_SMOKE") != "1":
                self.num_epochs = 430
