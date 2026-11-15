"""The output contract: which files of a SimNIBS charm head model folder (m2m_<id>) a navigation or
e-field program reads, so that a head model built here can stand in for charm's.

Two tiers:
  "locator"  what navigation needs: targets (mni2subject), scalp, EEG 10-10 positions, fiducials, cortex.
             navprep.py writes the labels, T1, warps and cortex; the skin surface, EEG positions and
             fiducials are made from them by the navigation program (check() lists what is missing).
  "efield"   what charm's full run leaves (labels rasterised from the mesh, the mesh itself), for
             SimNIBS and other FEM solvers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# charm's tissue labels (final_tissues_LUT.txt); the segmentation must use exactly these
TISSUES = {1: "White-Matter", 2: "Gray-Matter", 3: "CSF", 4: "Bone", 5: "Scalp", 6: "Eye_balls",
           7: "Compact_bone", 8: "Spongy_bone", 9: "Blood", 10: "Muscle"}

SEG_LABELS = "label_prep/tissue_labeling_upsampled.nii.gz"   # 0.5 mm, the mesh step's input
SKIN = "surfaces/skin.gii"   # closed outer skin from the labels (mm, scanner RAS), until a mesh exists

_COMMON = (
    "T1.nii.gz",
    "toMNI/Conform2MNI_nonl.nii.gz",
    "toMNI/MNI2Conform_nonl.nii.gz",
    "eeg_positions/EEG10-10_UI_Jurak_2007.csv",
    "eeg_positions/Fiducials.csv",
    "surfaces/lh.central.gii",
    "surfaces/rh.central.gii",
)
LOCATOR_FILES = _COMMON + (SEG_LABELS, "surfaces/lh.pial.gii", "surfaces/rh.pial.gii", SKIN)
EFIELD_FILES = _COMMON + ("final_tissues.nii.gz", "final_tissues_LUT.txt")   # + <id>.msh
TIERS = ("locator", "efield")


def required(subject_id: str, tier: str = "efield") -> tuple[str, ...]:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    return LOCATOR_FILES if tier == "locator" else EFIELD_FILES + (f"{subject_id}.msh",)


def _labels_problems(img, t1, factor: int, name: str) -> list[str]:
    problems = []
    found = set(np.unique(np.asarray(img.dataobj)).astype(int).tolist()) - {0}
    unknown = sorted(found - set(TISSUES))
    if unknown:
        problems.append(f"{name} has labels outside charm's LUT: {unknown}")
    if t1 is not None:
        expect = t1.affine.copy()
        if factor > 1:   # charm's upsampled grid: `factor` times finer, offset by a fraction of a voxel
            expect[:3, :3] = t1.affine[:3, :3] / factor
            expect[:3, 3] = np.asarray(t1.affine[:3, :3] @ np.full(3, (1 / factor - 1) / 2)) + t1.affine[:3, 3]
        shape = tuple(factor * s for s in t1.shape[:3])
        if tuple(img.shape[:3]) != shape or not np.allclose(img.affine, expect, atol=1e-3):
            problems.append(f"{name} is not on the {'T1' if factor == 1 else f'{factor}x T1'} grid")
    return problems


def check(m2m: str | Path, tier: str = "efield") -> list[str]:
    """Problems that would stop a downstream project from reading this head model (empty = OK)."""
    import nibabel as nib

    m2m = Path(m2m)
    sid = m2m.name.removeprefix("m2m_")
    problems = [f"missing {f}" for f in required(sid, tier) if not (m2m / f).is_file()]
    t1 = nib.load(m2m / "T1.nii.gz") if (m2m / "T1.nii.gz").is_file() else None
    name, factor = (SEG_LABELS, 2) if tier == "locator" else ("final_tissues.nii.gz", 1)
    if (m2m / name).is_file():
        problems += _labels_problems(nib.load(m2m / name), t1, factor, name)
    return problems
