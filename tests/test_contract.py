import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from headmodel import contract

ERNIE = Path(os.environ.get("HM_ERNIE_M2M", "/nonexistent/m2m_ernie"))   # SimNIBS example head (charm run)
CHARM46 = Path.home() / "Projects/ct-headmodel/runs/1BA001/m2m_ref"   # a full SimNIBS 4.6 charm run


@pytest.mark.skipif(not ERNIE.is_dir(), reason="SimNIBS ernie example not extracted")
def test_charm_full_run_meets_the_efield_tier():
    assert contract.check(ERNIE, "efield") == []


@pytest.mark.skipif(not CHARM46.is_dir(), reason="ct-headmodel charm run not present")
def test_charm_46_run_lacks_only_the_skin_for_the_locator_tier():
    # SimNIBS 4.6 writes pial surfaces and the 0.5 mm labels; skin.gii is the locator-ready step's
    assert contract.check(CHARM46, "locator") == [f"missing {contract.SKIN}"]


def _t1(m2m, shape=(4, 4, 4)):
    aff = np.array([[-1.0, 0, 0, 10], [0, 1, 0, -5], [0, 0, 1, 3], [0, 0, 0, 1]])
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), aff), m2m / "T1.nii.gz")
    return aff


def test_locator_tier_needs_no_mesh_and_checks_the_upsampled_grid(tmp_path):
    m2m = tmp_path / "m2m_x"
    for f in contract.required("x", "locator"):
        (m2m / f).parent.mkdir(parents=True, exist_ok=True)
        (m2m / f).touch()
    aff = _t1(m2m)
    up = aff.copy()
    up[:3, :3] /= 2
    up[:3, 3] = nib.affines.apply_affine(aff, [-0.25] * 3)
    labels = np.zeros((8, 8, 8), np.uint8)
    labels[1, 1, 1], labels[2, 2, 2] = 1, 5
    nib.save(nib.Nifti1Image(labels, up), m2m / contract.SEG_LABELS)
    assert contract.check(m2m, "locator") == []
    assert "missing x.msh" in contract.check(m2m, "efield")
    nib.save(nib.Nifti1Image(labels, aff), m2m / contract.SEG_LABELS)   # 1 mm grid: wrong
    assert contract.check(m2m, "locator") == [f"{contract.SEG_LABELS} is not on the 2x T1 grid"]


def test_unknown_labels_and_wrong_grid_are_reported(tmp_path):
    m2m = tmp_path / "m2m_y"
    m2m.mkdir()
    labels = np.zeros((4, 4, 4), np.uint8)
    labels[0, 0, 0] = 42
    nib.save(nib.Nifti1Image(labels, np.eye(4)), m2m / "final_tissues.nii.gz")
    nib.save(nib.Nifti1Image(labels, np.diag([2.0, 2, 2, 1])), m2m / "T1.nii.gz")
    problems = contract.check(m2m, "efield")
    assert "final_tissues.nii.gz has labels outside charm's LUT: [42]" in problems
    assert "final_tissues.nii.gz is not on the T1 grid" in problems
