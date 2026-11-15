from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from headmodel import dataset

CT_ROOT = Path.home() / "Projects/ct-headmodel"


def _grids(shape=(3, 4, 2)):
    t1_aff = np.array([[-1.0, 0, 0, 118], [0, -1, 0, 71], [0, 0, 1, -124], [0, 0, 0, 1]])
    up_aff = t1_aff.copy()
    up_aff[:3, :3] /= 2
    up_aff[:3, 3] = nib.affines.apply_affine(t1_aff, [-0.25] * 3)
    return shape, t1_aff, up_aff


def test_block_majority_takes_the_most_common_label_of_each_block():
    shape, t1_aff, up_aff = _grids()
    up = np.zeros(tuple(2 * s for s in shape), np.uint16)
    up[0:2, 0:2, 0:2] = 2                 # voxel (0,0,0): all GM
    up[2:4, 0:2, 0:2] = 3
    up[2, 0, 0] = up[3, 1, 1] = 1         # voxel (1,0,0): 6 CSF vs 2 WM
    up[4:6, 0:2, 0:2] = [[[5, 5], [5, 7]], [[7, 7], [7, 5]]]   # voxel (2,0,0): a 4-4 tie
    out = dataset.block_majority(up, up_aff, shape, t1_aff)
    assert out.shape == shape
    assert (out[0, 0, 0], out[1, 0, 0], out[2, 0, 0]) == (2, 3, 5)   # ties go to the lower label
    assert out[0, 1, 0] == 0


def test_block_majority_refuses_a_grid_that_is_not_twice_as_fine():
    shape, t1_aff, up_aff = _grids()
    up_aff[:3, 3] += 0.25                 # no quarter-voxel offset
    with pytest.raises(ValueError):
        dataset.block_majority(np.zeros(tuple(2 * s for s in shape), np.uint16), up_aff, shape, t1_aff)


def test_class_mapping_round_trips_and_is_consecutive():
    charm = np.array([0, 1, 2, 3, 5, 6, 7, 8, 9, 10], np.uint8)
    assert dataset.to_classes(charm).tolist() == list(range(10))
    assert (dataset.to_charm(dataset.to_classes(charm)) == charm).all()
    assert sorted(dataset.CLASSES.values()) == list(range(10))
    with pytest.raises(ValueError):
        dataset.to_classes(np.array([4], np.uint8))   # generic bone never appears after charm


def test_held_out_rule_matches_ct_headmodel(tmp_path):
    (tmp_path / "1BA001").mkdir()
    (tmp_path / "1BA001" / "validate.json").touch()
    assert dataset.is_test("1BA001", tmp_path) and not dataset.is_test("1BA002", tmp_path)
    assert dataset.is_test("PIT12", tmp_path) and not dataset.is_test("PIT13", tmp_path)


@pytest.mark.skipif(not (CT_ROOT / "runs").is_dir(), reason="ct-headmodel runs not present")
def test_manifest_on_ct_headmodel_runs():
    man = dataset.manifest(CT_ROOT)
    cases = man["cases"]
    assert len(cases) > 150
    assert {c["licence"] for c in cases.values()} <= {"CC BY-NC 4.0", "CC BY 4.0"}
    stale = {"MEN02", "MEN08", "PIT10", "PIT12", "PIT13", "PIT14", "PIT15", "PIT20", "PIT21", "PIT26"}
    for case in stale & set(cases):       # only after ct-headmodel's rerun replaced the stale folder
        assert not dataset.stale_origin(case, Path(cases[case]["m2m"]), CT_ROOT)
    test = {c for c, i in cases.items() if i["split"] == "test"}
    assert all(dataset.is_test(c, CT_ROOT / "runs") for c in test)


def test_manifest_r5_adds_ixi_by_the_sealed_split(tmp_path):
    def fake_m2m(c, done=True):
        m = tmp_path / c / f"m2m_{c}"
        (m / "label_prep").mkdir(parents=True)
        (m / "surfaces").mkdir()
        if done:
            for f in ("x.msh", dataset.SEG_LABELS, "surfaces/lh.central.gii"):
                (m / f).write_text("")
            (m / "charm_log.html").write_text("... charm run finished ...")
    for c in ("IXI1", "IXI2", "IXI3"):
        fake_m2m(c)
    fake_m2m("IXI4", done=False)
    r1 = {"cases": {"1BA001": {"split": "test", "m2m": "a"}, "1BA076": {"split": "train", "m2m": "b"}}, "skipped": {}}
    split = {"train": ["IXI1", "IXI4"], "val": ["IXI2"], "test": ["IXI3"]}
    m = dataset.manifest_r5(r1, split, ixi_root=tmp_path)
    assert m["dataset"] == dataset.DATASET_R5
    got = {c: (i["split"], i.get("r5_set")) for c, i in m["cases"].items()}
    assert got == {"1BA001": ("test", None), "1BA076": ("train", None), "IXI1": ("train", "train"),
                   "IXI2": ("train", "val"), "IXI3": ("test", "test")}
    assert "IXI4" in m["skipped"] and "IXI4" not in m["cases"]
