"""navprep (PLAN 10.4 step 1): the grid (LPS, 1 mm), the input checks, the
TopoFit surface conversion and the driver's navprep.json. No GPU and no model: the stages themselves
were run by hand on validation case 1BA097 (PLAN 10.4)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from headmodel import navprep

REPO = Path(__file__).resolve().parents[1]


def _img(shape=(20, 30, 40), zooms=(1.0, 1.0, 1.0), codes=("R", "A", "S")):
    aff = np.diag([*zooms, 1.0])
    aff[:3, 3] = [-10, 20, 5]
    ornt = nib.orientations.axcodes2ornt(codes)
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    img = nib.Nifti1Image(data, aff)
    tr = nib.orientations.ornt_transform(nib.orientations.io_orientation(aff), ornt)
    return nib.Nifti1Image(nib.orientations.apply_orientation(data, tr), aff @ nib.orientations.inv_ornt_aff(tr, shape))


def _world(img, ijk):
    return nib.affines.apply_affine(img.affine, ijk)


@pytest.mark.parametrize("codes", [("R", "A", "S"), ("L", "P", "S"), ("L", "A", "I"), ("A", "S", "R")])
def test_to_lps_is_lossless_and_keeps_the_world(codes):
    img = _img(codes=codes)
    out = navprep.to_lps(img)
    assert nib.aff2axcodes(out.affine) == navprep.ORIENTATION
    # the same world point holds the same value
    v = 1234.0
    ijk_in = np.argwhere(np.asanyarray(img.dataobj) == v)[0]
    ijk_out = np.argwhere(np.asanyarray(out.dataobj) == v)[0]
    assert np.allclose(_world(img, ijk_in), _world(out, ijk_out))


def test_iso_grid_covers_the_same_field_of_view():
    img = _img(shape=(100, 100, 30), zooms=(0.5, 0.5, 5.0))
    shape, aff = navprep.iso_grid(img)
    assert shape == (50, 50, 150)
    assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), 1.0)
    lo_in = _world(img, [-0.5, -0.5, -0.5])
    lo_out = _world(nib.Nifti1Image(np.zeros(shape), aff), [-0.5, -0.5, -0.5])
    assert np.allclose(lo_in, lo_out)   # the outer corners coincide


def test_conform_keeps_a_1mm_image_and_resamples_a_thick_one():
    same = navprep.conform(_img(codes=("L", "P", "S")), cval=0)
    assert same.shape == (20, 30, 40) and int(same.header["sform_code"]) == 1
    thick = navprep.conform(_img(shape=(20, 30, 8), zooms=(1.0, 1.0, 5.0)), cval=-1024)
    assert np.allclose(thick.header.get_zooms()[:3], 1.0) and thick.shape[2] == 40


def test_keep_head_keeps_dark_mri_bone_and_drops_stray_blobs():
    classes = np.zeros((30, 30, 30), np.uint8)
    classes[5:25, 5:25, 5:25] = 7          # skull shell: dark on a T1, outside the rough T1 mask
    classes[10:20, 10:20, 10:20] = 2       # brain: inside it
    classes[0, 0, 0] = 5                   # a stray voxel away from the head
    rough = np.zeros(classes.shape, bool)
    rough[10:20, 10:20, 10:20] = True
    mr = navprep.keep_head(classes, rough, "mr")
    assert (mr[5:25, 5:25, 5:25] == classes[5:25, 5:25, 5:25]).all() and mr[0, 0, 0] == 0


def test_input_checks_warn_on_thick_slices_and_a_cut_head():
    head = np.zeros((10, 10, 10), bool)
    head[3:7, 3:7, 4:] = True                       # touches the top
    src = nib.Nifti1Image(np.zeros((10, 10, 2), np.float32), np.diag([1.0, 1.0, 5.0, 1.0]))
    c = navprep.input_checks(src, head, "mr")
    assert c["voxel_mm"] == [1.0, 1.0, 5.0] and c["head_on_edge_voxels"] == {"top": 16}
    assert any("5.0 mm thick" in w for w in c["warnings"]) and any("(top)" in w for w in c["warnings"])
    head[:] = False
    head[3:7, 3:7, 3:7] = True
    ok = navprep.input_checks(nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), np.eye(4)), head, "mr")
    assert ok["warnings"] == [] and ok["head_on_edge_voxels"] == {}


def _ball(cut_top: bool) -> nib.Nifti1Image:
    """A 'head' (ball, radius 30 mm) in 6 mm axial slices from z = -45 to 45; cut_top: the ball's centre at z = 25,
    so the top slice (z = 45) runs 20 mm above it (about half its cross-section); else centred at z = -5 (inside)."""
    x, y, z = np.mgrid[-40:41, -40:41, -45:46:6].astype(float)
    zc = 25.0 if cut_top else -5.0
    vol = (x ** 2 + y ** 2 + (z - zc) ** 2 <= 30 ** 2).astype(np.float32) * 100
    return nib.Nifti1Image(vol, np.diag([-1.0, -1.0, 6.0, 1.0]))   # RAS-ish axes, flipped to LPS by to_lps


def test_input_checks_see_a_thick_slice_cut_that_the_conformed_grid_hides():
    whole, cut = _ball(False), _ball(True)
    hidden = np.zeros((81, 81, 91), bool)
    hidden[20:60, 20:60, 10:80] = True                  # conformed head stops short of every face
    ok = navprep.input_checks(whole, hidden, "mr")
    assert ok["head_on_edge_voxels"] == {} and not any("edge" in w for w in ok["warnings"])
    assert ok["edge_area_ratio"].get("top", 0.0) <= navprep.EDGE_AREA
    bad = navprep.input_checks(cut, hidden, "mr")
    assert bad["edge_area_ratio"]["top"] > navprep.EDGE_AREA
    assert any("(top)" in w for w in bad["warnings"])


def test_surface_gii_adds_c_ras(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    fs = tmp_path / "lh.pial"
    nib.freesurfer.write_geometry(str(fs), v, f, volume_info={"head": [2, 0, 20], "valid": "1  # volume info valid",
                                                              "filename": "t1.mgz", "volume": [256, 256, 256],
                                                              "voxelsize": [1, 1, 1], "xras": [-1, 0, 0],
                                                              "yras": [0, 0, -1], "zras": [0, 1, 0],
                                                              "cras": [5.0, -3.0, 2.0]})
    navprep.surface_gii(fs, tmp_path / "surfaces/lh.pial.gii")
    g = nib.load(tmp_path / "surfaces/lh.pial.gii")
    assert np.allclose(g.agg_data("NIFTI_INTENT_POINTSET"), v + [5.0, -3.0, 2.0])
    assert g.agg_data("NIFTI_INTENT_TRIANGLE").tolist() == f.tolist()


def test_driver_writes_navprep_json_from_both_stages(tmp_path):
    """scripts/navprep.py with stand-in stage interpreters that write the stage JSONs."""
    fake = tmp_path / "fakepy"
    fake.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
a = sys.argv[3:]   # -m headmodel.navprep <stage> ...
work = Path(a[a.index("--work") + 1])
if a[0] == "segment":
    work.mkdir(parents=True, exist_ok=True)
    (work / "segment.json").write_text(json.dumps({{"checks": {{"warnings": []}}, "model": "/x/Dataset602_MRHeadR5/tr",
        "folds": [0, 1], "steps": [{{"step": "segment", "s": 1.0}}]}}))
else:
    assert "--cap-gb" in a and a[a.index("--cap-gb") + 1] == "7.5"
    (work / "warp.json").write_text(json.dumps({{"steps": [{{"step": "warp", "s": 2.0}}]}}))
""")
    fake.chmod(0o755)
    m2m, work = tmp_path / "m2m_x", tmp_path / "work"
    env = {"HM_NAVPREP_SEGMENT_PY": str(fake), "HM_NAVPREP_WARP_PY": str(fake), "PATH": "/usr/bin:/bin",
           "HM_NAVPREP_LOCK": str(tmp_path / "lock")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr", str(tmp_path / "T1.nii.gz"),
                        str(m2m), "--work", str(work), "--cap-gb", "7.5"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((m2m / "navprep.json").read_text())
    assert info["provisional"] is True and info["modality"] == "mr" and info["input"] == "T1.nii.gz"
    assert [s["step"] for s in info["steps"]] == ["segment", "warp"]
    assert info["models"]["segment"] == {"model": "Dataset602_MRHeadR5", "folds": [0, 1]} and info["cap_gb"] == 7.5
    assert set(info["stage_wall_s"]) == {"segment", "warp"} and info["total_s"] >= 0 and info["queue_wait_s"] < 5


def test_driver_passes_on_an_unusable_input(tmp_path):
    fake = tmp_path / "fakepy"
    fake.write_text(f"#!{sys.executable}\nimport sys\nprint('navprep: the T1 is empty', file=sys.stderr)\nsys.exit(3)\n")
    fake.chmod(0o755)
    env = {"HM_NAVPREP_SEGMENT_PY": str(fake), "HM_NAVPREP_WARP_PY": str(fake), "PATH": "/usr/bin:/bin",
           "HM_NAVPREP_LOCK": str(tmp_path / "lock")}
    r = subprocess.run([sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr", "x.nii.gz",
                        str(tmp_path / "m2m")], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "T1 is empty" in r.stdout
    assert not (tmp_path / "m2m" / "navprep.json").exists()


def test_driver_waits_for_the_gpu_lock(tmp_path):
    """A second prep waits while another holds the lock (one 8 GB card fits one at a time)."""
    import fcntl
    import threading
    import time

    fake = tmp_path / "fakepy"
    fake.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(3)\n")   # fails at once: only the wait matters
    fake.chmod(0o755)
    lock = tmp_path / "lock"
    env = {"HM_NAVPREP_SEGMENT_PY": str(fake), "HM_NAVPREP_WARP_PY": str(fake), "PATH": "/usr/bin:/bin",
           "HM_NAVPREP_LOCK": str(lock)}
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        threading.Timer(1.0, lambda: fcntl.flock(fh, fcntl.LOCK_UN)).start()
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr", "x.nii.gz",
                            str(tmp_path / "m2m")], capture_output=True, text=True, env=env)
    assert time.perf_counter() - t0 >= 0.9 and "waiting for another prep" in r.stdout and r.returncode == 3


def test_driver_combines_several_mri_series_and_sets_the_tier(tmp_path):
    fake = tmp_path / "fakepy"
    fake.write_text(f"""#!{sys.executable}
import json, sys
from pathlib import Path
a = sys.argv[3:]
work = Path(a[a.index("--work") + 1])
work.mkdir(parents=True, exist_ok=True)
if a[0] == "combine":
    assert [Path(x).name for x in a[1:3]] == ["sag.nii.gz", "ax.nii.gz"]
    (work / "combined.nii.gz").write_text("x")
    (work / "combine.json").write_text(json.dumps({{"reference": "ax.nii.gz", "ncc": [0.9],
        "series": [{{"spacing_mm": 6.0}}, {{"spacing_mm": 5.5}}], "steps": [{{"step": "combine", "s": 9.0}}]}}))
elif a[0] == "segment":
    assert Path(a[1]).name == "combined.nii.gz"
    (work / "segment.json").write_text(json.dumps({{"checks": {{"voxel_mm": [1, 1, 1], "warnings": []}},
        "model": "/x/Dataset602_MRHeadR5/tr", "folds": [0], "steps": [{{"step": "segment", "s": 1.0}}]}}))
else:
    (work / "warp.json").write_text(json.dumps({{"steps": [{{"step": "warp", "s": 2.0}}]}}))
""")
    fake.chmod(0o755)
    m2m = tmp_path / "m2m_x"
    env = {"HM_NAVPREP_SEGMENT_PY": str(fake), "HM_NAVPREP_WARP_PY": str(fake), "PATH": "/usr/bin:/bin",
           "HM_NAVPREP_LOCK": str(tmp_path / "lock")}
    cmd = [sys.executable, str(REPO / "scripts/navprep.py"), "--modality", "mr", str(tmp_path / "sag.nii.gz"),
           str(tmp_path / "ax.nii.gz"), str(m2m), "--work", str(tmp_path / "work")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((m2m / "navprep.json").read_text())
    assert info["tier"] == "thick-combined" and info["input"] == "sag.nii.gz, ax.nii.gz"
    assert [s["step"] for s in info["steps"]] == ["combine", "segment", "warp"]
    assert "combine" in info["stage_wall_s"] and info["combine"]["ncc"] == [0.9]
    assert info["checks"]["warnings"][0].startswith("combined from 2 series (slices up to 6.0 mm)")
    cmd[3] = "ct"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    assert r.returncode == 2   # MRI only: argparse rejects any other modality


def test_tier_from_the_input():
    sys.path.insert(0, str(REPO / "scripts"))
    import navprep as driver
    assert driver.tier([Path("a")], [1.0, 1.0, 1.2]) == "1mm"
    assert driver.tier([Path("a")], [0.5, 0.5, 5.5]) == "thick"
    assert driver.tier([Path("a"), Path("b")], [1.0]) == "thick-combined"
