import numpy as np
import scipy.ndimage as ndi

from headmodel import rescan


def _phantom(shape=(64, 60, 56), seed=0):
    """Smooth random 'head': blobs inside an ellipsoid, asymmetric so a rigid fit is well posed."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij"), -1)
    ell = ((g / np.array([0.6, 0.55, 0.5])) ** 2).sum(-1) < 1   # stays inside the FOV after the test motion
    noise = ndi.gaussian_filter(rng.standard_normal(shape), 2.5)
    vol = ndi.gaussian_filter((ell * (1 + 3 * noise + 0.5 * g[..., 0])).astype(float), 1.5).astype(np.float32)   # no step edges
    return vol, ell


def _aff(zooms, shift=(0, 0, 0)):
    A = np.diag(list(zooms) + [1.0])
    A[:3, 3] = -np.asarray(zooms) * (np.array([64, 60, 56]) - 1) / 2 + np.asarray(shift)
    return A


def test_rotvec_matrix_is_a_rotation_and_rigid_inverts():
    import torch

    R = rescan.rotvec_matrix(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)).numpy()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1)
    M = rescan.rigid([0.1, -0.2, 0.3], [5, -3, 2], centre=[10, 20, 30])
    x = np.random.default_rng(1).random((20, 3)) * 50
    assert np.allclose(rescan.apply(np.linalg.inv(M), rescan.apply(M, x)), x)
    assert np.allclose(rescan.apply(M, [[10, 20, 30]]), [[15, 17, 32]])   # the centre only moves by t


def test_polar_rotation_and_init_from_affines():
    R = rescan.rigid([0.05, 0.1, -0.08], [0, 0, 0])[:3, :3]
    W_a = np.eye(4)
    W_a[:3, :3] = np.diag([1.1, 0.9, 1.0])
    W_b = rescan.rigid([0.05, 0.1, -0.08], [3, 4, 5]) @ W_a
    M = rescan.init_from_affines(W_a, W_b, np.array([1.0, 2, 3]), np.array([7.0, 8, 9]))
    assert np.allclose(M[:3, :3], polar := rescan.polar_rotation(R), atol=1e-9) and np.allclose(polar, R)
    assert np.allclose(rescan.apply(M, [[1, 2, 3]]), [[7, 8, 9]])


def test_register_rigid_identity_and_known_transform():
    vol, ell = _phantom()
    aff = _aff((1.0, 1.0, 1.25))
    # A vs A: identity from a perturbed start
    r0 = rescan.register_rigid(vol, aff, vol, aff, ell, init=rescan.rigid([0.03, -0.02, 0.02], [2, -1.5, 1]),
                               levels=((2.0, 2), (0.0, 1)))
    pts = rescan.apply(aff, np.argwhere(ell)[::50])
    assert np.abs(rescan.apply(r0["M"], pts) - pts).max() < 0.1
    # B = A seen through a known rigid transform (world_B = T world_A) on a shifted grid: resampled,
    # so interpolation differs from A
    T = rescan.rigid([0.06, -0.04, 0.08], [4.0, -3.0, 2.5], centre=[1, 2, 3])
    aff_b = T @ _aff((1.0, 1.0, 1.25), shift=(0.4, -0.3, 0.5))
    ijk_b = np.stack(np.meshgrid(*[np.arange(n) for n in vol.shape], indexing="ij"), -1).reshape(-1, 3)
    vox_a = rescan.apply(np.linalg.inv(aff) @ np.linalg.inv(T) @ aff_b, ijk_b)
    vol_b = ndi.map_coordinates(vol, vox_a.T, order=3).reshape(vol.shape).astype(np.float32)
    r = rescan.register_rigid(vol, aff, vol_b, aff_b, ell, levels=((2.0, 2), (1.0, 1), (0.0, 1)))
    err = np.linalg.norm(rescan.apply(r["M"], pts) - rescan.apply(T, pts), axis=1)
    assert err.max() < 0.2, err.max()
    assert r["ncc"] > r["ncc_init"]


def test_pair_errors_and_summarise():
    T = rescan.rigid([0.1, 0.0, -0.05], [3, 2, 1])
    rng = np.random.default_rng(2)
    pa = {"targets": rng.random((10, 3)) * 80, "eeg": rng.random((80, 3)) * 100}
    pb = {g: rescan.apply(T, v) for g, v in pa.items()}
    e = rescan.pair_errors(pa, pb, T)
    assert all(np.allclose(v, 0, atol=1e-9) for v in e.values())
    pb["eeg"][0] += [0, 0, 6.0]   # one point 6 mm off along z (B space, rotated back into A)
    e = rescan.pair_errors(pa, pb, T)
    assert np.isclose(e["eeg"][0], 6.0) and np.isclose(e["eeg"].mean(), 6 / 80)
    pairs = [{"errors": e}, {"errors": {"targets": np.full(10, 1.0), "eeg": np.full(80, 3.0)}}]
    s = rescan.summarise(pairs)
    assert s["n_pairs"] == 2
    assert np.isclose(s["eeg"]["mean"], (6 / 80 + 3) / 2) and np.isclose(s["eeg"]["max"], 6.0)
    assert s["eeg"]["pairs_pass"] == 0 and s["targets"]["pairs_pass"] == 2
    assert s["eeg"]["points_over_max"] == 1 and s["eeg"]["n_points"] == 160
    assert np.isclose(s["targets"]["sd"], np.std([0.0, 1.0], ddof=1))


def test_labels_on_grid_nearest():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[5:15, 5:15, 5:15] = 2
    aff_up = np.diag([0.5, 0.5, 0.5, 1.0])
    out = rescan.labels_on_grid(lab, aff_up, (10, 10, 10), np.diag([1.0, 1, 1, 1]))
    assert out[3, 3, 3] == 2 and out[0, 0, 0] == 0 and out[8, 8, 8] == 0


def test_labels_majority_matches_block_majority_on_a_2x_grid():
    from headmodel import dataset

    rng = np.random.default_rng(3)
    up = ndi.zoom(rng.integers(0, 4, (5, 6, 4)).astype(np.uint8), 4, order=0)   # 20 x 24 x 16, 0.5 mm
    t1_aff = np.diag([1.0, 1.0, 1.0, 1.0])
    up_aff = t1_aff.copy()
    up_aff[:3, :3] /= 2
    up_aff[:3, 3] = -0.25
    ref = dataset.block_majority(up, up_aff, (10, 12, 8), t1_aff)
    assert np.array_equal(rescan.labels_majority(up, up_aff, (10, 12, 8), t1_aff), ref)
    # 1.25 mm slices: every coarse voxel gets a label present in its slab
    aff_z = np.diag([1.0, 1.0, 1.25, 1.0])
    out = rescan.labels_majority(up, up_aff, (10, 12, 6), aff_z)
    assert out.shape == (10, 12, 6) and set(np.unique(out)) <= set(np.unique(up))


def _scripts():
    import sys
    from pathlib import Path

    p = str(Path(__file__).resolve().parents[1] / "scripts")
    if p not in sys.path:
        sys.path.insert(0, p)


def test_head_faces_names_sides_from_the_affine():
    _scripts()
    import eval_rescan as er

    lab = np.zeros((10, 12, 8), np.uint8)
    lab[0, 2:5, 2:5] = 1      # 9 voxels on the i = 0 face
    lab[3:5, 11, 1:4] = 5     # 6 voxels on the j = last face
    aff = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # i -> A, j -> S, k -> L
    fc = er.head_faces(lab, aff)
    assert fc == {"P": 9, "A": 0, "I": 0, "S": 6, "R": 0, "L": 0}


def test_summary_drops_points_near_the_fov_edge():
    _scripts()
    import eval_rescan as er

    rows = [{"errors": {"targets": [1.0, 1.0], "eeg_unprojected": [1.0, 9.0]},
             "fov_margin_mm": {"targets": [30.0, 30.0], "eeg_unprojected": [30.0, 2.0]}},
            {"errors": {"targets": [2.0, 2.0], "eeg_unprojected": [7.0]},
             "fov_margin_mm": {"targets": [30.0, 30.0], "eeg_unprojected": [-1.0]}}]
    s_all, s_in = er.summary(rows), er.summary(rows, 10.0)
    assert s_all["eeg_unprojected"]["max"] == 9.0 and s_all["eeg_unprojected"]["n_pairs"] == 2
    assert s_in["eeg_unprojected"]["max"] == 1.0 and s_in["eeg_unprojected"]["n_pairs"] == 1
    assert np.isclose(s_in["targets"]["mean"], 1.5) and "eeg_same_skin" not in s_in


def test_failed_charm_runs_are_not_pending(tmp_path, monkeypatch):
    _scripts()
    import eval_rescan as er

    log = tmp_path / "q.log"
    log.write_text("2026-10-28 07:05:02 60 cases\n2026-10-28 07:06:00 OAS1_TEST1_MR1_mpr1 start\n"
                   "2026-10-28 07:06:00 OAS1_TEST2_MR1_mpr1 start\n2026-10-28 09:00:00 OAS1_TEST1_MR1_mpr1 FAILED (x)\n"
                   "2026-10-28 09:01:00 OAS1_TEST3_MR1_mpr1 start\n2026-10-28 11:00:00 OAS1_TEST3_MR1_mpr1 done\n")
    ev_ = er.queue_events(log)
    assert ev_ == {"OAS1_TEST1_MR1_mpr1": "FAILED", "OAS1_TEST2_MR1_mpr1": "start", "OAS1_TEST3_MR1_mpr1": "done"}
    monkeypatch.setattr(er, "ROOT", tmp_path)
    assert er.status("OAS1_TEST1_MR1_mpr1", ev_) == "failed"
    assert er.status("OAS1_TEST2_MR1_mpr1", ev_) == "pending"
    (tmp_path / "OAS1_TEST4_MR1_mpr1").mkdir()
    (tmp_path / "OAS1_TEST4_MR1_mpr1" / "charm.out").write_text("...\nTraceback (most recent call last):\n  x\n")
    assert er.status("OAS1_TEST4_MR1_mpr1", ev_) == "failed"


def test_preregister_seals_once_and_detects_edits(tmp_path):
    import json
    import os

    import pytest

    _scripts()
    import preregister as pr

    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"what": "x", "rule": ["a"]}))
    with pytest.raises(ValueError):   # no no-folding requirement
        pr.seal(draft, "r1", tmp_path / "prereg")
    draft.write_text(json.dumps({"what": "x", "rule": ["a"], "no_folding": "written-map min J > 0 on all val cases"}))
    h = pr.seal(draft, "r1", tmp_path / "prereg")
    assert pr.verify("r1", tmp_path / "prereg") and pr.ledger(tmp_path / "prereg")["r1.json"]["sha256"] == h
    with pytest.raises(FileExistsError):
        pr.seal(draft, "r1", tmp_path / "prereg")
    f = tmp_path / "prereg" / "r1.json"
    os.chmod(f, 0o644)
    f.write_text(f.read_text().replace('"a"', '"b"'))
    assert not pr.verify("r1", tmp_path / "prereg")


def test_summary_adds_the_rtms_set_from_the_whole_cap():
    _scripts()
    import eval_register as ev
    import eval_rescan as er

    names = ev.eeg_cap()[0]
    e = np.ones(len(names))
    e[names.index("TP10")] = 9.0   # outside the rTMS set
    e2 = e.copy()
    e2[names.index("C3")] = 6.0
    rows = [{"errors": {"targets": [1.0], "eeg_unprojected": e.tolist()},
             "fov_margin_mm": {"targets": [30.0], "eeg_unprojected": [30.0] * len(names)}},
            {"errors": {"targets": [1.0], "eeg_unprojected": e2.tolist()},
             "fov_margin_mm": {"targets": [30.0], "eeg_unprojected": [30.0] * len(names)}}]
    s = er.summary(rows)
    assert s["eeg_unprojected"]["max"] == 9.0 and s["eeg_rtms_unprojected"]["max"] == 6.0
    assert s["eeg_rtms_unprojected"]["n_pairs"] == 2 and "eeg_rtms_same_skin" not in s
