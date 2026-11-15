import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import scipy.io
import scipy.ndimage as ndi
import torch

from headmodel import dataset, register

CT_ROOT = Path.home() / "Projects/ct-headmodel"
CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
needs_simnibs = pytest.mark.skipif(not (register.simnibs_dir() / "resources").is_dir(), reason="simnibs missing")
needs_cuda = pytest.mark.skipif(not CUDA, reason="full-size registration needs a GPU")


def test_grid_sample_matches_scipy_on_a_flipped_grid():
    rng = np.random.default_rng(0)
    vol = rng.random((7, 9, 5)).astype(np.float32)
    aff = np.array([[-1.5, 0, 0, 10], [0, 1.0, 0, -4], [0, 0, 2.0, 3], [0, 0, 0, 1]])
    g = register.Grid(torch.as_tensor(vol)[None], aff)
    ijk = rng.random((50, 3)) * (np.array(vol.shape) - 1)
    pts = ijk @ aff[:3, :3].T + aff[:3, 3]
    ours = g.sample(torch.as_tensor(pts, dtype=torch.float32))[0].numpy()
    ref = ndi.map_coordinates(vol, ijk.T, order=1)
    assert np.allclose(ours, ref, atol=1e-5)


def test_pooled_grid_keeps_world_coordinates():
    vol = torch.zeros(1, 8, 8, 8)
    vol[0, 4:6, 2:4, 0:2] = 1   # one 2x2x2 block -> one pooled voxel
    aff = np.diag([-1.0, 1, 1, 1])
    aff[:3, 3] = (5, -3, 2)
    p = register.Grid(vol, aff).pooled((2, 2, 2))
    centre = nib.affines.apply_affine(aff, [4.5, 2.5, 0.5])
    assert np.allclose(nib.affines.apply_affine(p.affine, [2, 1, 0]), centre)
    assert p.data[0, 2, 1, 0] == 1


def test_svf_exponentials_of_plus_and_minus_v_are_inverse():
    vg = register.VelocityGrid(np.array([-40.0] * 3), np.array([40.0] * 3), 4.0, "cpu")
    x = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in vg.shape], indexing="ij"), -1)
    xm = x * 4 - 40
    v = 3 * torch.exp(-(xm ** 2).sum(-1) / 400)[None, None] * torch.tensor([1.0, -0.5, 0.3])[None, :, None, None, None]
    uf, ui = vg.exp(v, 7), vg.exp(-v, 7)
    pts = torch.tensor([[0.0, 0, 0], [5, -3, 8], [-12, 10, 2]])
    y = pts + register.Grid(uf[0], vg.affine).sample(pts, padding="border").T
    back = y + register.Grid(ui[0], vg.affine).sample(y, padding="border").T
    assert (back - pts).norm(dim=1).max() < 0.2   # trilinear interpolation on the 4 mm grid
    assert register._jacobian_det(uf, 4.0).min() > 0


def _phantom(shape, aff):
    """Charm-valued head phantom on a grid: ellipsoid shells, eyes in front, asymmetric on purpose."""
    x = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    def ell(c, r):
        return (((x - np.array(c)) / np.array(r)) ** 2).sum(-1) <= 1
    lab = np.zeros(shape, np.uint8)
    lab[ell((0, -5, 0), (78, 98, 85))] = 5                     # scalp
    lab[ell((0, -5, 0), (72, 92, 79))] = 7                     # compact bone
    lab[ell((0, -5, 0), (68, 88, 75))] = 3                     # CSF
    lab[ell((0, -8, 8), (64, 84, 70))] = 2                     # GM
    lab[ell((5, -14, 14), (45, 60, 45))] = 1                   # WM, off-centre
    for sx in (-32, 32):
        lab[ell((sx, 62, -38), (12, 12, 12))] = 6              # eyes
    lab[ell((20, -60, -50), (25, 15, 20)) & (lab == 0)] = 10   # one-sided muscle
    return lab


@pytest.fixture(scope="module")
def phantom_template(tmp_path_factory):
    """A label template made from the phantom on the MNI152 grid (stands in for the train template)."""
    mni = nib.load(register.mni_t1())
    lab = _phantom(mni.shape[:3], mni.affine)
    prob = np.zeros(mni.shape[:3] + (register.N_LABELS,), np.uint8)
    for k in range(register.N_LABELS):
        prob[..., k] = (lab == k) * 255
    d = tmp_path_factory.mktemp("template")
    img = nib.Nifti1Image(prob, mni.affine)
    img.header.set_slope_inter(1 / 255, 0)
    nib.save(img, d / register.TEMPLATE_FILE)
    return d


def _subject(true_A):
    """Phantom subject seen through a known MNI -> subject affine, on a 1 mm LPS-like T1 grid."""
    aff = np.array([[-1.0, 0, 0, 100], [0, -1, 0, 120], [0, 0, 1, -110], [0, 0, 0, 1]])
    shape = (200, 240, 210)
    ys = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T)
    xs = nib.affines.apply_affine(np.linalg.inv(true_A), ys)
    mni = nib.load(register.mni_t1())
    ijk = nib.affines.apply_affine(np.linalg.inv(mni.affine), xs).T
    lab_m = _phantom(mni.shape[:3], mni.affine)
    lab = ndi.map_coordinates(lab_m, ijk, order=0).reshape(shape).astype(np.uint8)
    t1 = ndi.map_coordinates(np.asarray(mni.dataobj, np.float32), ijk, order=1).reshape(shape)
    return nib.Nifti1Image(t1, aff), nib.Nifti1Image(lab, aff)


def _true_affine():
    ang = np.radians([4.0, -3.0, 6.0])
    cx, cy, cz = np.cos(ang)
    sx, sy, sz = np.sin(ang)
    R = (np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    A = np.eye(4)
    A[:3, :3] = R @ np.diag([0.93, 1.05, 0.97])
    A[:3, 3] = (4.0, -9.0, 12.0)
    return A


@needs_simnibs
@needs_cuda
def test_known_affine_is_recovered(phantom_template, tmp_path):
    A = _true_affine()
    t1, lab = _subject(A)
    p = register.Params(svf_stages=(), w_ncc=0.0)
    out = register.register(t1, lab, tmp_path / "m2m_syn", device=DEV, params=p, template_dir=phantom_template,
                            write=False)
    got = np.asarray(out["affine"])
    pts = np.array([[x, y, z] for x in (-60, 60) for y in (-80, 60) for z in (-40, 70)], float)
    err = np.linalg.norm(nib.affines.apply_affine(got, pts) - nib.affines.apply_affine(A, pts), axis=1)
    assert err.max() < 1.5, err
    assert out["peak_mem_gb"] < 8


@pytest.fixture(scope="module")
def synthetic_m2m(phantom_template, tmp_path_factory):
    t1, lab = _subject(_true_affine())
    m2m = tmp_path_factory.mktemp("m2m") / "m2m_syn"
    p = register.Params(svf_stages=((4.0, 8.0, 40), (2.0, 4.0, 30)))
    out = register.register(t1, lab, m2m, device=DEV, params=p, template_dir=phantom_template)
    nib.save(t1, m2m / "T1.nii.gz")
    return m2m, out, t1, lab


@needs_simnibs
@needs_cuda
def test_written_warps_have_charms_formats_and_no_zero_voxels(synthetic_m2m):
    m2m, out, t1, _ = synthetic_m2m
    atlas = nib.load(register.atlas_template())
    m2c = nib.load(m2m / "toMNI/MNI2Conform_nonl.nii.gz")
    c2m = nib.load(m2m / "toMNI/Conform2MNI_nonl.nii.gz")
    assert m2c.shape == atlas.shape[:3] + (3,) and m2c.get_data_dtype() == np.float64
    assert np.allclose(m2c.affine, atlas.affine)
    assert c2m.shape == t1.shape[:3] + (3,) and c2m.get_data_dtype() == np.float32
    assert np.allclose(c2m.affine, t1.affine)
    for img in (m2c, c2m):
        assert int(img.header["sform_code"]) == 2
        d = np.asarray(img.dataobj)
        assert np.isfinite(d).all() and not (np.abs(d).sum(-1) == 0).any()
    mat = scipy.io.loadmat(m2m / "segmentation/coregistrationMatrices.mat")
    W = mat["worldToWorldTransformMatrix"]
    assert np.allclose(W, out["affine"])
    assert np.allclose(t1.affine @ mat["imageToImageTransformMatrix"], W @ atlas.affine)
    assert out["peak_mem_gb"] < 8 and out["losses"]["jacobian_min"] > 0


@needs_simnibs
@needs_cuda
def test_simnibs_reads_the_warps_and_the_round_trip_closes(synthetic_m2m):
    import simnibs

    m2m, out, _, _ = synthetic_m2m
    rng = np.random.default_rng(1)
    mni = rng.uniform([-60, -90, -40], [60, 60, 70], (200, 3))          # inside the head
    sub = simnibs.mni2subject_coords(mni, str(m2m))
    back = simnibs.subject2mni_coords(sub, str(m2m))
    assert np.linalg.norm(back - mni, axis=1).max() <= 0.5
    # the warp is near the true affine (the phantom subject is an exact affine copy of the template)
    true = nib.affines.apply_affine(_true_affine(), mni)
    assert np.linalg.norm(sub - true, axis=1).mean() < 1.5


@needs_simnibs
def test_template_refuses_test_cases(tmp_path):
    with pytest.raises(ValueError):
        register.build_template({"X": {"m2m": str(tmp_path), "split": "test"}}, tmp_path)


def _real_case():
    if not (CT_ROOT / "runs").is_dir() or not (register.TEMPLATE_DIR / register.TEMPLATE_FILE).is_file():
        return None
    meta = json.loads((register.TEMPLATE_DIR / "template.json").read_text())
    for row in meta["cases"]:   # a train case: the template was built from it, so no test case is touched
        m2m = CT_ROOT / "runs" / row["case"] / "m2m_ref"
        if (m2m / dataset.SEG_LABELS).is_file() and (m2m / "toMNI/MNI2Conform_nonl.nii.gz").is_file():
            return m2m
    return None


@needs_simnibs
@needs_cuda
@pytest.mark.skipif(_real_case() is None, reason="ct-headmodel runs or the WP3 template not present")
def test_real_case_matches_charms_files(tmp_path):
    import simnibs

    ref = _real_case()
    t1 = nib.load(ref / "T1.nii.gz")
    up = nib.load(ref / dataset.SEG_LABELS)
    lab = dataset.block_majority(dataset._vol(up), up.affine, t1.shape[:3], t1.affine)
    m2m = tmp_path / "m2m_real"
    out = register.register(t1, nib.Nifti1Image(lab, t1.affine), m2m, device=DEV)
    nib.save(t1, m2m / "T1.nii.gz")
    for f in ("toMNI/MNI2Conform_nonl.nii.gz", "toMNI/Conform2MNI_nonl.nii.gz"):
        a, b = nib.load(m2m / f), nib.load(ref / f)
        assert a.shape == b.shape and a.get_data_dtype() == b.get_data_dtype()
        assert np.allclose(a.affine, b.affine)
        assert int(a.header["sform_code"]) == int(b.header["sform_code"])
    keys = set(scipy.io.loadmat(ref / "segmentation/coregistrationMatrices.mat")) - {"__header__", "__version__", "__globals__"}
    assert keys <= set(scipy.io.loadmat(m2m / "segmentation/coregistrationMatrices.mat"))
    mni = np.array([[-38, 44, 26], [38, 44, 26], [-37, -21, 58], [0, -8, 97]], float)
    err = np.linalg.norm(simnibs.mni2subject_coords(mni, str(m2m)) - simnibs.mni2subject_coords(mni, str(ref)), axis=1)
    assert err.max() < 8, err          # a coarse sanity bound; the accuracy gate is scripts/eval_register.py
    assert out["timings"]["gpu_total"] < 120 and out["peak_mem_gb"] < 8


def test_signed_distance_of_a_ball():
    x = np.indices((41, 41, 41)).reshape(3, -1).T * 0.5            # 0.5 mm voxels
    r = np.linalg.norm(x - 10.0, axis=1).reshape(41, 41, 41)
    d = register.signed_distance(r <= 6.0, (0.5, 0.5, 0.5), clip=3.0)
    near = np.abs(r - 6.0) < 2.5
    assert np.abs(d[near] - (r[near] - 6.0)).max() <= 0.3           # within half a voxel
    assert d.max() == 3.0 and d.min() == -3.0 and d.dtype == np.float32
    assert (register.signed_distance(np.zeros((4, 4, 4)), (1, 1, 1), 2.0) == 2.0).all()


def test_wp3b_terms_are_off_by_default():
    p = register.Params()
    assert p.w_sdt == 0 and p.w_ncc_brain == 0 and p.svf_reg_scale == ()
    assert p.svf_class_weights == {"brain": 0.0}   # WP3b default: brain Dice off in the SVF, WM kept
    assert register.Params().svf_class_weights is not p.svf_class_weights   # no shared mutable default


@needs_simnibs
@needs_cuda
def test_wp3b_terms_run_and_keep_the_warp_near_the_truth(phantom_template):
    t1, lab = _subject(_true_affine())
    p = register.Params(svf_stages=((4.0, 8.0, 30), (2.0, 4.0, 30)), w_sdt=1.0, w_ncc_brain=0.5,
                        svf_class_weights={"wm": 2.0}, svf_reg_scale=(1.0, 2.0))
    out = register.register(t1, lab, None, device=DEV, params=p, template_dir=phantom_template, write=False)
    assert out["losses"]["jacobian_min"] > 0 and out["peak_mem_gb"] < 8
    tr = out["transform"]
    mni = np.array([[x, y, z] for x in (-40, 40) for y in (-60, 30) for z in (-20, 50)], float)
    ijk = nib.affines.apply_affine(np.linalg.inv(tr["atlas_affine"]), mni).T
    sub = np.stack([ndi.map_coordinates(tr["mni2conform"][..., i], ijk, order=1) for i in range(3)], 1)
    assert np.linalg.norm(sub - nib.affines.apply_affine(_true_affine(), mni), axis=1).mean() < 1.5


def test_written_map_jacobian():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_register as ev

    aff = np.diag([2.0, -2.0, 2.0, 1.0])
    aff[:3, 3] = (-20, 30, -10)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in (12, 10, 9)], indexing="ij"), -1).astype(float)
    xyz = nib.affines.apply_affine(aff, ijk)
    scale = np.array([1.1, 0.9, 1.2])
    ident = ev.map_jacobian(nib.Nifti1Image(xyz.astype(np.float32), aff))
    scaled = ev.map_jacobian(nib.Nifti1Image((xyz * scale).astype(np.float32), aff))
    assert np.allclose(ident, 1.0, atol=1e-4) and np.allclose(scaled, scale.prod(), atol=1e-4)
    fold = xyz.copy()
    fold[..., 0] = -fold[..., 0]   # a mirrored map: negative everywhere
    assert (ev.map_jacobian(nib.Nifti1Image(fold.astype(np.float32), aff)) < 0).all()


def test_wp3_eval_keeps_the_wp3_parameters(monkeypatch, tmp_path):
    """scripts/eval_register.py writes runs/wp3: it must register with WP3's parameters and template,
    whatever the current defaults are."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_register as ev

    seen = {}

    class Stop(Exception):
        pass

    def fake(t1, lab, out, device="cuda", params=None, template_dir=None, **kw):
        seen.update(params=params, template_dir=template_dir)
        raise Stop

    m2m = tmp_path / "m2m"
    (m2m / "label_prep").mkdir(parents=True)
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), aff), m2m / "T1.nii.gz")
    up = np.diag([0.5, 0.5, 0.5, 1.0])
    up[:3, 3] = -0.25
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.uint8), up), m2m / dataset.SEG_LABELS)
    monkeypatch.setattr(ev.register, "register", fake)
    monkeypatch.setattr(ev, "OUT", tmp_path / "out")
    for call in (lambda: ev.eval_case("X", {"m2m": str(m2m)}, np.zeros((1, 3)), np.zeros((1, 3)), "cpu"),
                 lambda: ev.backfill_affine_raw({"case": "X"}, {"m2m": str(m2m)}, np.zeros((1, 3)), np.zeros((1, 3)), "cpu")):
        seen.clear()
        with pytest.raises(Stop):
            call()
        p = seen["params"]
        assert p.frame == "labels" and p.svf_class_weights is None and seen["template_dir"] == register.WP3_TEMPLATE_DIR


def _smooth_map(shape, aff, seed=0):
    """World coordinates of a grid plus a smooth bounded displacement (no folding)."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).astype(np.float64)
    x = nib.affines.apply_affine(aff, ijk)
    rng = np.random.default_rng(seed)
    d = ndi.gaussian_filter(rng.standard_normal(tuple(shape) + (3,)), (4, 4, 4, 0)) * 3
    return (x + d).astype(np.float32)


def test_map_det_matches_the_evaluation_jacobian_slab_by_slab():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_register as ev

    aff = np.array([[-1.0, 0, 0, 30], [0, 1.2, 0, -20], [0, 0, 0.9, 5], [0, 0, 0, 1]])
    m = _smooth_map((20, 18, 70), aff)
    ref = ev.map_jacobian(nib.Nifti1Image(m, aff))
    np.testing.assert_allclose(register._map_det(m, aff, chunk=16), ref, rtol=1e-4, atol=1e-5)


def test_unfold_removes_a_planted_fold_and_leaves_the_rest_alone():
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    m = _smooth_map((30, 30, 30), aff)
    assert register._map_det(m, aff).min() > 0
    orig = m.copy()
    m[15, 15, 15, 0] -= 3.0   # one voxel pushed past its neighbour along x: J < 0 next to it
    assert (register._map_det(m, aff) <= 0).any()
    info = register._unfold(m, aff, 50)
    assert info["folded"] > 0 and info["min_det"] > 0
    changed = np.abs(m - orig).max(-1) > 0
    assert 0 < changed.sum() <= info["repaired_voxels"]
    far = np.ones(m.shape[:3], bool)
    far[8:23, 8:23, 8:23] = False
    assert not changed[far].any()


def test_unfold_leaves_a_fold_free_map_untouched():
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    m = _smooth_map((16, 16, 16), aff, seed=3)
    orig = m.copy()
    info = register._unfold(m, aff, 10)
    assert info == {"folded": 0, "repaired_voxels": 0, "rounds": 0, "min_det": None}
    assert np.array_equal(m, orig)


def test_adopted_setting_is_the_round4_setting():
    p, tdir = register.adopted_setting()
    assert p.init_warp.endswith("runs/wp3b/warpnet_r3/e1000/last.pt") and tdir.name == "template_trainminusval"
    assert p.svf_stages == ((2.0, 4.0, 150),) and p.svf_class_weights == {"brain": 0.0}
    assert p.w_init == 0.0 and p.net_fill and p.unfold_iters == 50


def test_same_template_accepts_a_release_copy_only():
    from headmodel.register import same_template
    dev = "/home/u/Projects/head-model/runs/wp3b/template_trainminusval"
    assert same_template(dev, dev)
    assert same_template(dev, "/mnt/x/releases/20261102-abc/runs/wp3b/template_trainminusval")
    assert not same_template(dev, "/mnt/x/releases/20261102-abc/runs/wp3b/template")
    assert not same_template(dev, "/elsewhere/template_trainminusval")
