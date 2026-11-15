import numpy as np
import nibabel as nib
import pytest
import torch

from headmodel import frame, register
from test_register import DEV, _subject, _true_affine, needs_cuda, needs_simnibs, phantom_template  # noqa: F401

PTS = np.array([[x, y, z] for x in (-60, 60) for y in (-80, 60) for z in (-40, 70)], float)


def _err(A, B):
    return np.linalg.norm(nib.affines.apply_affine(np.asarray(A), PTS) - nib.affines.apply_affine(np.asarray(B), PTS), axis=1)


def test_mutual_information_prefers_dependent_memberships():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(5000, generator=g)
    a = frame._parzen(x, 16)
    m = torch.ones(5000)
    same = frame.mutual_information(a, a, m)
    shuf = frame.mutual_information(a, a[:, torch.randperm(5000, generator=g)], m)
    assert same > 1.0 and shuf < 0.05 and torch.allclose(a.sum(0), torch.ones(5000))


def test_gauss_cost_is_lower_for_class_constant_values():
    t = torch.zeros(2, 1000)
    t[0, :500] = 1
    t[1, 500:] = 1
    m = torch.ones(1000)
    clean = torch.cat([torch.full((500,), 0.2), torch.full((500,), 0.8)]) + 0.01 * torch.randn(1000)
    mixed = clean[torch.randperm(1000)]
    assert frame.gauss_cost(t, clean, m) < frame.gauss_cost(t, mixed, m)


def test_frame_default_is_auto():
    assert register.Params().frame == "auto" and register.RIDGE_FILE == frame.RIDGE_FILE


def _perturbed(A):
    P = np.eye(4)
    P[:3, :3] *= 1.03
    P[:3, 3] = (4.0, -3.0, 5.0)
    return P @ A


@needs_simnibs
@needs_cuda
def test_intensity_affine_recovers_a_known_affine(phantom_template):
    A = _true_affine()
    t1, _ = _subject(A)   # the MNI152 T1 seen through A
    fp = frame.FrameParams(template="mni", region="head", stages=((8.0, 120), (4.0, 100), (2.0, 60)))
    A0 = _perturbed(A)
    assert _err(A0, A).mean() > 5
    torch.cuda.reset_peak_memory_stats()
    out = frame.intensity_affine(t1, A0, fp, device=DEV, label_template_dir=phantom_template)
    assert _err(out["affine"], A).mean() < 1.0, _err(out["affine"], A)
    assert out["peak_mem_gb"] < 8


@needs_simnibs
@needs_cuda
def test_register_with_the_intensity_frame(phantom_template):
    A = _true_affine()
    t1, lab = _subject(A)
    p = register.Params(svf_stages=(), frame="intensity", frame_params={"template": "mni", "region": "brain"})
    out = register.register(t1, lab, None, device=DEV, params=p, template_dir=phantom_template, write=False)
    assert not out["calibrated"] and "frame" in out["losses"]
    assert _err(out["affine"], A).mean() < 1.5
    with pytest.raises(ValueError):
        register.register(t1, lab, None, device=DEV, params=register.Params(svf_stages=(), frame="nope"),
                          template_dir=phantom_template, write=False)


def _rand_affine(rng, scale=0.05, shift=8.0):
    A = np.eye(4)
    A[:3, :3] += rng.normal(0, scale, (3, 3))
    A[:3, 3] = rng.normal(0, shift, 3)
    return A


def test_ridge_learns_a_fixed_correction():
    rng = np.random.default_rng(0)
    D = _rand_affine(rng, 0.02, 3.0)            # charm's frame = our affine @ D, for every case
    feats, W = [], []
    for _ in range(40):
        A = _rand_affine(rng)
        feats.append([A, A @ _rand_affine(rng, 0.01, 1.0)])
        W.append(A @ D)
    m = frame.fit_ridge(feats, W)
    assert m["loo_box_mm"] < 0.2 and m["n_cases"] == 40
    A = _rand_affine(rng)
    assert _err(frame.apply_ridge([A, A], m), A @ D).max() < 0.5


@needs_simnibs
@needs_cuda
def test_register_with_the_ridge_frame(phantom_template, tmp_path):
    import json
    import shutil

    A = _true_affine()
    t1, lab = _subject(A)
    d = tmp_path / "tpl"
    shutil.copytree(phantom_template, d)
    eye = np.eye(4)   # a ridge model that changes nothing: zero coefficients, zero mean correction
    m = frame.fit_ridge([[eye, eye], [eye, eye]], [eye, eye])
    m["B"] = np.zeros_like(np.asarray(m["B"])).tolist()
    (d / frame.RIDGE_FILE).write_text(json.dumps(m))
    p = register.Params(svf_stages=(), frame="ridge")
    out = register.register(t1, lab, None, device=DEV, params=p, template_dir=d, write=False)
    assert np.allclose(out["affine"], out["affine_raw"], atol=1e-6) and not out["calibrated"]
    assert _err(out["losses"]["frame_brain_affine"], A).mean() < 2.0
    # the default frame "auto" is "ridge" with a ridge model in the template dir, "labels" without
    auto = register.register(t1, lab, None, device=DEV, params=register.Params(svf_stages=()), template_dir=d,
                             write=False)
    assert auto["losses"]["frame_kind"] == "ridge" and np.allclose(auto["affine"], out["affine"], atol=1e-3)
    plain = register.register(t1, lab, None, device=DEV, params=register.Params(svf_stages=()),
                              template_dir=phantom_template, write=False)
    assert plain["losses"]["frame_kind"] == "labels" and "frame_brain_affine" not in plain["losses"]


def test_fit_ridge_stores_cases_and_flags_grid_edge():
    rng = np.random.default_rng(3)
    eye = np.eye(4)
    feats, W = [], []
    for _ in range(12):
        A = eye.copy()
        A[:3, 3] = rng.normal(0, 5, 3)
        feats.append([A, A])
        W.append(A)
    cases = [f"c{i:02d}" for i in range(12)]
    m = frame.fit_ridge(feats, W, cases=cases)
    assert m["cases"] == cases and m["lams"] == list(frame.RIDGE_LAMS)
    assert m["lam_at_grid_edge"] == (m["lam"] in (min(frame.RIDGE_LAMS), max(frame.RIDGE_LAMS)))
    assert frame.fit_ridge(feats, W, lams=(0.1,))["lam_at_grid_edge"] is False
    with pytest.raises(ValueError):
        frame.fit_ridge(feats, W, cases=cases[:3])
    assert min(frame.RIDGE_LAMS) < 0.01 and register.Params().ridge_file == frame.RIDGE_FILE
