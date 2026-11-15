"""Synthetic, fast (CPU) tests of headmodel.warpnet."""
import json

import nibabel as nib
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from headmodel import register, warpnet

SPEC = warpnet.GridSpec(lo=(-15.0, -17.0, -13.0), shape=(16, 16, 16), spacing=2.0)
NCFG = warpnet.NetConfig(channels=(4, 8), lo=SPEC.lo, shape=SPEC.shape, spacing=SPEC.spacing, steps=5)


def _xg():
    return SPEC.points("cpu")


def test_unet_shape_and_zero_init():
    m = warpnet.build_model(NCFG)
    x = torch.randn(2, warpnet.N_IN, *SPEC.shape)
    v = warpnet.velocity(m, x, SPEC)
    assert v.shape == (2, 3, *SPEC.shape)
    assert v.abs().max() < 1e-2   # the head starts near zero: the identity warp


def test_residual_inverts_the_affine():
    rng = np.random.default_rng(1)
    A = warpnet.random_affine(rng, 10, 0.1, 0.05, 10)
    x = rng.normal(0, 50, (20, 3))
    r = rng.normal(0, 3, (20, 3))
    y = nib.affines.apply_affine(A, x + r)
    assert np.allclose(warpnet.residual(y, A, x), r, atol=1e-8)


def test_random_affine_has_no_flip_and_is_small():
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = warpnet.random_affine(rng)
        assert np.linalg.det(T[:3, :3]) > 0
        pts = np.array([[60.0, 0, 0], [0, 80, 0], [0, -100, 60]])
        assert np.linalg.norm(nib.affines.apply_affine(T, pts) - pts, axis=1).max() < 20


def test_augmentation_is_consistent_with_a_changed_affine():
    """charm(x) = A(x + r(x)) must also hold as (A o T)(x + r'(x)), and the input becomes x_in(T(x))."""
    rng = np.random.default_rng(3)
    T = warpnet.random_affine(rng, 5, 0.05, 0.02, 3, centre=(0, 0, 0))
    xg = _xg()
    r = torch.stack([torch.sin(xg[..., 0] / 7), torch.cos(xg[..., 1] / 9), 0.5 * xg[..., 2] / 15], 0)
    # input: linear ramps, so resampling is exact (inside the grid)
    x_in = torch.cat([xg.permute(3, 0, 1, 2), torch.zeros(warpnet.N_IN - 3, *SPEC.shape)], 0)
    pts = torch.tensor([[1.0, 2, 3], [-5, 4, 0]])
    r_pts = torch.tensor([[0.5, -0.2, 0.1], [1.0, 0.0, -1.0]])
    x_new, r_new, rp_new = warpnet.augment(x_in, r, r_pts, pts, T, SPEC)
    Tt = torch.as_tensor(T, dtype=torch.float32)
    lhs = (xg + r_new.permute(1, 2, 3, 0)) @ Tt[:3, :3].T + Tt[:3, 3]
    assert torch.allclose(lhs, xg + r.permute(1, 2, 3, 0), atol=1e-4)
    lhs_p = (pts + rp_new) @ Tt[:3, :3].T + Tt[:3, 3]
    assert torch.allclose(lhs_p, pts + r_pts, atol=1e-4)
    Tx = xg @ Tt[:3, :3].T + Tt[:3, 3]
    inside = ((Tx >= torch.tensor(SPEC.lo)) & (Tx <= torch.tensor(SPEC.hi, dtype=torch.float32))).all(-1)
    assert torch.allclose(x_new[:3].permute(1, 2, 3, 0)[inside], Tx[inside], atol=1e-3)


def test_sample_charm_map_on_a_synthetic_warp(tmp_path):
    """A charm-like MNI2Conform map (atlas grid, diag(-1,1,1)) that is an affine of MNI mm, with zeros
    on a slab: values at grid points and points, and validity."""
    aff = np.diag([-1.0, 1, 1, 1])
    aff[:3, 3] = (20, -22, -18)
    shape = (41, 44, 40)
    x = nib.affines.apply_affine(aff, np.indices(shape).reshape(3, -1).T).reshape(*shape, 3)
    A = np.array([[1.02, 0.01, 0, 3], [0, 0.98, 0.02, -5], [0.01, 0, 1.0, 7], [0, 0, 0, 1]])
    m = nib.affines.apply_affine(A, x.reshape(-1, 3)).reshape(*shape, 3)
    m[:, :, :8] = 0   # charm leaves zeros (z < -10 mm)
    (tmp_path / "toMNI").mkdir()
    nib.save(nib.Nifti1Image(m, aff), tmp_path / "toMNI/MNI2Conform_nonl.nii.gz")
    pts = np.array([[0.3, 1.7, -2.2], [5.0, 5.0, -17.5]])
    y, valid, yp, vp = warpnet.sample_charm_map(tmp_path, SPEC, pts)
    xs = _xg().numpy().astype(float)
    assert valid.any() and not valid.all()
    assert np.allclose(y[valid], nib.affines.apply_affine(A, xs[valid]), atol=1e-6)
    assert vp.tolist() == [True, False]
    assert np.allclose(yp[0], nib.affines.apply_affine(A, pts[0]), atol=1e-6)
    r = warpnet.residual(y[valid], A, xs[valid])
    assert np.abs(r).max() < 1e-6


def test_yield_reason_reads_the_round1_log(tmp_path, monkeypatch):
    log = tmp_path / "train_round1.log"
    log.write_text("2026-10-27 23:57:14 waiting for ct-headmodel's CPU queue (timestamped ALL_DONE)\n")
    running = warpnet.yield_reason(tmp_path / "none.log")
    if running is None:   # no nnU-Net job on this machine right now
        assert warpnet.yield_reason(log) is None
    log.write_text(log.read_text() + "2026-10-28 03:00:00 slot free: 4 charm jobs, nnUNet_n_proc_DA=8\n")
    reason = warpnet.yield_reason(log)   # a running nnU-Net job is reported first
    assert reason is not None and ("round 1" in reason or running is not None)
    if running is None:
        monkeypatch.setenv("HM_WARPNET_IGNORE_ROUND1_LOG", "1")
        assert warpnet.yield_reason(log) is None


def _samples(n, seed=0):
    """Synthetic samples: the residual is a smooth function of a label blob position, so it can be learnt."""
    rng = np.random.default_rng(seed)
    xg = _xg()
    out = []
    for i in range(n):
        c = torch.tensor(rng.uniform(-4, 4, 3), dtype=torch.float32)
        blob = torch.exp(-((xg - c) ** 2).sum(-1) / 40)
        x = torch.zeros(warpnet.N_IN, *SPEC.shape)
        x[0] = blob
        x[len(warpnet.LABELS)] = blob
        x[-1] = 1
        r = (c[:, None, None, None] * 0.5 * torch.exp(-(xg ** 2).sum(-1) / 200)[None])
        pts = torch.tensor([[0.0, 0, 0], [3, -2, 1]])
        r_pts = register.Grid(r, SPEC.affine).sample(pts).T
        out.append({"case": f"S{i}", "x": x.half(), "r": r.half(), "valid": torch.ones(SPEC.shape, dtype=torch.bool),
                    "pts": pts, "r_pts": r_pts, "valid_pts": torch.ones(2, dtype=torch.bool), "n_targets": 1,
                    "A": torch.eye(4)})
    return out


def test_training_reduces_the_loss_checkpoints_resumes_and_yields(tmp_path):
    torch.manual_seed(0)
    tr, va = _samples(4), _samples(2, seed=1)
    tcfg = warpnet.TrainConfig(epochs=30, batch=2, lr=3e-3, val_every=10, p_aug=0.0, t1_jitter=0.0, warmup_epochs=0)
    res = warpnet.train(tr, va, tmp_path, NCFG, tcfg, "cpu", max_epochs_this_run=20, guard=lambda: None,
                        log=lambda m: None)
    assert res["epoch"] == 20 and res["stopped"] == "max epochs for this run"
    hist = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert hist[-1]["loss"] < 0.6 * hist[0]["loss"]
    assert (tmp_path / "best.pt").is_file() and (tmp_path / "last.pt").is_file()
    calls = []

    def guard():   # free at the start, then busy after the first epoch
        calls.append(1)
        return None if len(calls) == 1 else "nnU-Net process running (test)"
    res = warpnet.train(tr, va, tmp_path, NCFG, tcfg, "cpu", guard=guard, log=lambda m: None)
    assert res["epoch"] == 21 and res["stopped"].startswith("yielded")
    res = warpnet.train(tr, va, tmp_path, NCFG, tcfg, "cpu", guard=lambda: "busy", log=lambda m: None)
    assert res["epoch"] == 21 and res["stopped"] == "busy"


def test_predict_velocity_from_a_checkpoint(tmp_path):
    m = warpnet.build_model(NCFG)
    from dataclasses import asdict
    torch.save({"model": m.state_dict(), "config": asdict(NCFG)}, tmp_path / "net.pt")
    aff = np.diag([-2.0, 2, 2, 1])
    aff[:3, 3] = (20, -20, -20)
    lab = np.zeros((22, 22, 22), np.uint8)
    lab[5:17, 5:17, 5:17] = 5
    lab[8:14, 8:14, 8:14] = 2
    t1 = nib.Nifti1Image(lab.astype(np.float32) * 10, aff)
    v, vg = warpnet.predict_velocity(tmp_path / "net.pt", t1, lab, np.eye(4), "cpu")
    assert v.shape == (1, 3, *SPEC.shape) and tuple(vg.shape) == SPEC.shape
    x = warpnet.net_input(t1, lab, np.eye(4), "cpu", SPEC)
    assert x.shape == (warpnet.N_IN, *SPEC.shape)
    assert x[-1].max() == 1 and x[:len(warpnet.LABELS)].sum(0).max() <= 1 + 1e-5


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="full-size registration needs a GPU")
needs_simnibs = pytest.mark.skipif(not (register.simnibs_dir() / "resources").is_dir(), reason="simnibs missing")


@needs_simnibs
@needs_cuda
def test_register_starts_from_the_net(tmp_path):
    """register() with init_warp: the net's frame (raw affine) and velocity; an untrained net (zero
    head) gives the affine map, a wrong template_dir is refused."""
    from dataclasses import asdict

    from test_register import _phantom, _subject, _true_affine

    mni = nib.load(register.mni_t1())
    lab = _phantom(mni.shape[:3], mni.affine)
    prob = np.stack([(lab == k) * 255 for k in range(register.N_LABELS)], -1).astype(np.uint8)
    tdir = tmp_path / "template"
    tdir.mkdir()
    img = nib.Nifti1Image(prob, mni.affine)
    img.header.set_slope_inter(1 / 255, 0)
    nib.save(img, tdir / register.TEMPLATE_FILE)
    cfg = warpnet.NetConfig(channels=(2, 4, 4, 4, 4), template_dir=str(tdir))
    torch.save({"model": warpnet.build_model(cfg).state_dict(), "config": asdict(cfg)}, tmp_path / "net.pt")
    t1, sub_lab = _subject(_true_affine())
    p = register.Params(svf_stages=(), init_warp=str(tmp_path / "net.pt"))
    out = register.register(t1, sub_lab, tmp_path / "m2m", device="cuda", params=p, template_dir=tdir, write=False)
    assert out["affine"] == out["affine_raw"] and not out["calibrated"]
    m2c = out["transform"]["mni2conform"]
    aff = out["transform"]["atlas_affine"]
    ijk = np.array([[91, 119, 141], [60, 150, 200]])
    x = nib.affines.apply_affine(aff, ijk)
    lin = nib.affines.apply_affine(np.asarray(out["affine"]), x)
    assert np.abs(m2c[tuple(ijk.T)] - lin).max() < 0.05
    assert out["peak_mem_gb"] < 8
    p2 = register.Params(svf_stages=((4.0, 8.0, 5),), init_warp=str(tmp_path / "net.pt"))
    out2 = register.register(t1, sub_lab, tmp_path / "m2m", device="cuda", params=p2, template_dir=tdir, write=False)
    assert out2["losses"]["jacobian_min"] > 0
    other = tmp_path / "other_template"
    other.mkdir()
    (other / register.TEMPLATE_FILE).symlink_to(tdir / register.TEMPLATE_FILE)
    with pytest.raises(ValueError, match="trained on the raw affine"):
        register.register(t1, sub_lab, tmp_path / "m2m", device="cuda", params=p, template_dir=other, write=False)


def test_harmonic_fill_keeps_known_values_and_extends_a_linear_field_smoothly():
    """Filled from a known box, a linear field continues across the box's faces and its steps stay
    bounded (the harmonic extension bends at the box's edges and corners, so not exactly linear),
    known voxels are untouched and no value leaves the known range (the garbage outside is gone)."""
    shape = (20, 24, 18)
    ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in shape], indexing="ij"))
    lin = torch.stack([0.5 * ijk[0] - 0.2 * ijk[2], 0.3 * ijk[1], torch.full(shape, 2.0)])[None]
    known = torch.zeros(shape, dtype=torch.bool)
    known[4:16, 3:21, 2:16] = True
    v = torch.where(known, lin, torch.full_like(lin, 50.0))     # garbage outside
    u = warpnet.harmonic_fill(v, known, iters=200)
    assert torch.equal(u[:, :, known], v[:, :, known])
    ring = F.max_pool3d(known[None, None].float(), 3, 1, 1)[0, 0].bool() & ~known
    assert (u - lin)[:, :, ring].abs().max() < 2.0               # continuous across the faces (garbage: 50)
    grad = torch.stack([F.pad(u.diff(dim=d), [0, 1] if d == 4 else [0, 0, 0, 1] if d == 3 else [0, 0, 0, 0, 0, 1])
                        for d in (2, 3, 4)], 1).norm(dim=1)
    assert grad.max() <= 2 * 0.62                                # |grad lin| = 0.62
    assert u[0, 2].sub(2.0).abs().max() < 1e-4                   # a constant stays constant
    assert u.max() <= v[:, :, known].max() + 1e-4 and u.min() >= v[:, :, known].min() - 1e-4


def test_net_support_from_train_cases_is_cached(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i, c in enumerate(["a", "b", "c"]):
        valid = np.zeros((4, 4, 4), np.uint8)
        valid[: i + 2] = 1                                       # a: x<2, b: x<3, c: x<4
        np.savez_compressed(cache / f"{c}.npz", valid=valid)
    torch.save({"train_cases": ["c", "a", "b"]}, tmp_path / "best.pt")
    m = warpnet.net_support(tmp_path / "best.pt", cache=cache)
    assert m.dtype == torch.bool and m[:3].all() and not m[3:].any()    # valid in >= 2 of 3
    assert (tmp_path / "best_support.npz").is_file()
    (cache / "a.npz").unlink()                                   # the cached mask is reused
    assert torch.equal(warpnet.net_support(tmp_path / "best.pt", cache=cache), m)
    torch.save({"train_cases": ["a"], "support": np.ones((4, 4, 4), bool)}, tmp_path / "own.pt")
    assert warpnet.net_support(tmp_path / "own.pt", cache=cache).all()


def test_tail_term_is_the_mean_of_the_worst_points():
    torch.manual_seed(0)
    m = warpnet.build_model(NCFG)
    vg = SPEC.velocity_grid("cpu")
    batch = _samples(2)
    off = warpnet.TrainConfig(p_aug=0.0, t1_jitter=0.0)
    on = warpnet.TrainConfig(p_aug=0.0, t1_jitter=0.0, w_tail=2.0, tail_frac=0.5)
    l0, p0, u, _ = warpnet.losses_for(m, batch, SPEC, vg, off, NCFG.steps, "cpu")
    l1, p1, _, _ = warpnet.losses_for(m, batch, SPEC, vg, on, NCFG.steps, "cpu")
    assert "tail" not in p0
    pe = torch.cat([warpnet.point_errors(u[b], vg, s["pts"], s["r_pts"]) for b, s in enumerate(batch)])
    assert p1["tail"] == pytest.approx(pe.topk(2).values.mean().item(), rel=1e-5)   # 4 points, worst half
    assert p1["tail"] >= p1["pts"] - 1e-6
    assert l1.item() == pytest.approx(l0.item() + 2.0 * p1["tail"], rel=1e-5)


def test_shared_gpu_ignores_other_jobs_but_not_low_memory(tmp_path, monkeypatch):
    log = tmp_path / "train_round1.log"
    log.write_text("2026-10-28 03:00:00 slot free: 4 charm jobs, nnUNet_n_proc_DA=8\n")
    monkeypatch.setenv("HM_WARPNET_SHARE_GPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (20 * 2 ** 30, 32 * 2 ** 30))
    assert warpnet.yield_reason(log) is None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (2 * 2 ** 30, 32 * 2 ** 30))
    assert "2.0 GB free" in warpnet.yield_reason(log)


def test_gate_points_are_the_rule_sets():
    g = warpnet.gate_points()
    pts, nt = warpnet.mni_points()
    assert len(g["targets"]) == len(pts) == 90 and g["targets"].sum() == nt == 10
    assert g["rtms"].sum() == 13 and g["fid"].sum() == 4   # wp3_gate_rtms_set: 13 electrodes, the 4 fiducials
    kinds = warpnet.point_kinds()
    assert {n for (k, n), f in zip(kinds, g["fid"]) if f} == set(warpnet.GATE_FIDUCIALS)
    assert not (g["rtms"] & g["fid"]).any()   # Iz the electrode is not Iz the fiducial


def test_gated_points_weighted_in_the_point_loss(monkeypatch):
    torch.manual_seed(0)
    m = warpnet.build_model(NCFG)
    vg = SPEC.velocity_grid("cpu")
    batch = _samples(2)   # 2 points each: the first counts as gated here
    monkeypatch.setattr(warpnet, "_point_weight",
                        lambda n, w, dev: torch.tensor([w] + [1.0] * (n - 1)))
    plain = warpnet.TrainConfig(p_aug=0.0, t1_jitter=0.0)
    gated = warpnet.TrainConfig(p_aug=0.0, t1_jitter=0.0, w_gate=4.0)
    _, p0, u, _ = warpnet.losses_for(m, batch, SPEC, vg, plain, NCFG.steps, "cpu")
    _, p1, _, _ = warpnet.losses_for(m, batch, SPEC, vg, gated, NCFG.steps, "cpu")
    pe = torch.stack([warpnet.point_errors(u[b], vg, s["pts"], s["r_pts"]) for b, s in enumerate(batch)])
    assert p0["pts"] == pytest.approx(pe.mean().item(), rel=1e-5)
    want = (4 * pe[:, 0].sum() + pe[:, 1].sum()) / (2 * 4 + 2)
    assert p1["pts"] == pytest.approx(want.item(), rel=1e-5)
