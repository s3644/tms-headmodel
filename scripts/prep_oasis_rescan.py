"""TASK F, step 1: OASIS-1 reliability subset (scan-rescan) -> NIfTI T1s for charm.

Data: /mnt/24tb/datasets/oasis/oasis1/extracted/disc*/OAS1_XXXX_MR{1,2}/RAW/OAS1_XXXX_MRn_mpr-k_anon.{hdr,img}
(Analyze 7.5, 16-bit big-endian, 256 x 256 x 128, 1 x 1 x 1.25 mm, sagittal, face-masked). OASIS DUA:
non-commercial research; the images and everything derived stay on this machine and out of git.

Orientation. The Analyze header has orient = 2 ("sagittal unflipped") and no usable origin, which does
not fix a voxel -> RAS map. The convention used here is the one of the Aramis lab's OASIS-1 converter
(clinica, converters/oasis_to_bids, get_image_with_good_orientation: "OASIS-1 images have the same header
but sform is incorrect"), written there for the 1 mm SUBJ_111 average of the same native acquisition
space:   x = -k,  y = +i,  z = +j   (voxel i -> anterior, j -> superior, k -> patient left).
For RAW the k axis has 1.25 mm slices: RAS = P_DOC @ diag(1, 1, 1.25) @ (ijk - centre).

Verification on the data (every image):
  1. atlas check: each of the 48 signed axis permutations is scored by NCC against the MNI152 T1 after a
     moment alignment (foreground centroid and isotropic scale from the second moments). The documented
     orientation or its x-mirror must win, and both must beat every other candidate (S/I or A/P wrong) by
     a clear margin. Left/right cannot be decided from anatomy (the x-mirror differs only by the brain's
     small asymmetries), so the mirror's NCC difference is reported but not used.
  2. same-subject check: the 48 candidates of RAW mpr-k are scored against the session's own
     SUBJ_111 average with clinica's affine; there the x-mirror is a different brain shape, so this checks
     that RAW and SUBJ_111 share the axis order and handedness (i.e. that clinica's L/R convention
     applies to RAW as well). It cannot check the convention itself.

Left/right is therefore ASSUMED (clinica's convention, written for SUBJ_111; its source is not on this
machine). Supporting, not proof: after the 7-dof refinement against the (asymmetric) MNI152 T1 the
documented orientation beats its mirror in 19 of the 20 subjects (mean over the subject's 3 scans; sign
test p = 4e-5, lr_evidence), i.e. the images carry the population's brain asymmetries on the expected
side; with the moment alignment alone the preference is at chance (12/20). case03 prefers the mirror
in all three scans by ~0.002 NCC. The OASIS t4 files (RAW -> 711-2C atlas, det > 0) do not settle it:
their sign depends on the 4dfp axis conventions of the RAW conversion, which are not shipped (no .ifh).
A wrong L/R swaps which hemisphere the _l/_r targets land in; the scan-rescan pair errors hardly change.

Output: /mnt/24tb/headmodel/rescan/nifti/<subj>_<sess>_mpr<k>.nii.gz (int16, qform = sform, code 1) for
MR1 mpr-1, MR1 mpr-2 and MR2 mpr-1 of the 20 subjects with an MR2 session; runs/rescan/orientation.json
and runs/rescan/orientation.md (evidence), runs/rescan/cases.json (the charm ids, queue order).

Usage (rtms-gpu env):  PYTHONPATH=src python scripts/prep_oasis_rescan.py [--redo] [--report]
(--report: rewrite orientation.md and the summary from the existing orientation.json, no image work)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from headmodel import register  # noqa: E402

SRC = Path("/mnt/24tb/datasets/oasis/oasis1/extracted")
OUT_NII = Path("/mnt/24tb/headmodel/rescan/nifti")
OUT_RUN = REPO / "runs/rescan"
# voxel axis -> RAS axis: columns are the images of i, j, k (clinica's OASIS-1 convention)
P_DOC = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], float)
MIRROR_X = np.diag([-1.0, 1, 1])
SCANS = (("MR1", 1), ("MR1", 2), ("MR2", 1))   # A, B (same session), C (different day)
RES = 3.0   # mm, grid of the orientation scores


def signed_permutations() -> list[np.ndarray]:
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            P = np.zeros((3, 3))
            for col, row in enumerate(perm):
                P[row, col] = signs[col]
            out.append(P)
    return out


def raw_path(subj: str, sess: str, k: int) -> Path:
    hits = sorted(SRC.glob(f"disc*/{subj}_{sess}/RAW/{subj}_{sess}_mpr-{k}_anon.hdr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{subj}_{sess} mpr-{k}: {hits}")
    return hits[0]


def sbj111_path(subj: str, sess: str) -> Path:
    hits = sorted(SRC.glob(f"disc*/{subj}_{sess}/PROCESSED/MPRAGE/SUBJ_111/{subj}_{sess}_mpr_n*_anon_sbj_111.hdr"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{subj}_{sess} SUBJ_111: {hits}")
    return hits[0]


def rescan_subjects() -> list[str]:
    return sorted({p.name[:9] for p in SRC.glob("disc*/OAS1_*_MR2")})


def load_analyze(f: Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(f))
    d = np.asarray(img.dataobj).astype(np.float32)   # int16; scl_slope is nan (= unscaled)
    d = d.reshape(d.shape[:3])
    return d, np.asarray(img.header.get_zooms()[:3], float)


def affine_for(P: np.ndarray, shape, zooms) -> np.ndarray:
    """voxel -> RAS with the voxel-centre box centred on the origin."""
    A = np.eye(4)
    A[:3, :3] = P @ np.diag(zooms)
    A[:3, 3] = -A[:3, :3] @ ((np.asarray(shape[:3]) - 1) / 2)
    return A


def foreground(d: np.ndarray) -> np.ndarray:
    t = 0.15 * np.percentile(d[d > 0], 99) if (d > 0).any() else 0
    m = ndi.binary_opening(d > t, iterations=1)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return m


def moments(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, float]:
    ijk = np.argwhere(mask)[:: max(1, int(mask.sum() // 200000))]
    xyz = ijk @ affine[:3, :3].T + affine[:3, 3]
    c = xyz.mean(0)
    return c, float(np.linalg.det(np.cov((xyz - c).T)) ** (1 / 6))


class Reference:
    """A fixed image on a RES grid: values, mask, foreground centroid and scale."""

    def __init__(self, d: np.ndarray, affine: np.ndarray):
        zoom = np.sqrt((affine[:3, :3] ** 2).sum(0))
        f = zoom / RES
        self.d = ndi.zoom(ndi.gaussian_filter(d, RES / zoom / 2.355), f, order=1)
        self.affine = affine.copy()
        self.affine[:3, :3] = affine[:3, :3] @ np.diag(1 / f)
        self.affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ ((1 / f - 1) / 2)   # same field of view
        self.mask = foreground(self.d)
        self.c, self.s = moments(self.mask, self.affine)
        ijk = np.argwhere(ndi.binary_dilation(self.mask, iterations=2))
        self.xyz = ijk @ self.affine[:3, :3].T + self.affine[:3, 3]
        self.v = self.d[tuple(ijk.T)]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def score_candidates(d: np.ndarray, zooms, ref: Reference, cands, fixed_scale: bool = False) -> np.ndarray:
    """NCC of the moving image (voxel array d) under every candidate orientation, after aligning the
    foreground centroids (and, unless fixed_scale, the isotropic moment scale) to ref."""
    sm = ndi.gaussian_filter(d, RES / np.asarray(zooms) / 2.355)
    mask = foreground(sm)
    out = []
    for P in cands:
        A = affine_for(P, d.shape, zooms)
        c, s = moments(mask, A)
        k = 1.0 if fixed_scale else s / ref.s
        xyz = (ref.xyz - ref.c) * k + c                    # ref world -> moving world
        ijk = (xyz - A[:3, 3]) @ np.linalg.inv(A[:3, :3]).T
        v = ndi.map_coordinates(sm, ijk.T, order=1, cval=0.0)
        out.append(ncc(ref.v, v))
    return np.array(out)


def _rotvec(r):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(r).as_matrix()


def refine_candidates(d: np.ndarray, zooms, ref: Reference, cands, idx, iters: int = 2) -> dict:
    """Bounded 7-dof refinement (rotation, translation, isotropic scale; Powell on -NCC) of the moment alignment,
    for the candidates idx only: a candidate that is merely tilted by the moment alignment can catch up,
    a wrong axis assignment cannot."""
    from scipy.optimize import minimize

    sm = ndi.gaussian_filter(d, RES / np.asarray(zooms) / 2.355)
    mask = foreground(sm)
    out = {}
    for i in idx:
        A = affine_for(cands[i], d.shape, zooms)
        c, s = moments(mask, A)
        Ainv = np.linalg.inv(A[:3, :3])

        def cost(p):
            R = _rotvec(p[:3] / 10)   # 0.1 rad per unit: comparable steps for Powell
            xyz = ((ref.xyz - ref.c) @ R.T) * (s / ref.s) * np.exp(p[6] / 10) + c + p[3:6] * 5
            ijk = (xyz - A[:3, 3]) @ Ainv.T
            return -ncc(ref.v, ndi.map_coordinates(sm, ijk.T, order=1, cval=0.0))

        # bounded to a small correction (rotation <= ~20 deg per axis, shift <= 20 mm, scale +-10 %): a large
        # rotation would turn a wrong axis assignment into the right one and void the comparison
        bounds = [(-3.5, 3.5)] * 3 + [(-4, 4)] * 3 + [(-1, 1)]
        res = minimize(cost, np.zeros(7), method="Powell", bounds=bounds,
                       options={"maxiter": iters, "xtol": 1e-2, "ftol": 1e-4})
        out[int(i)] = float(-res.fun)
    return out


def same(P, Q) -> bool:
    return bool(np.allclose(P, Q))


def evidence(scores: np.ndarray, cands) -> dict:
    i_doc = next(i for i, P in enumerate(cands) if same(P, P_DOC))
    i_mir = next(i for i, P in enumerate(cands) if same(P, MIRROR_X @ P_DOC))
    others = [i for i in range(len(cands)) if i not in (i_doc, i_mir)]
    best_other = max(others, key=lambda i: scores[i])
    order = np.argsort(-scores)
    return {"ncc_doc": float(scores[i_doc]), "ncc_mirror": float(scores[i_mir]),
            "ncc_best_other": float(scores[best_other]), "best_other": cands[best_other].astype(int).tolist(),
            "margin_vs_other": float(min(scores[i_doc], scores[i_mir]) - scores[best_other]),
            "doc_minus_mirror": float(scores[i_doc] - scores[i_mir]),
            "winner_is_doc_or_mirror": bool(order[0] in (i_doc, i_mir) and order[1] in (i_doc, i_mir)),
            "winner_is_doc": bool(order[0] == i_doc)}


def write_nifti(d: np.ndarray, zooms, f: Path) -> np.ndarray:
    A = affine_for(P_DOC, d.shape, zooms)
    img = nib.Nifti1Image(np.round(d).astype(np.int16), A)
    img.set_qform(A, code=1)
    img.set_sform(A, code=1)
    img.header.set_xyzt_units("mm")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f".{f.name}.tmp.nii.gz")
    nib.save(img, str(tmp))
    tmp.replace(f)
    return A


_REF_MNI = None


def one_image(job: tuple) -> dict:
    """Orientation evidence (and the NIfTI) for one RAW image; runs in a worker process."""
    global _REF_MNI
    subj, sess, k, role, redo = job
    cands = signed_permutations()
    if _REF_MNI is None:
        mni = nib.load(str(register.mni_t1()))
        _REF_MNI = Reference(np.asarray(mni.dataobj, np.float32), mni.affine)
    cid = f"{subj}_{sess}_mpr{k}"
    d, zooms = load_analyze(raw_path(subj, sess, k))
    r = {"id": cid, "role": role, "shape": list(d.shape), "zooms": zooms.tolist()}
    sc = score_candidates(d, zooms, _REF_MNI, cands)
    r["mni"] = evidence(sc, cands)
    top = list(np.argsort(-sc)[:8])   # documented and mirror are among them (checked by evidence)
    ref_sc = np.full(len(cands), -np.inf)
    for i, v in refine_candidates(d, zooms, _REF_MNI, cands, top).items():
        ref_sc[i] = v
    r["mni_refined"] = evidence(ref_sc, cands)
    r["mni_refined"]["n_refined"] = len(top)
    sd, sz = load_analyze(sbj111_path(subj, sess))
    sref = Reference(sd, affine_for(P_DOC, sd.shape, sz))
    r["sbj111"] = evidence(score_candidates(d, zooms, sref, cands, fixed_scale=True), cands)
    f = OUT_NII / f"{cid}.nii.gz"
    if redo or not f.is_file():
        write_nifti(d, zooms, f)
    r["nifti"] = str(f)
    m, s, q = r["mni"], r["sbj111"], r["mni_refined"]
    print(f"{cid}: MNI refined doc {q['ncc_doc']:.3f} mirror {q['ncc_mirror']:.3f} other {q['ncc_best_other']:.3f} "
          f"(margin {q['margin_vs_other']:.3f}) | MNI doc {m['ncc_doc']:.3f} mirror {m['ncc_mirror']:.3f} other "
          f"{m['ncc_best_other']:.3f} (margin {m['margin_vs_other']:.3f}) | SUBJ_111 doc {s['ncc_doc']:.3f} mirror "
          f"{s['ncc_mirror']:.3f} other {s['ncc_best_other']:.3f}", flush=True)
    return r


def main():
    import multiprocessing as mp

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report", action="store_true", help="only rewrite orientation.md from orientation.json")
    args = ap.parse_args()
    if args.report:
        meta = json.loads((OUT_RUN / "orientation.json").read_text())
        meta["left_right"] = lr_evidence(meta["images"])
        (OUT_RUN / "orientation.json").write_text(json.dumps(meta, indent=1))
        return write_report(meta)
    subjects = rescan_subjects()
    jobs = [(subj, sess, k, role, args.redo) for subj in subjects for (sess, k), role in zip(SCANS, "ABC")]
    with mp.get_context("fork").Pool(min(4, args.workers)) as pool:
        rows = pool.map(one_image, jobs, chunksize=1)
    cases = {"A": [], "B": [], "C": []}
    for r in rows:
        cases[r["role"]].append(r["id"])
    summ = {}
    for ref in ("mni", "mni_refined", "sbj111"):
        e = [r[ref] for r in rows]
        summ[ref] = {"n": len(e),
                     "all_doc_or_mirror_top2": all(x["winner_is_doc_or_mirror"] for x in e),
                     "doc_wins": int(sum(x["winner_is_doc"] for x in e)),
                     "min_margin_vs_other": float(min(x["margin_vs_other"] for x in e)),
                     "median_margin_vs_other": float(np.median([x["margin_vs_other"] for x in e])),
                     "doc_minus_mirror_min": float(min(x["doc_minus_mirror"] for x in e)),
                     "doc_minus_mirror_median": float(np.median([x["doc_minus_mirror"] for x in e])),
                     "doc_minus_mirror_max": float(max(x["doc_minus_mirror"] for x in e))}
    OUT_RUN.mkdir(parents=True, exist_ok=True)
    meta = {"convention": "x=-k, y=+i, z=+j (clinica oasis_to_bids, OASIS-1); RAW k spacing 1.25 mm; centre at origin",
            "P_doc": P_DOC.astype(int).tolist(), "summary": summ, "images": rows}
    meta["left_right"] = lr_evidence(rows)
    (OUT_RUN / "orientation.json").write_text(json.dumps(meta, indent=1))
    order = cases["A"] + cases["C"]
    queue = [x for pair in zip(cases["A"], cases["C"]) for x in pair] + cases["B"]
    assert sorted(queue) == sorted(order + cases["B"])
    (OUT_RUN / "cases.json").write_text(json.dumps({"subjects": subjects, **cases, "queue": queue,
                                                    "nifti_dir": str(OUT_NII)}, indent=1))
    (OUT_NII.parent / "queue.txt").write_text("\n".join(queue) + "\n")
    write_report(meta)


def lr_evidence(rows: list[dict]) -> dict:
    """Left/right support from the MNI scores: per subject the mean doc - mirror NCC over its scans, the
    subjects preferring the documented orientation and a two-sided sign test (supporting evidence only)."""
    from scipy.stats import binomtest

    out = {"assumed": "left/right follows clinica's OASIS-1 convention (not checked independently)"}
    for ref in ("mni", "mni_refined"):
        by = {}
        for r in rows:
            by.setdefault(r["id"].split("_MR")[0], []).append(r[ref]["doc_minus_mirror"])
        subj = {k: float(np.mean(v)) for k, v in by.items()}
        wins = int(sum(v > 0 for v in subj.values()))
        out[ref] = {"subjects": len(subj), "doc_preferred": wins,
                    "sign_test_p": float(binomtest(wins, len(subj), 0.5).pvalue),
                    "mirror_preferred": sorted(k for k, v in subj.items() if v <= 0)}
    return out


def write_report(meta: dict) -> None:
    summ = meta["summary"]
    lr = meta.get("left_right") or lr_evidence(meta["images"])
    L = ["# OASIS-1 RAW orientation evidence", "", meta["convention"], "",
         "| reference | images | doc or mirror are the top 2 | doc wins | margin over the best other candidate "
         "(min / median NCC) | doc - mirror (min / median / max) |", "|---|---|---|---|---|---|"]
    for ref, name in (("mni", "MNI152 T1 (moment-aligned)"),
                      ("mni_refined", "MNI152 T1 (moment + 7-dof NCC refinement of the 8 best candidates)"), ("sbj111", "own SUBJ_111 average (clinica affine)")):
        s = summ[ref]
        L.append(f"| {name} | {s['n']} | {s['all_doc_or_mirror_top2']} | {s['doc_wins']}/{s['n']} | "
                 f"{s['min_margin_vs_other']:.3f} / {s['median_margin_vs_other']:.3f} | {s['doc_minus_mirror_min']:+.4f} / "
                 f"{s['doc_minus_mirror_median']:+.4f} / {s['doc_minus_mirror_max']:+.4f} |")
    L += ["", "Left/right is assumed (clinica's convention, written for SUBJ_111), not checked independently. "
          "Supporting evidence from the brain's population asymmetries (MNI152 is an asymmetric template): subjects "
          "whose mean doc - mirror NCC over their scans is > 0:"]
    for ref, name in (("mni", "moment-aligned"), ("mni_refined", "7-dof refined")):
        e = lr[ref]
        L.append(f"- {name}: {e['doc_preferred']}/{e['subjects']} (sign test p = {e['sign_test_p']:.1g}); mirror "
                 f"preferred: {', '.join(e['mirror_preferred']) or '-'}")
    L += ["", "The OASIS t4 files (RAW -> 711-2C) do not settle it: their determinant sign depends on the 4dfp axis "
          "conventions of the RAW conversion, which are not shipped. A wrong left/right swaps the hemisphere of the "
          "_l/_r targets; the scan-rescan pair errors hardly change."]
    (OUT_RUN / "orientation.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == "__main__":
    main()
