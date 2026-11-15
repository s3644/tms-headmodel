"""WP1: the nnU-Net v2 training set Dataset601_MRHead (T1 -> charm's tissue labels).

Input: the T1 that charm segmented (m2m_ref/T1.nii.gz). Label: charm's mesh input,
label_prep/tissue_labeling_upsampled.nii.gz (0.5 mm, already updated from the cortical surfaces),
reduced to the T1 grid by a majority vote over each 2x2x2 block. Not final_tissues.nii.gz: that is
the mesh rasterised back onto the T1 grid, i.e. smoothed by one mesh round trip.

Cases come from finished charm runs in a local data folder (<root>/runs/<case>/m2m_ref). A case is used only if
charm finished the whole run (mesh present, "charm run finished" in its log), so the labels are
post-surface-update and no running job is still writing the folder. Held out (imagesTs): is_test()
(SynthRAD cases with a validate.json, Sun Yat-sen ids divisible by 4).
Stale Sun Yat-sen runs (T1 grid no longer matching the recentred data) are skipped.

Licence: SynthRAD is CC BY-NC 4.0 (Sun Yat-sen is CC BY 4.0), so everything trained on this set is research
only (PLAN.md, section 6). Each case's source and licence go into the manifest.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np

from .contract import TISSUES

DATASET = "Dataset601_MRHead"
SEG_LABELS = "label_prep/tissue_labeling_upsampled.nii.gz"
MARGIN = 5

# charm labels present after charm's surface update (4 = generic bone is never used: compact 7 and
# spongy 8 replace it) -> consecutive nnU-Net classes
CHARM_LABELS = [1, 2, 3, 5, 6, 7, 8, 9, 10]
CHARM_TO_CLASS = {c: i + 1 for i, c in enumerate(CHARM_LABELS)}
CLASS_TO_CHARM = {v: k for k, v in CHARM_TO_CLASS.items()}
CLASSES = {"background": 0, **{TISSUES[c]: i for c, i in CHARM_TO_CLASS.items()}}

SOURCES = {  # case-name pattern -> dataset, licence, data folder (relative to the data root)
    r"1B[ABC]\d+": ("SynthRAD2023 Task 1 brain (Zenodo 7260705)", "CC BY-NC 4.0", "data/synthrad"),
    r"(MEN|PIT|VS|CV)\d+": ("Sun Yat-sen paired head CT-MRI (Zenodo 17486320)", "CC BY 4.0", "data/sysu"),
}


def source(case: str) -> tuple[str, str, str] | None:
    for pat, src in SOURCES.items():
        if re.fullmatch(pat, case):
            return src
    return None


def is_test(case: str, runs: Path) -> bool:
    """The held-out rule of the round-1 dataset: SynthRAD cases with a validate.json, Sun Yat-sen ids divisible by 4."""
    if case.startswith("1B"):
        return (runs / case / "validate.json").exists()
    m = re.match(r"[A-Z]+(\d+)$", case)
    return bool(m) and int(m.group(1)) % 4 == 0


def charm_finished(m2m: Path) -> bool:
    log = m2m / "charm_log.html"
    return (any(m2m.glob("*.msh")) and (m2m / SEG_LABELS).is_file() and (m2m / "surfaces/lh.central.gii").is_file()
            and log.is_file() and "charm run finished" in log.read_text(errors="ignore"))


def stale_origin(case: str, m2m: Path, ct_root: Path) -> bool:
    """Sun Yat-sen runs made before the data were recentred: T1 grid != the recentred data."""
    src = source(case)
    if src is None or src[2] != "data/sysu":
        return False
    ref = ct_root / src[2] / case / "ct.nii.gz"
    return ref.is_file() and not np.allclose(nib.load(m2m / "T1.nii.gz").affine, nib.load(ref).affine, atol=1e-2)


def block_majority(up: np.ndarray, up_affine: np.ndarray, t1_shape: tuple, t1_affine: np.ndarray) -> np.ndarray:
    """Labels on the T1 grid from charm's 0.5 mm grid: T1 voxel i covers upsampled voxels 2i, 2i+1
    on each axis (charm's upsampled grid is exactly twice as fine, offset by a quarter voxel)."""
    expect = t1_affine.copy()
    expect[:3, :3] = t1_affine[:3, :3] / 2
    expect[:3, 3] = nib.affines.apply_affine(t1_affine, [-0.25] * 3)
    if tuple(up.shape) != tuple(2 * s for s in t1_shape[:3]) or not np.allclose(up_affine, expect, atol=1e-3):
        raise ValueError("upsampled labels are not on the 2x T1 grid")
    nx, ny, nz = t1_shape[:3]
    blocks = up.reshape(nx, 2, ny, 2, nz, 2).transpose(0, 2, 4, 1, 3, 5).reshape(nx, ny, nz, 8)
    n = int(blocks.max()) + 1
    counts = np.zeros((nx, ny, nz, n), np.uint8)
    for lab in range(n):
        counts[..., lab] = (blocks == lab).sum(-1)
    # ties go to the lower label (background, then WM, GM ...), the same for every case
    return counts.argmax(-1).astype(np.uint8)


def t1_grid_labels(up: nib.Nifti1Image, t1_shape: tuple, t1_affine: np.ndarray) -> np.ndarray:
    """charm's 0.5 mm labels on the T1 grid: block_majority when charm's grid is exactly 2x the T1's
    (SynthRAD, Sun Yat-sen), else the general majority vote of rescan.labels_majority (IXI: 0.94 x 0.94 x 1.2 mm)."""
    try:
        return block_majority(_vol(up), up.affine, t1_shape, t1_affine)
    except ValueError:
        from .rescan import labels_majority

        return labels_majority(_vol(up), up.affine, t1_shape[:3], t1_affine).astype(np.uint8)


def to_classes(charm: np.ndarray) -> np.ndarray:
    unknown = set(np.unique(charm).tolist()) - set(CHARM_TO_CLASS) - {0}
    if unknown:
        raise ValueError(f"unexpected charm labels {sorted(unknown)}")
    lut = np.zeros(max(CHARM_TO_CLASS) + 1, np.uint8)
    for c, i in CHARM_TO_CLASS.items():
        lut[c] = i
    return lut[charm]


def to_charm(classes: np.ndarray) -> np.ndarray:
    lut = np.zeros(max(CLASS_TO_CHARM) + 1, np.uint8)
    for i, c in CLASS_TO_CHARM.items():
        lut[i] = c
    return lut[classes]


def head_box(mask: np.ndarray, margin: int = MARGIN) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo, hi = np.maximum(idx.min(0) - margin, 0), np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def crop(affine: np.ndarray, data: np.ndarray, box: tuple[slice, ...]) -> nib.Nifti1Image:
    """Cropped image whose affine keeps world coordinates unchanged."""
    aff = affine.copy()
    aff[:3, 3] = nib.affines.apply_affine(affine, [s.start for s in box])
    img = nib.Nifti1Image(data[box], aff)
    img.set_qform(aff, code=1)
    img.set_sform(aff, code=1)
    return img


def _vol(img: nib.Nifti1Image) -> np.ndarray:
    """3-D data (charm writes some volumes as x*y*z*1)."""
    data = np.asarray(img.dataobj)
    return data.reshape(data.shape[:3]) if data.ndim == 4 and data.shape[3] == 1 else data


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(ct_root: Path) -> dict:
    """Every candidate case with its split, or the reason it is skipped. Reads headers only."""
    runs = ct_root / "runs"
    cases, skipped = {}, {}
    for m2m in sorted(runs.glob("*/m2m_ref")):
        case = m2m.parent.name
        src = source(case)
        if src is None:
            continue
        if not charm_finished(m2m):
            skipped[case] = "charm run not finished (no mesh, surfaces or log line)"
        elif stale_origin(case, m2m, ct_root):
            skipped[case] = "stale origin (made before the recentring fix)"
        else:
            cases[case] = {"split": "test" if is_test(case, runs) else "train", "dataset": src[0],
                           "licence": src[1], "m2m": str(m2m)}
    return {"dataset": DATASET, "label_source": SEG_LABELS, "classes": CLASSES, "cases": cases, "skipped": skipped}


IXI_ROOT = Path("/mnt/24tb/headmodel/ixi")
DATASET_R5 = "Dataset602_MRHeadR5"
IXI_SOURCE = ("IXI T1 (brain-development.org; subset Zenodo 7047668)", "CC BY-SA 3.0 (images)")


def manifest_r5(round1: dict, split: dict, ixi_root: Path = IXI_ROOT) -> dict:
    """R5 (PLAN 10.8): round 1's cases with their splits, plus the IXI heads of the sealed r5_ixi_split
    (train and val -> train, so nnU-Net's 5-fold cross-validation predicts every one of them out of fold; test ->
    test). IXI heads whose charm run is not finished are listed as skipped, never moved to another set."""
    cases = {c: dict(info) for c, info in round1["cases"].items()}
    skipped = dict(round1.get("skipped", {}))
    for split_name, ids in (("train", split["train"] + split["val"]), ("test", split["test"])):
        for c in ids:
            m2m = ixi_root / c / f"m2m_{c}"
            if not charm_finished(m2m):
                skipped[c] = "charm run not finished or failed"
                continue
            cases[c] = {"split": split_name, "dataset": IXI_SOURCE[0], "licence": IXI_SOURCE[1], "m2m": str(m2m),
                        "r5_set": "val" if c in split["val"] else split_name}
    return {"dataset": DATASET_R5, "label_source": SEG_LABELS, "classes": CLASSES, "cases": cases, "skipped": skipped}


OASIS3_MRI_ROOT = Path("/mnt/24tb/headmodel/oasis3_mri")
DATASET_R6 = "Dataset603_MRHeadR6"
OASIS3_SOURCE = ("OASIS-3 T1 (LaMontagne et al. 2019; NITRC, Data Use Agreement)",
                 "OASIS-3 DUA: non-commercial, no redistribution, never off this machine")
N_VAL_PER_STRATUM_R6 = 20


def manifest_r6(r5: dict, split: dict, root: Path = OASIS3_MRI_ROOT, n_val_per_stratum: int = N_VAL_PER_STRATUM_R6,
                selection: Path | None = None) -> dict:
    """R6 (docs/r6_plan_draft.md): R5's frozen cases with their splits (so R5's test heads stay held out), plus the
    OASIS-3 heads of the sealed r6_oasis3_split: train -> train, test -> test. Warp validation heads ("r5_set":
    "val", the key r5_warp.sets reads): per stratum the n_val_per_stratum training participants with the smallest
    sha1('r6-oasis3-val:' + participant), chosen from the sealed train list before looking at any charm run. A head
    whose charm run is not finished is listed as skipped, never moved to another set. `selection` (default
    root/selection.tsv) gives the stratum; None for a smoke manifest."""
    import csv
    import hashlib

    cases = {c: dict(info) for c, info in r5["cases"].items()}
    skipped = dict(r5.get("skipped", {}))
    stratum: dict[str, str] = {}
    sel = selection if selection is not None else root / "selection.tsv"
    if sel.is_file():
        stratum = {r["id"]: r["stratum"] for r in csv.DictReader(sel.open(), delimiter="\t")}
    val: set[str] = set()
    for s in sorted(set(stratum.values())):
        ids = [c for c in split["train"] if stratum.get(c) == s]
        key = lambda c: hashlib.sha1(f"r6-oasis3-val:{c.split('_')[0]}".encode()).hexdigest()  # noqa: E731
        val |= set(sorted(ids, key=key)[:n_val_per_stratum])
    for split_name, ids in (("train", split["train"]), ("test", split["test"])):
        for c in ids:
            m2m = root / c / f"m2m_{c}"
            if not charm_finished(m2m):
                skipped[c] = "charm run not finished or failed"
                continue
            cases[c] = {"split": split_name, "dataset": OASIS3_SOURCE[0], "licence": OASIS3_SOURCE[1], "m2m": str(m2m),
                        "stratum": stratum.get(c, "unknown"), "r5_set": "val" if c in val else split_name}
    return {"dataset": DATASET_R6, "label_source": SEG_LABELS, "classes": CLASSES, "cases": cases, "skipped": skipped}


def build(man: dict, out_root: Path) -> dict:
    """Write <out_root>/<man's dataset, default Dataset601_MRHead>/{imagesTr,labelsTr,imagesTs,labelsTs} +
    dataset.json from a manifest; adds sha256 of each written file to the manifest. Resumable (existing pairs are
    kept)."""
    ds = Path(out_root) / man.get("dataset", DATASET)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    for case, info in man["cases"].items():
        split = "Ts" if info["split"] == "test" else "Tr"
        img_out, lab_out = ds / f"images{split}" / f"{case}_0000.nii.gz", ds / f"labels{split}" / f"{case}.nii.gz"
        if not (img_out.exists() and lab_out.exists()):
            m2m = Path(info["m2m"])
            t1, up = nib.load(m2m / "T1.nii.gz"), nib.load(m2m / SEG_LABELS)
            t1_data = _vol(t1).astype(np.float32)
            charm = t1_grid_labels(up, t1_data.shape, t1.affine)
            box = head_box(charm > 0)
            nib.save(crop(t1.affine, t1_data, box), img_out)
            nib.save(crop(t1.affine, to_classes(charm), box), lab_out)
        info["image_sha256"], info["label_sha256"] = _sha256(img_out), _sha256(lab_out)
    n_train = sum(c["split"] == "train" for c in man["cases"].values())
    meta = {"channel_names": {"0": "T1"}, "labels": CLASSES, "numTraining": n_train, "file_ending": ".nii.gz",
            "description": __doc__.strip().splitlines()[0],
            "licence": "research only (CC BY-NC 4.0 sources; IXI CC BY-SA 3.0)"}
    (ds / "dataset.json").write_text(json.dumps(meta, indent=1))
    return man
