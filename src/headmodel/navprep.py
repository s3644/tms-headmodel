"""A T1 MRI -> a SimNIBS-format m2m folder in about a minute on an 8 GB GPU, without charm.

  <m2m>/T1.nii.gz                                    the conformed T1
  <m2m>/label_prep/tissue_labeling_upsampled.nii.gz  charm-valued labels on the same grid
  <m2m>/toMNI/{MNI2Conform,Conform2MNI}_nonl.nii.gz  the WP3b round-4 warp (register.adopted_setting)
  <m2m>/segmentation/coregistrationMatrices.mat      its affine part
  <m2m>/surfaces/{lh,rh}.pial.gii                    TopoFit cortex (display and cortex distance only)
  <m2m>/navprep.json                                 provisional flag, input checks, per-step timings, GPU peaks

Skin surface, EEG positions and fiducials are not written here (see README: SimNIBS's mni2subject_coords
on this folder gives positions in subject space).

Grid: the input is reoriented to LPS (lossless; most training images are LPS) and resampled to 1 mm
isotropic, the spacing both nnU-Net plans use, unless it already is 1 mm. A 5 mm clinic scan is
therefore interpolated once here; how much that costs in accuracy is PLAN 10.5 track B.

Several thick MRI series of one visit (e.g. a 5 mm sagittal and a 5 mm axial T1) are first combined into one
1 mm volume (`combine`: multistack.py, rigid registration + super-resolution reconstruction, ~15 s on the GPU).

Two stages, because the models live in two envs: `segment` (nnunet env: nnU-Net v2) and `warp`
(rtms-gpu: SimNIBS resources for the warp, BrainNet for the cortex). scripts/navprep.py runs both and
writes navprep.json. Segmenter: nnU-Net (the released R5 model: Dataset602, 5 folds, set by RELEASE.json).
Every result is marked provisional: research software, checked against charm, not against ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRAINER_DIR = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
MODELS = {   # modality -> (nnU-Net model folder, folds)
    "mr": (Path(os.environ.get("HM_NAVPREP_MR_MODEL", REPO / "data/nnunet/results/Dataset601_MRHead" / TRAINER_DIR)),
           tuple(int(f) for f in os.environ.get("HM_NAVPREP_MR_FOLDS", "0").split(","))),   # R5: "0,1,2,3,4"
}
MODALITIES = tuple(MODELS)
GRID_MM = 1.0
ORIENTATION = ("L", "P", "S")
LABELS = "label_prep/tissue_labeling_upsampled.nii.gz"   # charm's label image name
CONTEXT_GB = 0.6          # CUDA context, not counted by torch
MARGIN = 5                # voxels around the head box, as both training sets were cropped

# nnU-Net classes -> charm labels: dataset.CLASS_TO_CHARM.
# input checks (PLAN 10.2 item 3); warnings only, the clinician sees them with the provisional result
THICK_MM = 2.5


class InputError(ValueError):
    """The image cannot be used (wrong modality, not in HU, no head)."""


# ------------------------------------------------------------------------------------------ grid

def to_lps(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Lossless reorientation to the closest LPS axes (a flip/permutation of the voxel array)."""
    cur = nib.orientations.io_orientation(img.affine)
    want = nib.orientations.axcodes2ornt(ORIENTATION)
    tr = nib.orientations.ornt_transform(cur, want)
    if np.array_equal(tr, [[0, 1], [1, 1], [2, 1]]):
        return img
    data = nib.orientations.apply_orientation(np.asanyarray(img.dataobj), tr)
    aff = img.affine @ nib.orientations.inv_ornt_aff(tr, img.shape[:3])
    return nib.Nifti1Image(data, aff, img.header)


def iso_grid(img: nib.Nifti1Image, mm: float = GRID_MM) -> tuple[tuple[int, int, int], np.ndarray]:
    """(shape, affine) of a grid with mm voxels, the same axes and the same field of view as img."""
    R = img.affine[:3, :3]
    zooms = np.linalg.norm(R, axis=0)
    dirs = R / zooms
    extent = np.asarray(img.shape[:3]) * zooms
    shape = tuple(int(n) for n in np.maximum(np.ceil(extent / mm - 1e-6), 1))
    corner = img.affine[:3, 3] - dirs @ (zooms / 2)          # outer corner of voxel 0
    aff = np.eye(4)
    aff[:3, :3] = dirs * mm
    aff[:3, 3] = corner + dirs @ np.full(3, mm / 2)
    return shape, aff


def conform(img: nib.Nifti1Image, cval: float, mm: float = GRID_MM) -> nib.Nifti1Image:
    """LPS, then mm isotropic by trilinear interpolation (skipped when already mm within 1 %)."""
    from nibabel.processing import resample_from_to

    img = to_lps(img)
    if img.ndim == 4 and img.shape[3] == 1:
        img = nib.Nifti1Image(np.asanyarray(img.dataobj)[..., 0], img.affine, img.header)
    if img.ndim != 3:
        raise InputError(f"expected a 3-D image, got shape {img.shape}")
    if np.allclose(img.header.get_zooms()[:3], mm, rtol=0.01):
        out = nib.Nifti1Image(np.asanyarray(img.dataobj).astype(np.float32), img.affine)
    else:
        out = resample_from_to(nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine), iso_grid(img, mm),
                               order=1, cval=cval)
    return _with_codes(out)


def _with_codes(img: nib.Nifti1Image) -> nib.Nifti1Image:
    img.set_qform(img.affine, code=1)
    img.set_sform(img.affine, code=1)
    return img


def head_box(mask: np.ndarray, margin: int = MARGIN) -> tuple[slice, ...]:
    idx = np.argwhere(mask)
    lo, hi = np.maximum(idx.min(0) - margin, 0), np.minimum(idx.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def crop(img: nib.Nifti1Image, box: tuple[slice, ...]) -> nib.Nifti1Image:
    """Cropped image whose affine keeps world coordinates unchanged."""
    aff = img.affine.copy()
    aff[:3, 3] = nib.affines.apply_affine(img.affine, [s.start for s in box])
    return _with_codes(nib.Nifti1Image(np.asanyarray(img.dataobj)[box], aff))


# ------------------------------------------------------------------------------------------ head masks

def _largest(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def mr_head_mask(t1: np.ndarray) -> np.ndarray:
    """Rough head of a T1 for cropping only: above 10 % of the 99th percentile, largest component."""
    from scipy import ndimage as ndi

    p99 = float(np.percentile(t1[t1 > 0], 99)) if (t1 > 0).any() else 0.0
    if p99 <= 0:
        raise InputError("the T1 is empty")
    return ndi.binary_fill_holes(_largest(ndi.binary_opening(t1 > 0.1 * p99, iterations=2)))


# ------------------------------------------------------------------------------------------ labels

def keep_head(classes: np.ndarray, head: np.ndarray, modality: str) -> np.ndarray:
    """Predicted classes on the head only: the largest connected part of the prediction itself; the rough T1 mask
    is for cropping only, because bone and some scalp are dark on a T1 and the mask misses 5-30 % of the head."""
    out = classes.copy()
    out[~_largest(out > 0)] = 0
    return out


def mr_to_charm(classes: np.ndarray) -> np.ndarray:
    from .dataset import to_charm

    return to_charm(classes)


# ------------------------------------------------------------------------------------------ checks

def input_checks(src: nib.Nifti1Image, head: np.ndarray, modality: str) -> dict:
    """Warnings for the clinician (PLAN 10.2 item 3): slice thickness and a head cut by the image edge.
    `head` is on the conformed LPS grid, so its faces are named: L/R, P/A (front = low index on P),
    I/S."""
    zooms = [float(z) for z in src.header.get_zooms()[:3]]
    faces = {"right": head[0], "left": head[-1], "front": head[:, 0], "back": head[:, -1],
             "bottom": head[:, :, 0], "top": head[:, :, -1]}
    cut = {k: int(v.sum()) for k, v in faces.items() if v.sum() > 0}
    warnings = []
    if max(zooms) > THICK_MM:
        warnings.append(f"slices {max(zooms):.1f} mm thick (> {THICK_MM} mm): targets less certain")
    for k in ("top", "front", "left", "right", "back"):
        if k in cut:
            warnings.append(f"head touches the image edge ({k}): check coverage of nearby targets")
    return {"modality": modality, "voxel_mm": zooms, "shape": [int(n) for n in src.shape[:3]],
            "head_on_edge_voxels": cut, "warnings": warnings}


# ------------------------------------------------------------------------------------------ stage 0

def combine(images: list[Path], work: Path, cap_gb: float | None = None, device: str = "cuda") -> Path:
    """Stage 0 (MRI, two or more series of one visit): -> work/combined.nii.gz + work/combine.json."""
    import torch

    from headmodel import multistack

    if len(images) < 2:
        raise InputError("combine needs two or more series")
    work.mkdir(parents=True, exist_ok=True)
    _cap(torch, cap_gb)
    clock = Clock()
    out = work / "combined.nii.gz"
    try:
        info = multistack.combine(images, out, device)
    except ValueError as e:
        raise InputError(str(e)) from e
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    clock.lap("combine", gpu_peak_gb=_gpu_peak_gb(torch))
    (work / "combine.json").write_text(json.dumps({**info, "steps": clock.steps}, indent=1))
    return out


# ------------------------------------------------------------------------------------------ stage 1

class Clock:
    def __init__(self):
        self.steps: list[dict] = []
        self._t = time.perf_counter()

    def lap(self, name: str, **extra) -> None:
        now = time.perf_counter()
        self.steps.append({"step": name, "s": round(now - self._t, 3), **extra})
        self._t = now


def _gpu_peak_gb(torch) -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 2 ** 30 + CONTEXT_GB, 2)


def _cap(torch, cap_gb: float | None) -> None:
    """Hold this process to cap_gb of GPU memory (the clinic's 8 GB card, PLAN 10.4 step 1)."""
    if cap_gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, max(cap_gb - CONTEXT_GB, 0.5) / total))


def load_predictor(model: Path, folds, device: str = "cuda"):
    """An nnU-Net predictor of `model` (a trained model folder) with these folds, navprep's settings."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=True, device=torch.device(device),
                           verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(model), use_folds=tuple(folds), checkpoint_name="checkpoint_final.pth")
    pred.hm_model, pred.hm_folds = Path(model), tuple(folds)   # for segment.json
    return pred


def segment(image: Path, modality: str, work: Path, cap_gb: float | None = None, device: str = "cuda",
            predictor=None) -> dict:
    """Stage 1 (nnunet env): image -> work/t1.nii.gz + work/labels.nii.gz (charm labels) on the conformed
    grid, and work/segment.json (checks, timings). `predictor`: a load_predictor() to use instead of MODELS
    (R5's out-of-fold labels: one fold's model on its held-out heads, loaded once)."""
    import torch

    if modality not in MODELS:
        raise InputError(f"modality must be one of {MODALITIES}")
    work.mkdir(parents=True, exist_ok=True)
    _cap(torch, cap_gb)
    clock = Clock()
    src = nib.load(str(image))
    img = conform(src, cval=0.0)
    vol = img.get_fdata(dtype=np.float32)
    head = mr_head_mask(vol)
    checks = input_checks(src, head, modality)
    box = head_box(head)
    nn_in = work / "nn_in.nii.gz"
    nib.save(crop(nib.Nifti1Image(vol.astype(np.float32), img.affine), box), nn_in)
    clock.lap("conform_crop")

    model, folds = MODELS[modality]
    if predictor is None:
        pred = load_predictor(model, folds, device)
    else:
        pred, model, folds = predictor, predictor.hm_model, predictor.hm_folds
    clock.lap("model_load")
    rw = pred.plans_manager.image_reader_writer_class()
    data, props = rw.read_images([str(nn_in)])
    classes = pred.predict_single_npy_array(data, props, None, None, False)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    clock.lap("segment", folds=list(folds), gpu_peak_gb=_gpu_peak_gb(torch))
    rw.write_seg(np.asarray(classes, np.uint8), str(work / "nn_out.nii.gz"), props)
    cls_crop = np.asanyarray(nib.load(str(work / "nn_out.nii.gz")).dataobj).astype(np.uint8)
    full = np.zeros(vol.shape, np.uint8)
    full[box] = cls_crop
    full = keep_head(full, head, modality)
    labels = mr_to_charm(full)
    t1 = vol
    nib.save(_with_codes(nib.Nifti1Image(labels, img.affine)), work / "labels.nii.gz")
    nib.save(_with_codes(nib.Nifti1Image(t1.astype(np.float32), img.affine)), work / "t1.nii.gz")
    clock.lap("labels")
    out = {"checks": checks, "model": str(model), "folds": list(folds), "steps": clock.steps}
    (work / "segment.json").write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------------------ stage 2

def surface_gii(fs_surface: Path, gii: Path) -> None:
    """A FreeSurfer surface (TopoFit output, coordinates relative to c_ras) -> .gii in scanner mm,
    plain POINTSET + TRIANGLE (no volume geometry), which common GIfTI readers take as is."""
    from nibabel.gifti import GiftiDataArray, GiftiImage

    v, f, meta = nib.freesurfer.read_geometry(str(fs_surface), read_metadata=True)
    if "cras" in meta:
        v = v + np.asarray(meta["cras"], float)
    img = GiftiImage(darrays=[
        GiftiDataArray(np.asarray(v, np.float32), intent="NIFTI_INTENT_POINTSET", datatype="NIFTI_TYPE_FLOAT32"),
        GiftiDataArray(np.asarray(f, np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32")])
    gii.parent.mkdir(parents=True, exist_ok=True)
    tmp = gii.with_name(gii.name + ".part.gii")
    nib.save(img, tmp)
    os.replace(tmp, gii)


def warp(work: Path, m2m: Path, cap_gb: float | None = None, device: str = "cuda", cortex: bool = True) -> dict:
    """Stage 2 (rtms-gpu env): work/{t1,labels}.nii.gz -> the m2m folder (see the module docstring) and
    work/warp.json."""
    import shutil

    import torch

    from . import register as reg

    _cap(torch, cap_gb)
    clock = Clock()
    m2m.mkdir(parents=True, exist_ok=True)
    t1, labels = nib.load(str(work / "t1.nii.gz")), nib.load(str(work / "labels.nii.gz"))
    params, template_dir = reg.adopted_setting()
    clock.lap("warp_load")
    res = reg.register(t1, labels, None, device=device, params=params, template_dir=template_dir, write=False)
    clock.lap("warp", gpu_peak_gb=res.get("peak_mem_gb"), gpu_total_s=res.get("timings", {}).get("gpu_total"))
    # the toMNI maps (~7 s of gzip on the CPU) are written while the GPU builds the cortex
    writer = _Background(reg.write_m2m, m2m, *(res["transform"][k] for k in
                                               ("mni2conform", "atlas_affine", "conform2mni", "t1_affine", "A")))
    shutil.copyfile(work / "t1.nii.gz", m2m / "T1.nii.gz")
    (m2m / LABELS).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / "labels.nii.gz", m2m / LABELS)
    clock.lap("copy")
    if cortex:
        _topofit(work / "t1.nii.gz", work / "topofit", device)
        for h in ("lh", "rh"):
            surface_gii(work / "topofit" / f"{h}.pial", m2m / f"surfaces/{h}.pial.gii")
        clock.lap("cortex", gpu_peak_gb=_gpu_peak_gb(torch))
    clock.lap("write_maps", own_s=writer.join())   # own_s: the writer's own time, overlapped with the cortex
    out = {"steps": clock.steps}
    (work / "warp.json").write_text(json.dumps(out, indent=1))
    return out


class _Background:
    """fn(*args) in a thread; join() waits, re-raises its exception and returns its seconds."""

    def __init__(self, fn, *args):
        import threading

        self._err: BaseException | None = None
        self._s = 0.0

        def run():
            t = time.perf_counter()
            try:
                fn(*args)
            except BaseException as e:  # noqa: BLE001 - re-raised in join()
                self._err = e
            self._s = time.perf_counter() - t

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def join(self) -> float:
        self._t.join()
        if self._err is not None:
            raise self._err
        return round(self._s, 3)


def _topofit(t1: Path, out: Path, device: str) -> None:
    from argparse import Namespace

    import brainnet.helpers.topofit as topofit

    out.mkdir(parents=True, exist_ok=True)
    topofit.predict(Namespace(image=t1, out_dir=out, conform=True, device=device, mni_space="mni152",
                              contrast="synth", resolution="random", transform=None, mni_direction="mni2sub",
                              hemi="both"))


# ------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("combine")
    c.add_argument("images", type=Path, nargs="+")
    c.add_argument("--work", type=Path, required=True)
    s = sub.add_parser("segment")
    s.add_argument("image", type=Path)
    s.add_argument("--modality", choices=MODALITIES, required=True)
    s.add_argument("--work", type=Path, required=True)
    w = sub.add_parser("warp")
    w.add_argument("--work", type=Path, required=True)
    w.add_argument("--m2m", type=Path, required=True)
    w.add_argument("--no-cortex", action="store_true")
    for p in (c, s, w):
        p.add_argument("--cap-gb", type=float, default=None, help="GPU memory cap (the clinic card: 7.5)")
        p.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    try:
        if a.stage == "combine":
            combine(a.images, a.work, a.cap_gb, a.device)
        elif a.stage == "segment":
            segment(a.image, a.modality, a.work, a.cap_gb, a.device)
        else:
            warp(a.work, a.m2m, a.cap_gb, a.device, cortex=not a.no_cortex)
    except InputError as e:
        print(f"navprep: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
