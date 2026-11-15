# tms-headmodel

A GPU route from a T1-weighted head MRI to a SimNIBS-format head model folder (`m2m_<id>`) for TMS
navigation. It takes a 1 mm research T1 or a routine thick-slice (5 mm) clinical T1, and runs in about
35 s on an 8 GB GPU (SimNIBS `charm` needs about an hour on the CPU).

> **Research software. Not a medical device, not validated for clinical use.** The paper reports how it
> was checked; results are against charm, which is a reference, not ground truth.

## What the route writes

| step | method | output in `m2m_<id>/` |
|---|---|---|
| combine series | several thick-slice series of one visit fused onto one grid | (internal) |
| conform | reoriented to LPS, resampled to 1 mm | `T1.nii.gz` |
| segmentation | nnU-Net v2 (5 folds), trained on charm's own labels | `label_prep/tissue_labeling_upsampled.nii.gz` (charm's tissue labels) |
| MNI warp | a learned registration network | `toMNI/{Conform2MNI,MNI2Conform}_nonl.nii.gz`, `segmentation/coregistrationMatrices.mat` |
| cortex | TopoFit (SimNIBS / BrainNet) on CUDA | `surfaces/{lh,rh}.pial.gii` |
| record | input checks, tier, per-step time and GPU peak | `navprep.json` |

The folder follows SimNIBS's layout, so SimNIBS functions that need only the warps work on it, for example
targets or EEG positions in subject space (below). It has **no** tetrahedral mesh, skin surface, EEG position
files or fiducials; run `charm` for electric-field work.

## Install

Linux x86-64, an NVIDIA GPU with 8 GB or more, and [micromamba](https://mamba.readthedocs.io). Two
environments, pinned in `envs/`:

    micromamba env create -n rtms-gpu -f envs/rtms-gpu.yml    # SimNIBS 4.6, PyTorch 2.11 (CUDA 12.8), BrainNet
    micromamba env create -n nnunet   -f envs/nnunet.yml      # nnU-Net v2 2.8.1
    scripts/install_nnunet_shim.sh                            # makes the R5 trainer class visible to nnU-Net

`navprep.py` finds the two environments under `$MAMBA_ROOT_PREFIX/envs` (default `~/.local/share/mamba/envs`).
With another layout (e.g. conda), point it at their interpreters:

    export HM_NAVPREP_SEGMENT_PY=/path/to/nnunet/bin/python
    export HM_NAVPREP_WARP_PY=/path/to/rtms-gpu/bin/python

The model files (R5: five nnU-Net folds, the warp network and the label template) are on Zenodo:
**doi:10.5281/zenodo.24215663** (all versions; CC BY-NC 4.0). The latest version only corrects the model card; the
model files are the same in every version. Unpack them into the repository root, keeping the code's GPL
`LICENSE` in place (the model licence is saved as `LICENSE-MODEL`), and check them:

    unzip r5-model.zip -x LICENSE
    unzip -p r5-model.zip LICENSE > LICENSE-MODEL
    grep -v ' LICENSE$' SHA256SUMS | sha256sum -c

The archive carries `RELEASE.json`, which sets the model paths for `navprep.py`.

## Use

    micromamba run -n rtms-gpu python scripts/navprep.py --modality mr T1.nii.gz m2m_subject
    micromamba run -n rtms-gpu python scripts/navprep.py --modality mr ax_T1_5mm.nii.gz sag_T1_5mm.nii.gz m2m_subject

`m2m_subject/navprep.json` records the input checks, the input tier (`1mm`, `thick`, `thick-combined`),
each step's time and the GPU peak. Exit code 3 means the input was not usable.

Positions in subject space from MNI coordinates, e.g. a DLPFC target and the EEG 10-10 cap that ships with
SimNIBS (unprojected: project them onto a skin surface if you need scalp points; SimNIBS warns that the
folder has no mesh, which does not matter here):

    import os, simnibs
    from simnibs.utils.csv_reader import read_csv_positions

    m2m = "m2m_subject"
    target = simnibs.mni2subject_coords([-38, 44, 26], m2m)
    cap = os.path.join(os.path.dirname(simnibs.__file__), "resources", "ElectrodeCaps_MNI",
                       "EEG10-10_UI_Jurak_2007.csv")
    _, xyz, _, names, _, _ = read_csv_positions(cap)
    eeg = dict(zip(names, simnibs.mni2subject_coords(xyz, m2m)))

## Tests

    micromamba run -n rtms-gpu python -m pytest -q

Tests that need local datasets or charm runs are skipped when those are absent.

## Reproducing the paper

- Every decision rule was sealed before the work it governs: `runs/prereg/*.json`, with the SHA-256 and UTC
  time of each in `runs/prereg/ledger.txt` (`scripts/preregister.py`). `REDACTIONS.md` lists what differs from
  the private originals.
- End-to-end scoring: `scripts/r5_e2e.py`, `scripts/m1_mri_e2e.py`; thick-slice simulation:
  `scripts/sim_series.py`; comparators: `scripts/p01_comparators.py`; worst points: `scripts/worst_points.py`.
  The rows "on our skin" used the closed skin surface of our navigation application, which is not part of
  this repository (`HM_LOCATOR_BACKEND` in `scripts/eval_register.py`).
- Training: `scripts/train_r5.sh`, `scripts/train_r5_warp.sh`. The scripts use our local data layout
  (paths at the top of each script); the datasets must be obtained from their providers.

## Data

No images are distributed here. Training and test data:

- IXI (CC BY-SA 3.0), via the keypoint-annotated release of Biondi et al.;
- SynthRAD2023 brain MRI (CC BY-NC 4.0) and a paired head CT-MRI set from Sun Yat-sen University (CC BY 4.0);
- OASIS-1 and OASIS-3 (used under their data use agreements; test only for the released model).

## Licence

- Code: GNU General Public License v3.0 (`LICENSE`); it uses SimNIBS, which is GPL-3.0.
- Model files (Zenodo): CC BY-NC 4.0, because part of the training data allows only non-commercial use.

## Citation

See `CITATION.cff`. Preprint: Jitpimolmard J. TMS targets in about a minute from routine clinical MRI. Zenodo, 2026. doi:10.5281/zenodo.24215659 (all versions).
