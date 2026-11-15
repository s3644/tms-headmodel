# Environments

Two micromamba environments, exported from the machine the paper was run on:

- `rtms-gpu.yml`: SimNIBS 4.6, PyTorch 2.11 + CUDA 12.8, BrainNet/TopoFit, nibabel. SimNIBS and its helper
  packages are not on PyPI; the file installs them from their GitHub release wheels and git tags, and PyTorch
  from download.pytorch.org (cu128). Linux x86-64, Python 3.11.
- `nnunet.yml`: nnU-Net v2 2.8.1 (PyTorch 2.14 from PyPI).

The R5 trainer class `nnUNetTrainerHMThick` (src/headmodel/nnunet_r5.py) is made visible to nnU-Net by one guarded
shim file in its package, written by `scripts/install_nnunet_shim.sh`:
- nnU-Net finds trainers only by class name inside its own package.
- The shim imports the class only when this repository's src is on PYTHONPATH (navprep and the training scripts set
  it), so other uses of the env are unchanged.
- Re-run the script after reinstalling nnunetv2.
