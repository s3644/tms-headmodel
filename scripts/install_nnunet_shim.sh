#!/usr/bin/env bash
# Make head-model's R5 trainer (src/headmodel/nnunet_r5.py) visible to nnU-Net, which looks trainers up by class
# name inside its own package. Writes one guarded file into the nnunet env; idempotent. The env is shared with
# ct-headmodel: without head-model's src on PYTHONPATH the shim imports nothing, so their runs are unchanged.
set -eu
PY=${HM_NNUNET_PY:-$HOME/.local/share/mamba/envs/nnunet/bin/python}
dir=$("$PY" -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training/nnUNetTrainer/variants'))")
cat > "$dir/headmodel_r5_shim.py" <<'PYEOF'
# head-model R5 trainer shim (head-model scripts/install_nnunet_shim.sh, envs/README.md). Imports
# nnUNetTrainerHMThick only when head-model's src is on PYTHONPATH; otherwise defines nothing.
try:
    from headmodel.nnunet_r5 import nnUNetTrainerHMThick, nnUNetTrainerHMThick430  # noqa: F401
except ImportError:
    pass
PYEOF
echo "shim: $dir/headmodel_r5_shim.py"
