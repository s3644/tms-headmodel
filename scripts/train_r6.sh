#!/usr/bin/env bash
# R6 segmenter (docs/r6_plan_draft.md; rules r6_oasis3_split, r6_train_plan), detached and resumable:
#   data/manifests/r6.json (R5's frozen cases + the sealed OASIS-3 split) -> Dataset603_MRHeadR6 -> wait for a free
#   GPU (no other nnU-Net process), >= 40 GB RAM -> plan + preprocess (16 processes) -> folds 0-4 with
#   nnUNetTrainerHMThick430 (R5's trainer, 430 epochs = R5's exposure per head; 16 augmentation workers).
#   nnU-Net's validation at the end of each fold writes the out-of-fold labels the R6 warp net trains on.
#   Test heads (imagesTs: R5's test heads and the OASIS-3 test heads) are never predicted here.
#
#   (setsid nohup scripts/train_r6.sh > runs/logs/train_r6.log 2>&1 < /dev/null &)
#   HM_R6_SMOKE_ROOT=<scratch> scripts/train_r6.sh --smoke   # minutes: 6 OASIS-3 training heads, 1 epoch of 5 iterations
set -u
cd "$(dirname "$0")/.."
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
NN=$HOME/.local/share/mamba/envs/nnunet/bin
export PYTHONPATH=$PWD/src
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-16}
log() { echo "$(date '+%F %T') $*"; }
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
TR=nnUNetTrainerHMThick430
scripts/install_nnunet_shim.sh >/dev/null || exit 1
"$PY" scripts/preregister.py verify r6_oasis3_split >/dev/null || { log "r6_oasis3_split not sealed or changed"; exit 1; }

if [ "${1:-}" = "--smoke" ]; then
  ROOT=${HM_R6_SMOKE_ROOT:?set HM_R6_SMOKE_ROOT to a scratch directory}
  ID=698; DS=Dataset698_R6Smoke; MAN=$ROOT/manifest.json; FOLDS="0"; export HM_R5_SMOKE=1
else
  "$PY" scripts/preregister.py verify r6_train_plan >/dev/null || { log "r6_train_plan not sealed or changed"; exit 1; }
  ROOT=$PWD/data/nnunet
  ID=603; DS=Dataset603_MRHeadR6; MAN=data/manifests/r6.json; FOLDS="0 1 2 3 4"
fi
export nnUNet_raw=$ROOT/raw nnUNet_preprocessed=$ROOT/preprocessed nnUNet_results=$ROOT/results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$(dirname "$MAN")"

if [ ! -f "$MAN" ]; then
  log "freeze the manifest"
  "$PY" - "$MAN" "$DS" "${HM_R5_SMOKE:-0}" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
from headmodel import dataset
out, ds, smoke = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
split = json.loads(Path("runs/prereg/r6_oasis3_split.json").read_text())
r5 = json.loads(Path("data/manifests/r5.json").read_text())
if smoke:   # 6 finished OASIS-3 training heads only, never test heads; no R5 cases
    done = [c for c in split["train"] if dataset.charm_finished(dataset.OASIS3_MRI_ROOT / c / f"m2m_{c}")][:6]
    m = dataset.manifest_r6({"cases": {}, "skipped": {}}, {"train": done, "test": []}, n_val_per_stratum=0)
else:
    m = dataset.manifest_r6(r5, split)
m["dataset"] = ds
out.write_text(json.dumps(m, indent=1))
n = [c["split"] for c in m["cases"].values()]
v = [c for c in m["cases"].values() if c.get("r5_set") == "val" and c.get("stratum")]
print(n.count("train"), "train,", n.count("test"), "test,", len(v), "OASIS-3 warp val,", len(m["skipped"]), "skipped")
PYEOF
fi

log "dataset $DS"
"$PY" - "$MAN" "$nnUNet_raw" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
from headmodel import dataset
p = Path(sys.argv[1]); m = dataset.build(json.loads(p.read_text()), Path(sys.argv[2]))
p.write_text(json.dumps(m, indent=1))
PYEOF

log "waiting for a free GPU (no other nnU-Net process) and >= 40 GB RAM"
while pgrep -f 'nnUNetv2_(train|predict|plan_and_preprocess)' >/dev/null || [ "$(avail_gb)" -lt 40 ]; do sleep 120; done

if [ ! -f "$nnUNet_preprocessed/$DS/nnUNetPlans.json" ]; then
  log "plan + preprocess"
  "$NN/nnUNetv2_plan_and_preprocess" -d $ID -c 3d_fullres -np 16 --verify_dataset_integrity || exit 1
fi

for f in $FOLDS; do
  out="$nnUNet_results/$DS/${TR}__nnUNetPlans__3d_fullres/fold_$f"
  [ -f "$out/checkpoint_final.pth" ] && [ -f "$out/validation/summary.json" ] && { log "fold $f done"; continue; }
  cont=(); [ -f "$out/checkpoint_latest.pth" ] && cont=(--c)
  log "train fold $f ($TR)"
  "$NN/nnUNetv2_train" $ID 3d_fullres "$f" -tr "$TR" "${cont[@]}" || exit 1
done
log "ALL_DONE"
