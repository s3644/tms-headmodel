#!/usr/bin/env bash
# R5 segmenter (PLAN.md 10.8), detached and resumable:
#   wait for the IXI charm queue (its ALL_DONE) -> data/manifests/r5.json (round 1 + the sealed r5_ixi_split)
#   -> Dataset602_MRHeadR5 -> wait for a free GPU (no other nnU-Net process), >= 40 GB RAM -> plan + preprocess
#   -> folds 0-4 with nnUNetTrainerHMThick (250 epochs, thick-slice augmentation). nnU-Net's own validation at the
#   end of each fold writes the out-of-fold labels of every training head (fold_N/validation), which the R5 warp
#   net trains on. Test heads (imagesTs) are not predicted here: that is the sealed scoring rule's job.
#
#   setsid nohup scripts/train_r5.sh > runs/logs/train_r5.log 2>&1 < /dev/null &
#   scripts/train_r5.sh --smoke   # minutes: 6 finished IXI heads, Dataset699, 1 epoch of 5 iterations, scratch dirs
set -u
cd "$(dirname "$0")/.."
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
NN=$HOME/.local/share/mamba/envs/nnunet/bin
export PYTHONPATH=$PWD/src
log() { echo "$(date '+%F %T') $*"; }
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
TR=nnUNetTrainerHMThick
scripts/install_nnunet_shim.sh >/dev/null || exit 1
"$PY" scripts/preregister.py verify r5_ixi_split >/dev/null || { log "r5_ixi_split not sealed or changed"; exit 1; }
"$PY" scripts/preregister.py verify r5_train_plan_amend1 >/dev/null || { log "r5_train_plan_amend1 not sealed or changed"; exit 1; }

if [ "${1:-}" = "--smoke" ]; then
  ROOT=${HM_R5_SMOKE_ROOT:?set HM_R5_SMOKE_ROOT to a scratch directory}
  ID=699; DS=Dataset699_R5Smoke; MAN=$ROOT/manifest.json; FOLDS="0"; export HM_R5_SMOKE=1
else
  ROOT=$PWD/data/nnunet
  ID=602; DS=Dataset602_MRHeadR5; MAN=data/manifests/r5.json; FOLDS="0 1 2 3 4"
  log "waiting for the IXI charm queue (ALL_DONE in runs/logs/queue_ixi.log)"
  until grep -qE '^[0-9-]+ [0-9:]+ ALL_DONE' runs/logs/queue_ixi.log 2>/dev/null; do sleep 300; done
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
split = json.loads(Path("runs/prereg/r5_ixi_split.json").read_text())
r1 = json.loads(Path("data/manifests/round1.json").read_text())
if smoke:   # 6 finished IXI training heads only, never test heads
    done = [c for c in split["train"] if dataset.charm_finished(dataset.IXI_ROOT / c / f"m2m_{c}")][:6]
    r1, split = {"cases": {}, "skipped": {}}, {"train": done, "val": [], "test": []}
m = dataset.manifest_r5(r1, split)
m["dataset"] = ds
out.write_text(json.dumps(m, indent=1))
n = [c["split"] for c in m["cases"].values()]
print(n.count("train"), "train,", n.count("test"), "test,", len(m["skipped"]), "skipped")
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
  "$NN/nnUNetv2_plan_and_preprocess" -d $ID -c 3d_fullres -np 8 --verify_dataset_integrity || exit 1
fi

for f in $FOLDS; do
  out="$nnUNet_results/$DS/${TR}__nnUNetPlans__3d_fullres/fold_$f"
  [ -f "$out/checkpoint_final.pth" ] && [ -f "$out/validation/summary.json" ] && { log "fold $f done"; continue; }
  cont=(); [ -f "$out/checkpoint_latest.pth" ] && cont=(--c)
  log "train fold $f ($TR)"
  "$NN/nnUNetv2_train" $ID 3d_fullres "$f" -tr "$TR" "${cont[@]}" || exit 1
done
log "ALL_DONE"
