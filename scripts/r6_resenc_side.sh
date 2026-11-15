#!/usr/bin/env bash
# R6 side experiment (docs/r6_plan_draft.md; reported only, gates nothing): nnU-Net's residual-encoder preset M
# (Isensee et al. 2024, "nnU-Net Revisited") on Dataset603, fold 0 only, same trainer (nnUNetTrainerHMThick430),
# compared with the R6 default fold 0 on nnU-Net's own fold-0 validation (training heads only; no test head is
# read). Runs after the R6 chain so the GPU is free; the result (per-class Dice, both models) goes to
# runs/r6/resenc_side/summary.md and informs the next round's design, not R6's decision.
#   (setsid nohup scripts/r6_resenc_side.sh > runs/logs/r6_resenc_side.log 2>&1 < /dev/null &)
set -u
cd "$(dirname "$0")/.."
NN=$HOME/.local/share/mamba/envs/nnunet/bin
export PYTHONPATH=$PWD/src nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-16}
export nnUNet_raw=$PWD/data/nnunet/raw nnUNet_preprocessed=$PWD/data/nnunet/preprocessed nnUNet_results=$PWD/data/nnunet/results
DS=Dataset603_MRHeadR6; TR=nnUNetTrainerHMThick430; PL=nnUNetResEncUNetMPlans
OUT=runs/r6/resenc_side; mkdir -p "$OUT"
log() { echo "$(date '+%F %T') $*"; }
scripts/install_nnunet_shim.sh >/dev/null || exit 1

log "waiting for the R6 chain (END in runs/logs/train_r6_warp.log)"
until grep -qE '^[0-9-]+ [0-9:]+ END' runs/logs/train_r6_warp.log 2>/dev/null; do sleep 600; done
while pgrep -f 'nnUNetv2_(train|predict|plan_and_preprocess)' >/dev/null; do sleep 120; done

if [ ! -f "$nnUNet_preprocessed/$DS/$PL.json" ]; then
  log "plan (ResEncM) + preprocess"
  "$NN/nnUNetv2_plan_and_preprocess" -d 603 -c 3d_fullres -pl nnUNetPlannerResEncM -np 16 || exit 1
fi
out="$nnUNet_results/$DS/${TR}__${PL}__3d_fullres/fold_0"
if ! { [ -f "$out/checkpoint_final.pth" ] && [ -f "$out/validation/summary.json" ]; }; then
  cont=(); [ -f "$out/checkpoint_latest.pth" ] && cont=(--c)
  log "train fold 0 ($TR, $PL)"
  "$NN/nnUNetv2_train" 603 3d_fullres 0 -tr "$TR" -p "$PL" "${cont[@]}" || exit 1
fi
python3 - "$nnUNet_results/$DS/${TR}__nnUNetPlans__3d_fullres/fold_0/validation/summary.json" "$out/validation/summary.json" "$OUT/summary.md" <<'EOF'
import json, sys
a, b, out = sys.argv[1:]
names = {1: "WM", 2: "GM", 3: "CSF", 4: "Scalp", 5: "Eyes", 6: "CompactBone", 7: "SpongyBone", 8: "Blood", 9: "Muscle"}
da, db = json.load(open(a))["mean"], json.load(open(b))["mean"]
lines = ["# ResEncM side experiment (Dataset603 fold 0, nnU-Net's fold-0 validation; reported only)", "",
         "| class | default Dice | ResEncM Dice | diff |", "|---|---|---|---|"]
for k, n in names.items():
    x, y = da[str(k)]["Dice"], db[str(k)]["Dice"]
    lines.append(f"| {n} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
mx = sum(da[str(k)]["Dice"] for k in names) / 9; my = sum(db[str(k)]["Dice"] for k in names) / 9
lines += ["", f"mean foreground Dice: default {mx:.4f}, ResEncM {my:.4f} (diff {my - mx:+.4f})"]
open(out, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
EOF
log "SIDE_DONE ($OUT/summary.md)"
