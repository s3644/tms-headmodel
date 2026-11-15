#!/usr/bin/env bash
# WP1 + WP2 round 1, detached and resumable (PLAN.md sections 4-5):
#   wait for ct-headmodel's CPU queue (its timestamped ALL_DONE: the stale Sun Yat-sen reruns exist)
#   -> freeze data/manifests/round1.json -> Dataset601_MRHead -> wait for a free GPU slot (no other
#   nnU-Net process; ct-headmodel's round-2 rerun, if running, past its GPU steps: its log reached
#   "net route"), >= 40 GB RAM available and <= 10 charm jobs -> plan/preprocess -> train fold 0
#   (epochs from the pilot below, user rule 2026-10-28) -> predict the held-out cases. Augmentation workers by charm load: 6 above 6 charm
#   jobs, 8 with any, nnU-Net's default with none (RAM rule, PLAN.md section 5).
#
#   setsid nohup scripts/train_round1.sh > runs/logs/train_round1.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
CT=${HM_CT_ROOT:-$HOME/Projects/ct-headmodel}
ROOT=$PWD/data/nnunet
export nnUNet_raw=$ROOT/raw nnUNet_preprocessed=$ROOT/preprocessed nnUNet_results=$ROOT/results
export PYTHONPATH=$PWD/src
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
NN=$HOME/.local/share/mamba/envs/nnunet/bin
TR=${HM_NNUNET_TRAINER:-}            # set: skip the epoch pilot and train this trainer
PILOT=runs/round1_pilot
DS=Dataset601_MRHead
MAN=data/manifests/round1.json
log() { echo "$(date '+%F %T') $*"; }
charm_jobs() { pgrep -fc 'ct_headmodel.charm|simnibs.cli.charm|bin/charm ' || true; }
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
# ct-headmodel's round-2 rerun (net_round2.sh -> train_nnunet.sh, log train_nnunet_round2.log) is not
# running, or is past its predict step (CPU-only charm net route from there on)
ct_gpu_done() {
  ! pgrep -f 'scripts/(train_nnunet|net_round2)\.sh' >/dev/null \
    || grep -qsE '^[0-9-]+ [0-9:]+ net route' "$CT/runs/logs/train_nnunet_round2.log" "$CT/runs/logs/net_round2.log"
}

log "waiting for ct-headmodel's CPU queue (timestamped ALL_DONE)"
until grep -qE '^[0-9-]+ [0-9:]+ ALL_DONE' "$CT/runs/logs/queue_cpu.log" 2>/dev/null; do sleep 300; done

if [ ! -f "$MAN" ]; then
  log "freeze the manifest"
  mkdir -p data/manifests
  "$PY" -c "
import json; from pathlib import Path; from headmodel import dataset
m = dataset.manifest(Path('$CT'))
Path('$MAN').write_text(json.dumps(m, indent=1))
n = [c['split'] for c in m['cases'].values()]
print(n.count('train'), 'train,', n.count('test'), 'test,', len(m['skipped']), 'skipped')" || exit 1
fi

log "dataset"
"$PY" -c "
import json; from pathlib import Path; from headmodel import dataset
p = Path('$MAN'); m = dataset.build(json.loads(p.read_text()), Path('$nnUNet_raw'))
p.write_text(json.dumps(m, indent=1))" || exit 1

log "waiting for a free GPU slot, >= 40 GB RAM available and <= 10 charm jobs"
while pgrep -f 'nnUNetv2_(train|predict|plan_and_preprocess)' >/dev/null || ! ct_gpu_done \
   || [ "$(avail_gb)" -lt 40 ] || [ "$(charm_jobs)" -gt 10 ]; do sleep 120; done
# RAM rule: fewer augmentation workers while charm still runs
n=$(charm_jobs)
if [ "$n" -gt 6 ]; then export nnUNet_n_proc_DA=6; elif [ "$n" -gt 0 ]; then export nnUNet_n_proc_DA=8; fi
log "slot free: $(charm_jobs) charm jobs, nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-default}"

if [ ! -f "$nnUNet_preprocessed/$DS/nnUNetPlans.json" ]; then
  log "plan + preprocess"
  "$NN/nnUNetv2_plan_and_preprocess" -d 601 -c 3d_fullres -np 8 --verify_dataset_integrity || exit 1
fi

train_fold0() {  # <trainer>: fold 0, resumable
  local out="$nnUNet_results/$DS/${1}__nnUNetPlans__3d_fullres/fold_0" cont=()
  [ -f "$out/checkpoint_final.pth" ] && return 0
  log "train fold 0 ($1)"
  [ -f "$out/checkpoint_latest.pth" ] && cont=(--c)
  "$NN/nnUNetv2_train" 601 3d_fullres 0 -tr "$1" "${cont[@]}"
}

# Epoch pilot (user rule 2026-10-28): the 50- and 100-epoch schedules on fold 0, each scored on the
# first 5 cases of fold 0's validation split (training data; never the test set, which stays for the
# round-1 result). Rule, fixed before the runs (mean Dice over labels 1-9 and the 5 cases):
#   gain = Dice(100) - Dice(50);  gain < 0.005 -> 100 epochs (the 100-epoch pilot model is kept),
#   gain < 0.02 -> 250 epochs,    else -> 500 epochs.
pilot_dice() {  # <trainer> -> mean Dice on the 5 validation cases
  local tr=$1
  if [ ! -s "$PILOT/dice_$tr.json" ]; then
    # stdout of both goes to the log (stderr): only the Dice number is the function's output
    "$NN/nnUNetv2_predict" -i "$PILOT/val5" -o "$PILOT/pred_$tr" -d 601 -c 3d_fullres -tr "$tr" -f 0 >&2 || return 1
    "$PY" scripts/pilot_dice.py "$PILOT/pred_$tr" "$nnUNet_raw/$DS/labelsTr" "$nnUNet_raw/$DS/dataset.json" \
      "$PILOT/dice_$tr.json" >&2 || return 1
  fi
  "$PY" -c "import json; print(json.load(open('$PILOT/dice_$tr.json'))['mean'])"
}
if [ -z "$TR" ]; then
  mkdir -p "$PILOT"
  train_fold0 nnUNetTrainer_50epochs || exit 1
  if [ ! -d "$PILOT/val5" ]; then  # splits_final.json exists once fold 0 has started training
    mkdir -p "$PILOT/val5"
    "$PY" -c "
import json; s = json.load(open('$nnUNet_preprocessed/$DS/splits_final.json'))[0]['val']
print('\n'.join(sorted(s)[:5]))" > "$PILOT/val5.txt" || exit 1
    while read -r c; do ln -sf "$nnUNet_raw/$DS/imagesTr/${c}_0000.nii.gz" "$PILOT/val5/"; done < "$PILOT/val5.txt"
  fi
  d50=$(pilot_dice nnUNetTrainer_50epochs) || exit 1
  train_fold0 nnUNetTrainer_100epochs || exit 1
  d100=$(pilot_dice nnUNetTrainer_100epochs) || exit 1
  TR=$("$PY" -c "
g = $d100 - $d50
print('nnUNetTrainer_100epochs' if g < 0.005 else 'nnUNetTrainer_250epochs' if g < 0.02 else 'nnUNetTrainer_500epochs')")
  log "epoch pilot on $(tr '\n' ' ' < "$PILOT/val5.txt"): Dice 50 ep $d50, 100 ep $d100 -> $TR"
  echo "Dice 50 ep $d50 | 100 ep $d100 | gain $("$PY" -c "print(round($d100 - $d50, 4))") -> $TR" > "$PILOT/decision.txt"
fi
train_fold0 "$TR" || exit 1

log "predict held-out cases"
"$NN/nnUNetv2_predict" -i "$nnUNet_raw/$DS/imagesTs" -o "$ROOT/pred_ts_${TR#nnUNetTrainer_}" -d 601 -c 3d_fullres -tr "$TR" -f 0 || exit 1
log "ALL_DONE"
