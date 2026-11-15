#!/usr/bin/env bash
# R5 warp net and the R5 end-to-end score (PLAN 10.8), after the R5 segmenter. Detached and resumable:
#   (setsid nohup scripts/train_r5_warp.sh > runs/logs/train_r5_warp.log 2>&1 < /dev/null &)
# waits for train_r5.sh's ALL_DONE (5 folds of Dataset602), then, under the sealed rules r5_warp_train and r5_e2e:
#   out-of-fold labels (nnunet env) -> warp cache -> epoch pilot: 100 and 200 epochs, scored on the first 5
#   validation heads -> the final length -> support stored in the checkpoint -> runs/r5/warp/final
#   -> r5_e2e: the pilot set (a machinery check; stops on a failed run or a folded map) -> the IXI test set once
#   -> the 57 WP3 cases for comparison.
# The warp training holds ~30 GB of RAM (every head in memory) through resource_guard; one training at a time.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HM_WARPNET_SHARE_GPU=1
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
NNPY=$HOME/.local/share/mamba/envs/nnunet/bin/python
GUARD="python3 $PWD/scripts/resource_guard.py"
W=runs/r5/warp
mkdir -p "$W" runs/r5/e2e
log() { echo "$(date '+%F %T') $*"; }
for r in r5_warp_train r5_e2e; do
  "$PY" scripts/preregister.py verify $r >/dev/null || { log "rule $r not sealed or changed"; exit 1; }
done

log "waiting for the R5 segmenter (ALL_DONE in runs/logs/train_r5.log)"
until grep -qE '^[0-9-]+ [0-9:]+ ALL_DONE' runs/logs/train_r5.log 2>/dev/null; do sleep 300; done

log "out-of-fold labels"
$GUARD run --need 12 --label r5:oof -- "$NNPY" scripts/r5_warp.py oof --cap-gb 7.5 || { log "oof failed"; exit 1; }
log "warp cache"
$GUARD run --need 16 --label r5:cache -- nice -n 5 "$PY" scripts/r5_warp.py cache --workers 4 || { log "cache failed"; exit 1; }

# train <epochs>: runs/r5/warp/e<epochs>, resumed until its epochs are done
train() {
  local out=$W/e$1
  mkdir -p "$out"
  while ! grep -qs '"epochs done"' "$out/status.json"; do
    $GUARD run --need 34 --label "r5:warp-e$1" -- nice -n 5 "$PY" scripts/r5_warp.py train --epochs "$1" \
      >> "$out/train.log" 2>&1 || { log "train $1 failed (see $out/train.log)"; return 1; }
    grep -qs '"epochs done"' "$out/status.json" || sleep 300
  done
}
score() {
  "$PY" scripts/r5_warp.py score --ckpt "$W/e$1/last.pt" --json "$W/pilot_e$1.json" | tee -a "$W/pilot.log" \
    | sed -n 's/^SCORE //p'
}

if [ ! -e "$W/final" ]; then
  log "epoch pilot: 100, then 200 epochs (one at a time: RAM)"
  train 100 || exit 1
  train 200 || exit 1
  s100=$(score 100) && s200=$(score 200) || { log "pilot scoring failed"; exit 1; }
  read -r final gain < <(python3 - "$s100" "$s200" <<'PYEOF'
import sys
s100, s200 = map(float, sys.argv[1:3])
gain = s100 - s200                       # mm, lower score is better (rule r5_warp_train)
print(200 if gain < 0.03 else 350 if gain < 0.10 else 500, f"{gain:.4f}")
PYEOF
)
  log "pilot scores: 100 epochs $s100, 200 epochs $s200, gain $gain mm -> $final epochs"
  printf '%s\n' "rule r5_warp_train: score_100 $s100 score_200 $s200 gain $gain final_epochs $final" > "$W/decision.txt"
  [ "$final" = 200 ] || train "$final" || exit 1
  "$PY" scripts/r5_warp.py finalise --ckpt "$W/e$final/last.pt" || exit 1
  ln -sfn "e$final" "$W/final"
fi
log "final warp net $(readlink "$W/final")/last.pt"

log "r5_e2e pilot set (machinery check)"
"$PY" scripts/r5_e2e.py --set pilot > runs/r5/e2e/pilot.log 2>&1 || { log "pilot set failed (runs/r5/e2e/pilot.log)"; exit 1; }
"$PY" - <<'PYEOF' || { log "pilot set: a folded map; stopped before the test set (rule r5_e2e)"; exit 1; }
import json, sys
s = json.load(open("runs/r5/e2e/pilot/summary.json"))["summary"]
sys.exit(0 if all(a["cases_no_fold"] == a["n_cases"] for m in s.values() for a in m.values()) else 1)
PYEOF
log "r5_e2e test set (IXI, once)"
"$PY" scripts/r5_e2e.py --set test --rule r5_e2e > runs/r5/e2e/test.log 2>&1 || { log "test set failed"; exit 1; }
log "r5_e2e compare set (57 WP3 cases)"
"$PY" scripts/r5_e2e.py --set compare --rule r5_e2e > runs/r5/e2e/compare.log 2>&1 || { log "compare set failed"; exit 1; }
log "END (decision in runs/r5/e2e/test/summary.md; a new release needs the user's go)"
