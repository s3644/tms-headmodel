#!/usr/bin/env bash
# WP3b round 3: epoch pilot, then the warp net's final training (user rule 2026-10-28: pilot every
# training on the first 5 validation cases to choose the epochs). Detached and resumable:
#   (setsid nohup scripts/train_warpnet_r3.sh > runs/logs/train_warpnet_r3.log 2>&1 < /dev/null &)
# Rules sealed beforehand: runs/prereg/wp3b_r3_epochs.json (which length), wp3b_r3_adopt.json (adoption,
# judged by scripts/eval_warpnet.py --rule). Changes vs task N: worst-points loss term (--w-tail 1) and
# the harmonic fill outside the training support (register's Params.net_fill, at inference). The model
# of a run is its final checkpoint (last.pt), not the best on validation. Shares the GPU with nnU-Net
# jobs (HM_WARPNET_SHARE_GPU=1; one machine only), pausing while less than 6 GB is free.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HM_WARPNET_SHARE_GPU=1
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
R3=runs/wp3b/warpnet_r3
mkdir -p "$R3"
log() { echo "$(date '+%F %T') $*"; }

# train <epochs>: runs/wp3b/warpnet_r3/e<epochs>, resumed until its epochs are done
train() {
  local n=$1 out=$R3/e$1
  mkdir -p "$out"
  while ! grep -qs '"epochs done"' "$out/status.json"; do
    nice -n 5 "$PY" scripts/train_warpnet.py train --out "$out" --epochs "$n" --max-hours 12 --val-every 20 \
      --w-tail 1 >> "$out/train.log" 2>&1 || { log "train $n failed (see $out/train.log)"; return 1; }
    grep -qs '"epochs done"' "$out/status.json" || sleep 300
  done
}
# score <epochs>: the pilot score of e<epochs>/last.pt on the first 5 validation cases (prints it)
score() {
  "$PY" scripts/train_warpnet.py score --ckpt "$R3/e$1/last.pt" --json "$R3/pilot_e$1.json" --n-val 5 \
    | tee -a "$R3/pilot.log" | sed -n 's/^SCORE //p'
}

"$PY" scripts/preregister.py verify wp3b_r3_epochs >/dev/null || { log "rule wp3b_r3_epochs not sealed"; exit 1; }
log "cache (new train cases only)"
nice -n 5 "$PY" scripts/train_warpnet.py cache --workers 4 || { log "cache failed"; exit 1; }
log "pilot: 200 and 400 epochs side by side"
train 200 & p1=$!
train 400 & p2=$!
wait $p1 || exit 1
wait $p2 || exit 1
s200=$(score 200) && s400=$(score 400) || { log "pilot scoring failed"; exit 1; }
read -r final gain < <("$PY" - "$s200" "$s400" <<'PYEOF'
import sys
s200, s400 = map(float, sys.argv[1:3])
gain = s200 - s400                      # mm, lower score is better
final = 400 if gain < 0.03 else 700 if gain < 0.10 else 1000
print(final, f"{gain:.4f}")
PYEOF
)
log "pilot scores: 200 epochs $s200, 400 epochs $s400, gain $gain mm -> $final epochs"
printf '%s\n' "rule wp3b_r3_epochs: score_200 $s200 score_400 $s400 gain $gain final_epochs $final" > "$R3/decision.txt"
[ "$final" = 400 ] || train "$final" || exit 1
ln -sfn "e$final" "$R3/final"
log "final model $R3/e$final/last.pt; validation evaluation under rule wp3b_r3_adopt"
nice -n 5 "$PY" scripts/eval_warpnet.py --ckpt "$R3/e$final/last.pt" --tag r3_last_b0 --out "$R3/eval_val_last_b0" \
  --svf-stages '[[2,4,150]]' --svf-class-weights '{"brain": 0}' --rule wp3b_r3_adopt || exit 1
log "END"
