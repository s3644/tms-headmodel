#!/usr/bin/env bash
# WP3b task A: build the warp-network cache (if needed), then train, detached and resumable.
#   (setsid nohup scripts/train_warpnet.sh > runs/wp3b/warpnet/train.log 2>&1 < /dev/null &)
# Training yields the GPU (checkpoint, clean exit) to any nnU-Net job and to head-model's round 1
# once its log says "slot free"; rerun this script later to resume from runs/wp3b/warpnet/last.pt
# (HM_WARPNET_IGNORE_ROUND1_LOG=1 once round 1 has finished). Budget: WARPNET_EPOCHS (900) epochs,
# WARPNET_HOURS (4) hours of training in total over all runs.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
log() { echo "$(date '+%F %T') $*"; }

log "cache"
nice -n 5 "$PY" scripts/train_warpnet.py cache --workers 4 || exit 1
log "train"
nice -n 5 "$PY" scripts/train_warpnet.py train --epochs "${WARPNET_EPOCHS:-900}" --max-hours "${WARPNET_HOURS:-4}" \
  --val-every "${WARPNET_VAL_EVERY:-10}" || exit 1
log "END $(cat runs/wp3b/warpnet/status.json 2>/dev/null | tr -d '\n ')"
