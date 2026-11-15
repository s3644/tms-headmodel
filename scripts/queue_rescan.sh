#!/usr/bin/env bash
# TASK F charm scan-rescan queue: charm on the 60 OASIS-1 reliability T1s written by
# scripts/prep_oasis_rescan.py (/mnt/24tb/headmodel/rescan/nifti), in the order of
# /mnt/24tb/headmodel/rescan/queue.txt: the 20 MR1 mpr-1 + MR2 mpr-1 pairs first, then the 20 MR1 mpr-2
# repeats. Output /mnt/24tb/headmodel/rescan/<id>/m2m_<id> (OASIS DUA: never in git). Resumable.
#
# Starts a job only while the machine runs fewer charm jobs than limit() and >= MIN_AVAIL_GB of RAM is
# available. limit(): 10 (about all 24 cores; ~2 cores and 7-10 GB each) while no nnUNetv2_train runs
# and round 1 is not next in line (ct-headmodel's train_nnunet / net_round2 still running, or round 1
# trained); otherwise 6: nnU-Net training needs CPU and RAM for its loaders, and round 1 starts only at
# <= 6 charm jobs. The IXI queue keeps its own limit (6, 8 after round 1) and shares the slots.
# Both queues count and then start, so they can overshoot the limit by one when they start at the same
# moment; this queue re-counts after a random 1-20 s delay to make that unlikely (it cannot exclude it
# without a lock the IXI queue would also take).
#
#   (setsid nohup scripts/queue_rescan.sh >> runs/logs/queue_rescan.log 2>&1 < /dev/null &)
set -u
cd "$(dirname "$0")/.."
ROOT=${HM_RESCAN_ROOT:-/mnt/24tb/headmodel/rescan}
SRC=$ROOT/nifti
CHARM=$HOME/.local/share/mamba/envs/rtms/bin/charm
MAX=${HM_RESCAN_MAX_CHARM:-10}
MIN_AVAIL_GB=${HM_RESCAN_MIN_AVAIL_GB:-24}
log() { echo "$(date '+%F %T') $*"; }
charm_jobs() { pgrep -fc 'ct_headmodel.charm|simnibs.cli.charm|bin/charm ' || true; }
# round 1 past training: the training script's own timestamped lines (as scripts/queue_ixi.sh)
trained() { grep -qE '^[0-9-]+ [0-9:]+ (predict held-out cases|ALL_DONE)$' runs/logs/train_round1.log 2>/dev/null; }
limit() {
  local l=6
  if ! pgrep -f nnUNetv2_train >/dev/null \
     && { trained || pgrep -f 'scripts/(train_nnunet|net_round2)\.sh' >/dev/null; }; then l=10; fi
  [ "$l" -gt "$MAX" ] && l=$MAX
  echo "$l"
}
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
free_slot() { [ "$(charm_jobs)" -lt "$(limit)" ] && [ "$(avail_gb)" -ge "$MIN_AVAIL_GB" ] \
  && sleep $((RANDOM % 20 + 1)) && [ "$(charm_jobs)" -lt "$(limit)" ]; }
done_case() { ls "$ROOT/$1/m2m_$1"/*.msh >/dev/null 2>&1 && grep -qs "charm run finished" "$ROOT/$1/m2m_$1/charm_log.html"; }

[ -s "$ROOT/queue.txt" ] || { log "no $ROOT/queue.txt (run scripts/prep_oasis_rescan.py)"; exit 1; }
ids=$(cat "$ROOT/queue.txt")
log "$(echo "$ids" | wc -l) cases, at most limit() charm jobs on the machine (now $(limit), cap $MAX)"
for id in $ids; do
  done_case "$id" && continue
  pgrep -f "bin/charm $id " >/dev/null && { log "$id already running (earlier queue), skipped"; continue; }
  [ -s "$SRC/$id.nii.gz" ] || { log "$id: no $SRC/$id.nii.gz, skipped"; continue; }
  n=$(charm_jobs)
  [ "$n" -ge "$(limit)" ] && log "waiting ($n charm jobs, limit $(limit)) before $id"
  until free_slot; do sleep 60; done
  mkdir -p "$ROOT/$id"
  log "$id start"
  ( cd "$ROOT/$id" && "$CHARM" "$id" "$SRC/$id.nii.gz" --forceqform --forcerun > charm.out 2>&1 \
      && log "$id done" || log "$id FAILED ($ROOT/$id/charm.out)" ) &
  sleep 60   # let the job show up in the process list before counting again
done
wait
log "ALL_DONE"
