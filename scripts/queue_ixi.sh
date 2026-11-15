#!/usr/bin/env bash
# IXI charm queue (PLAN.md sections 4-5): full charm runs on the 507 IXI T1s, for generalisation
# tests (WP8), more training labels once the CC BY-SA question is settled, and the e-field solver's
# multi-head validation (the heads in scripts/ixi_first.txt go first).
#
# Starts after ct-headmodel's CPU queue is done. Runs at most 10 charm jobs on the machine, 6 while
# any nnU-Net training runs (RAM rule), and starts none below MIN_AVAIL_GB of available RAM; its
# charm runs at nice 10, so ct-headmodel's jobs and nnU-Net's loaders keep the CPU first.
# Starts no new case while /mnt/24tb/headmodel/oasis3/PRIORITY exists (the OASIS-3 pilot goes first,
# scripts/queue_oasis3.sh). Output: /mnt/24tb/headmodel/ixi/<id>/m2m_<id>, never in git. Resumable; a
# case whose charm still runs from an earlier queue is skipped.
#
#   setsid nohup scripts/queue_ixi.sh > runs/logs/queue_ixi.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
CT=${HM_CT_ROOT:-$HOME/Projects/ct-headmodel}
SRC=/mnt/24tb/datasets/mri-brain/ixi-keypoints/files
OUT=${HM_IXI_OUT:-/mnt/24tb/headmodel/ixi}
CHARM=$HOME/.local/share/mamba/envs/rtms/bin/charm
log() { echo "$(date '+%F %T') $*"; }
# charm processes only: python running .../bin/charm (not the resource_guard.py wrapper whose arguments contain it)
charm_jobs() { pgrep -fc 'ct_headmodel.charm|simnibs.cli.charm|^[^ ]*python[^ ]* [^ ]*/bin/charm ' || true; }
# RAM guard (scripts/resource_guard.py): a charm run is started only if it fits after every guarded job has
# grown to its declared peak (charm: CHARM_GB), with headroom for other services and the desktop
GUARD="python3 $PWD/scripts/resource_guard.py"
CHARM_GB=${HM_CHARM_GB:-9}
limit() { if pgrep -f nnUNetv2_train >/dev/null; then echo 6; else echo 10; fi; }
MIN_AVAIL_GB=${HM_IXI_MIN_AVAIL_GB:-24}
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
PRIORITY=${HM_IXI_PRIORITY_FILE:-/mnt/24tb/headmodel/oasis3/PRIORITY}
free_slot() { [ ! -e "$PRIORITY" ] && [ "$(charm_jobs)" -lt "$(limit)" ] && [ "$(avail_gb)" -ge "$MIN_AVAIL_GB" ] \
  && $GUARD check --need "$CHARM_GB"; }
done_case() { ls "$OUT/$1/m2m_$1"/*.msh >/dev/null 2>&1 && grep -qs "charm run finished" "$OUT/$1/m2m_$1/charm_log.html"; }

log "waiting for ct-headmodel's CPU queue"
until grep -qE '^[0-9-]+ [0-9:]+ ALL_DONE' "$CT/runs/logs/queue_cpu.log" 2>/dev/null; do sleep 300; done

mkdir -p "$OUT"
ids=$( { cat scripts/ixi_first.txt; ls "$SRC" | grep -E '^IXI[0-9]+-(Guys|HH)-[0-9]+-T1\.nii\.gz$' | sed 's/-T1\.nii\.gz$//' | sort; } | awk '!seen[$0]++')
log "$(echo "$ids" | wc -l) cases"
for id in $ids; do
  done_case "$id" && continue
  pgrep -f "^[^ ]*python[^ ]* [^ ]*/bin/charm $id " >/dev/null && { log "$id already running (earlier queue), skipped"; continue; }
  until free_slot; do sleep 60; done
  mkdir -p "$OUT/$id"
  ( cd "$OUT/$id" && $GUARD run --need "$CHARM_GB" --label "charm:$id" -- "$CHARM" "$id" "$SRC/$id-T1.nii.gz" --forceqform --forcerun > charm.out 2>&1 \
      && log "$id done" || log "$id FAILED ($OUT/$id/charm.out)" ) &
  sleep 30   # let the job show up in the process list before counting again
done
wait
log "ALL_DONE"
