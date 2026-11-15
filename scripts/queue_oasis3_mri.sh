#!/usr/bin/env bash
# R6 OASIS-3 MRI charm queue (user go 2026-11-05; split sealed in runs/prereg/r6_oasis3_split.json, written by
# scripts/prep_oasis3_mri.py): charm on the 500 T1s in $ROOT/queue.txt (100 test first, then 400 train).
# CPU: only the 16 E-cores (CPUs 8-23, user 2026-11-05), at most 6 charm runs at 4 threads each (4 until 2026-11-05 16:45: E-cores ~45 % busy) (charm.ini copy with
# threads = 4), nice 10, every run through resource_guard. Output $ROOT/<id>/m2m_<id> (OASIS DUA: never in git, never
# off this machine). Resumable; a finished case is skipped. A run longer than 6 h is stopped (FAILED): a normal
# head takes ~50 min; case02 hung 38 h in samseg after a NaN (2026-11-09, dropped).
#
#   (setsid nohup scripts/queue_oasis3_mri.sh >> runs/logs/queue_oasis3_mri.log 2>&1 < /dev/null &)
set -u
cd "$(dirname "$0")/.."
ROOT=${HM_OASIS3_MRI_ROOT:-/mnt/24tb/headmodel/oasis3_mri}
CHARM=$HOME/.local/share/mamba/envs/rtms/bin/charm
CPUS=${HM_OASIS3_MRI_CPUS:-8-23}
MAX=${HM_OASIS3_MRI_MAX:-6}
THREADS=4
SETTINGS=$ROOT/charm_4threads.ini
MIN_AVAIL_GB=${HM_OASIS3_MRI_MIN_AVAIL_GB:-24}
GUARD="python3 $PWD/scripts/resource_guard.py"
CHARM_GB=${HM_CHARM_GB:-9}
log() { echo "$(date '+%F %T') $*"; }
mine() { pgrep -fc "^[^ ]*python[^ ]* [^ ]*/bin/charm OAS3[0-9]{4}_d[0-9]{4} " || true; }
avail_gb() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
free_slot() { [ "$(mine)" -lt "$MAX" ] && [ "$(avail_gb)" -ge "$MIN_AVAIL_GB" ] && $GUARD check --need "$CHARM_GB"; }
done_case() { ls "$ROOT/$1/m2m_$1"/*.msh >/dev/null 2>&1 && grep -qs "charm run finished" "$ROOT/$1/m2m_$1/charm_log.html"; }
t1_of() { awk -F'\t' -v id="$1" 'NR > 1 && $1 == id {print $7}' "$ROOT/selection.tsv"; }

[ -s "$ROOT/queue.txt" ] && [ -s "$SETTINGS" ] || { log "no queue.txt or $SETTINGS"; exit 1; }
python3 scripts/preregister.py verify r6_oasis3_split || { log "split rule hash differs; stop"; exit 1; }
ids=$(cat "$ROOT/queue.txt")
log "$(echo "$ids" | wc -l) charm runs on CPUs $CPUS, at most $MAX at a time, $THREADS threads each"
for id in $ids; do
  done_case "$id" && continue
  pgrep -f "^[^ ]*python[^ ]* [^ ]*/bin/charm $id " >/dev/null && { log "$id already running, skipped"; continue; }
  t1=$(t1_of "$id")
  [ -s "$t1" ] || { log "$id: input missing, skipped"; continue; }
  until free_slot; do sleep 60; done
  mkdir -p "$ROOT/$id"
  log "$id start"
  ( cd "$ROOT/$id" && OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS \
      NUMBA_NUM_THREADS=$THREADS ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$THREADS \
      $GUARD run --need "$CHARM_GB" --label "charm:$id" -- timeout 6h taskset -c "$CPUS" nice -n 10 \
      "$CHARM" "$id" "$t1" --forceqform --forcerun --usesettings "$SETTINGS" > charm.out 2>&1 \
      && log "$id done" || log "$id FAILED ($ROOT/$id/charm.out)" ) &
  sleep 60
done
wait
log "ALL_DONE"
