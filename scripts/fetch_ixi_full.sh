#!/usr/bin/env bash
# IXI full (brain-development.org/ixi-dataset, CC BY-SA 3.0): T1, T2, PD + demographics -> /mnt/24tb/datasets/mri-brain/ixi
# (PLAN 10.9: T2/PD of the same heads for the contrast-agnostic segmenter). DTI and MRA are not fetched:
# no diffusion (user, 2026-11-02) and no use for the head model. Resumable (wget -c); each tar is extracted
# into its own folder and every .nii.gz is gzip-tested. Log: <root>/fetch.log, ALL_DONE at the end.
set -u
ROOT=${IXI_FULL_ROOT:-/mnt/24tb/datasets/mri-brain/ixi}
URL=http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI
mkdir -p "$ROOT/tar"
log() { echo "$(date '+%F %T') $*" >> "$ROOT/fetch.log"; }
log "start"
wget -q -c -O "$ROOT/IXI.xls" "$URL/IXI.xls" || log "IXI.xls failed"
for m in T1 T2 PD; do
  for try in 1 2 3 4 5; do
    wget -q -c --tries=3 --timeout=60 -O "$ROOT/tar/IXI-$m.tar" "$URL/IXI-$m.tar" && break
    log "IXI-$m.tar attempt $try failed"; sleep 60
  done
  want=$(curl -sIL "$URL/IXI-$m.tar" | grep -i content-length | tail -1 | tr -dc 0-9)
  have=$(stat -c %s "$ROOT/tar/IXI-$m.tar")
  [ "$want" = "$have" ] || { log "IXI-$m.tar size $have != $want"; continue; }
  mkdir -p "$ROOT/$m"
  tar -xf "$ROOT/tar/IXI-$m.tar" -C "$ROOT/$m" || { log "IXI-$m.tar extract failed"; continue; }
  n=0; bad=0
  for f in "$ROOT/$m"/*.nii.gz; do n=$((n+1)); gzip -t "$f" 2>/dev/null || bad=$((bad+1)); done
  log "IXI-$m: $n files, $bad corrupt"
done
log "ALL_DONE"
