#!/usr/bin/env bash
# Rule p01_comparators: pilot (machinery), then the IXI test set once if every pilot job ran. Resumable.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/.local/share/mamba/envs/rtms-gpu/bin/python
export PYTHONPATH=$PWD/src
python3 scripts/preregister.py verify p01_comparators || { echo "rule hash differs; stop"; exit 1; }
$PY scripts/p01_comparators.py run --set pilot --jobs 6
bad=$(grep -l '"ok": false' runs/p01_comparators/pilot/*/*/*/eval.json 2>/dev/null | wc -l)
n=$(ls runs/p01_comparators/pilot/*/*/*/eval.json 2>/dev/null | wc -l)
echo "$(date '+%F %T') pilot: $n jobs, $bad failed"
[ "$n" -eq 20 ] && [ "$bad" -eq 0 ] || { echo "$(date '+%F %T') PILOT_FAILED: stop for a disclosed fix"; exit 1; }
$PY scripts/p01_comparators.py run --set test --jobs 6
echo "$(date '+%F %T') END"
