#!/usr/bin/env bash
# A pinned copy of navprep for the clinic's locator (phase 1: MRI on the GPU): the committed code at HEAD plus
# the model files navprep loads, so production never runs the working tree or a model that is being retrained.
#
#   scripts/make_release.sh            -> $HM_RELEASES/<YYYYmmdd>-<sha7>/ (+ RELEASE.json with sha256 of every model file)
#   scripts/make_release.sh --current  ... and point $HM_RELEASES/current at it
#
# Layout = the repo's, so navprep.REPO resolves inside the release. The envs (nnunet, rtms-gpu) are not copied:
# the locator mounts them read-only; changing those envs means re-running the release smoke test (DEPLOY.md).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RELEASES=${HM_RELEASES:-/mnt/24tb/headmodel/releases}
cd "$REPO"
[ -z "$(git status --porcelain -- src scripts/navprep.py)" ] || { echo "uncommitted changes in src/ or scripts/navprep.py" >&2; exit 1; }
sha=$(git rev-parse --short=7 HEAD)
dest="$RELEASES/$(date +%Y%m%d)-$sha"
[ -e "$dest" ] && { echo "exists: $dest" >&2; exit 1; }
tmp="$dest.part"
mkdir -p "$tmp"
git archive HEAD src scripts/navprep.py | tar -x -C "$tmp"
# R5 (adopted by the sealed rule r5_e2e, 2026-11-05; user go for the release the same day): Dataset602 folds 0-4
# ensembled + the R5 warp net (its training support is inside the checkpoint) + the adopted label template.
MR=data/nnunet/results/Dataset602_MRHeadR5/nnUNetTrainerHMThick__nnUNetPlans__3d_fullres
NET=runs/r5/warp/e500/last.pt
models=("$MR/dataset.json" "$MR/plans.json")
for f in 0 1 2 3 4; do models+=("$MR/fold_$f/checkpoint_final.pth"); done
models+=("$NET" runs/wp3b/template_trainminusval)
for m in "${models[@]}"; do
  mkdir -p "$tmp/$(dirname "$m")"
  cp -a "$m" "$tmp/$m"
done
python3 - "$tmp" "$sha" "${models[@]}" <<'EOF'
import hashlib, json, sys, datetime
from pathlib import Path
root, sha, models = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
files = {}
for m in models:
    p = root / m
    for f in sorted([p] if p.is_file() else p.rglob("*")):
        if f.is_file():
            files[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
json.dump({"git": sha, "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "scope": "phase 1: MRI fast route (navprep --modality mr); CT not included",
           "segmenter": "Dataset602_MRHeadR5 folds 0-4 (R5)", "warp": "R5 warp net runs/r5/warp/e500 (500 epochs)",
           "rule": "runs/prereg/r5_e2e.json (ADOPT R5, docs/r5_e2e_report.md)",
           "env": {"HM_NAVPREP_MR_MODEL": models[0].rsplit("/", 1)[0], "HM_NAVPREP_MR_FOLDS": "0,1,2,3,4",
                   "HM_WARP_NET": next(m for m in models if m.endswith("last.pt"))},
           "files": files}, open(root / "RELEASE.json", "w"), indent=1)
EOF
chmod -R a-w "$tmp"
mv "$tmp" "$dest"
echo "$dest"
if [ "${1:-}" = "--current" ]; then
  ln -sfn "$(basename "$dest")" "$RELEASES/current.new" && mv -T "$RELEASES/current.new" "$RELEASES/current"
  echo "current -> $(basename "$dest")"
fi
