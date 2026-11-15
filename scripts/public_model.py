#!/usr/bin/env python3
"""Build the public R5 model archive for Zenodo, local only.

    python3 scripts/public_model.py [RELEASE] [DEST]
        RELEASE  default /mnt/24tb/headmodel/releases/current (scripts/make_release.sh output)
        DEST     default /mnt/24tb/headmodel/public/r5-model  (+ DEST.zip)

Copies only the model files named in RELEASE.json (checked against their SHA-256), adds the model card, the
CC BY-NC 4.0 licence, a split list with public case IDs only (no local paths, no image hashes), and SHA256SUMS.
Fails if any file names an OASIS participant, holds a local path, or is not in RELEASE.json's list. Nothing is
uploaded: publishing needs the user's go.
"""
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REL = Path(sys.argv[1] if len(sys.argv) > 1 else "/mnt/24tb/headmodel/releases/current").resolve()
DEST = Path(sys.argv[2] if len(sys.argv) > 2 else "/mnt/24tb/headmodel/public/r5-model")
CC_BY_NC = REPO / "public/LICENSE-CC-BY-NC-4.0.txt"
OASIS = re.compile(rb"OAS[1-4]_?\d{4}")
LOCAL = re.compile(rb"/home/|/mnt/|jukrapope")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if DEST.exists():
        sys.exit(f"exists: {DEST}")
    rel = json.loads((REL / "RELEASE.json").read_text())
    tmp = DEST.with_name(DEST.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for f, h in rel["files"].items():
        if sha(REL / f) != h:
            sys.exit(f"checksum mismatch in the release: {f}")
        (tmp / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REL / f, tmp / f)
    pub = {k: rel[k] for k in ("segmenter", "warp", "env", "files")}
    pub["model"] = "R5"
    pub["rule"] = "runs/prereg/r5_e2e.json in the code repository (ADOPT R5)"
    (tmp / "RELEASE.json").write_text(json.dumps(pub, indent=1) + "\n")

    man = json.loads((REPO / "data/manifests/r5.json").read_text())["cases"]
    splits = [{"case": c, "dataset": v["dataset"], "split": v["split"]} for c, v in sorted(man.items())]
    (tmp / "splits.json").write_text(json.dumps(splits, indent=0) + "\n")
    shutil.copy(REPO / "public/MODEL_CARD.md", tmp / "MODEL_CARD.md")
    shutil.copy(CC_BY_NC, tmp / "LICENSE")

    problems = []
    for f in sorted(p for p in tmp.rglob("*") if p.is_file()):
        rel_name = str(f.relative_to(tmp))
        data = f.read_bytes()
        if OASIS.search(data):
            problems.append(f"{rel_name}: OASIS participant ID")
        if f.suffix in (".json", ".md", ".txt", "") and LOCAL.search(data):
            problems.append(f"{rel_name}: local path or user name")
        if rel_name not in rel["files"] and rel_name not in ("RELEASE.json", "splits.json", "MODEL_CARD.md", "LICENSE"):
            problems.append(f"{rel_name}: not a listed model file")
    if problems:
        print("\n".join(problems))
        sys.exit(f"audit failed; partial copy left at {tmp}")
    sums = [f"{sha(p)}  {p.relative_to(tmp)}" for p in sorted(tmp.rglob("*")) if p.is_file()]
    (tmp / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    tmp.rename(DEST)
    with zipfile.ZipFile(DEST.with_suffix(".zip"), "w", zipfile.ZIP_STORED) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(DEST))
    print(f"{DEST} ({len(sums)} files) and {DEST.with_suffix('.zip')}; audit clean")


if __name__ == "__main__":
    main()
