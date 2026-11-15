"""Pre-register a decision rule before the runs it judges: seal it in its own file, never edited afterwards.

  seal DRAFT.json NAME   copies DRAFT.json to runs/prereg/NAME.json (refuses if NAME exists), makes it
                         read-only and appends "<UTC time> <sha256> NAME.json" to runs/prereg/ledger.txt;
                         prints the hash. Results go in a separate file (e.g. decision_result.json) that
                         names the rule's hash, never into the rule file.
  verify NAME            recomputes the hash and checks it against the ledger (exit 1 if it differs).

A rule that adopts a registration setting must state its no-folding requirement (key "no_folding", e.g.
"written-map min Jacobian > 0 (MNI2Conform_nonl and Conform2MNI_nonl) on every validation case"), or say
why it does not apply; seal refuses a draft without that key. runs/ is not in git: the orchestrator may
also commit the sealed file (it holds no data) before the runs start.

Usage:  python scripts/preregister.py seal /tmp/rule.json wp3b_round3
        python scripts/preregister.py verify wp3b_round3
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "runs/prereg"
REQUIRED = ("what", "rule", "no_folding")


def sha256(f: Path) -> str:
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def ledger(root: Path = PREREG) -> dict:
    out = {}
    f = root / "ledger.txt"
    if f.is_file():
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) == 3:
                out[parts[2]] = {"time": parts[0], "sha256": parts[1]}
    return out


def seal(draft: Path, name: str, root: Path = PREREG) -> str:
    rule = json.loads(Path(draft).read_text())
    missing = [k for k in REQUIRED if k not in rule]
    if missing:
        raise ValueError(f"{draft}: missing {missing} (a registration rule needs its no-folding requirement)")
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f"{name}.json"
    if dst.name in ledger(root):
        raise FileExistsError(f"{dst.name} is already in the ledger")
    data = Path(draft).read_bytes()
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)   # fails if it exists
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    h = sha256(dst)
    t = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(root / "ledger.txt", "a") as fh:
        fh.write(f"{t} {h} {dst.name}\n")
    return h


def verify(name: str, root: Path = PREREG) -> bool:
    dst = root / f"{name}.json"
    entry = ledger(root).get(dst.name)
    return entry is not None and dst.is_file() and sha256(dst) == entry["sha256"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("seal")
    a.add_argument("draft", type=Path)
    a.add_argument("name")
    b = sub.add_parser("verify")
    b.add_argument("name")
    args = ap.parse_args()
    if args.cmd == "seal":
        print(seal(args.draft, args.name))
    else:
        ok = verify(args.name)
        print("OK" if ok else "MISMATCH or not sealed")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
