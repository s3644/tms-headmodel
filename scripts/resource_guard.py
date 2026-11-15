#!/usr/bin/env python3
"""Machine-wide RAM guard for batch jobs (charm, SimNIBS FEM, training): start a job only when it fits.

A plain "MemAvailable >= X" check at launch is not enough: charm and SimNIBS keep growing for minutes after they
start, so several jobs launched together can still exhaust RAM (the OOM killer struck on 2026-10-29 20:05). Here
every job declares its expected peak (--need GB) and is recorded in a shared ledger. A job may start only if

    MemAvailable - sum over running ledger jobs of max(0, need - current RSS of its process tree)  >=  need + HEADROOM

i.e. the RAM still free after every running job has grown to its declared peak. The check and the launch happen
under one lock, so two queues cannot both take the last room. Jobs run with oom_score_adj +500 and nice 10, so if RAM
still runs out the kernel kills a batch job before other services or the desktop.

  resource_guard.py run --need 12 --label charm:IXI002 -- charm IXI002 T1.nii.gz   # waits, then runs
  resource_guard.py check --need 12        # exit 0 if a 12 GB job fits now (for queue slot checks)
  resource_guard.py status                 # the ledger and the effective free RAM

Standard library only (any python3). Ledger: /dev/shm/rtms_ram_ledger.json (cleared at reboot).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LEDGER = Path(os.environ.get("RTMS_RAM_LEDGER", "/dev/shm/rtms_ram_ledger.json"))
HEADROOM_GB = float(os.environ.get("RTMS_RAM_HEADROOM_GB", "16"))   # for the desktop, other services, page cache
POLL_S = 30


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def _children() -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(ppid, []).append(int(d.name))
    return kids


def tree_rss_gb(pid: int, kids: dict[int, list[int]] | None = None) -> float:
    """Resident memory of a process and all its descendants."""
    kids = _children() if kids is None else kids
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        stack += kids.get(p, [])
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except OSError:
            pass
    return total / 2 ** 20


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def locked():
    LEDGER.touch(exist_ok=True)
    with open(LEDGER, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            raw = fh.read().strip()
            jobs = [j for j in (json.loads(raw) if raw else []) if alive(j["pid"])]
            yield jobs
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(jobs))
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def effective_free_gb(jobs: list[dict]) -> float:
    kids = _children()
    pending = sum(max(0.0, j["need_gb"] - tree_rss_gb(j["pid"], kids)) for j in jobs)
    return mem_available_gb() - pending


def fits(jobs: list[dict], need: float) -> bool:
    return effective_free_gb(jobs) >= need + HEADROOM_GB


def _low_priority():
    try:
        Path("/proc/self/oom_score_adj").write_text("500")
    except OSError:
        pass
    os.nice(10)


def run(need: float, label: str, cmd: list[str], wait: bool = True) -> int:
    said = False
    while True:
        with locked() as jobs:
            if fits(jobs, need):
                proc = subprocess.Popen(cmd, preexec_fn=_low_priority)
                jobs.append({"pid": proc.pid, "need_gb": need, "label": label, "start": time.time()})
                break
            free = effective_free_gb(jobs)
        if not wait:
            print(f"resource_guard: {label} ({need:g} GB) does not fit ({free:.0f} GB effectively free)", file=sys.stderr)
            return 75
        if not said:
            print(f"resource_guard: {label} waits for RAM ({need:g} GB + {HEADROOM_GB:g} headroom; "
                  f"{free:.0f} GB effectively free)", file=sys.stderr, flush=True)
            said = True
        time.sleep(POLL_S)
    rc = proc.wait()
    with locked() as jobs:
        jobs[:] = [j for j in jobs if j["pid"] != proc.pid]
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--need", type=float, required=True, help="expected peak RAM of the job, GB")
    r.add_argument("--label", default="job")
    r.add_argument("--no-wait", action="store_true", help="exit 75 instead of waiting when it does not fit")
    r.add_argument("command", nargs=argparse.REMAINDER)
    c = sub.add_parser("check")
    c.add_argument("--need", type=float, required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            ap.error("run needs a command after --")
        sys.exit(run(a.need, a.label, cmd, wait=not a.no_wait))
    if a.cmd == "check":
        with locked() as jobs:
            sys.exit(0 if fits(jobs, a.need) else 1)
    with locked() as jobs:
        kids = _children()
        print(f"MemAvailable {mem_available_gb():.1f} GB, effectively free {effective_free_gb(jobs):.1f} GB, "
              f"headroom {HEADROOM_GB:g} GB")
        for j in jobs:
            print(f"  {j['label']}: need {j['need_gb']:g} GB, now {tree_rss_gb(j['pid'], kids):.1f} GB, "
                  f"{(time.time() - j['start']) / 60:.0f} min")


if __name__ == "__main__":
    main()
