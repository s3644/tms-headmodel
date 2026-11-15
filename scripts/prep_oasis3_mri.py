"""OASIS-3 MRI heads for R6 (user go 2026-11-05): one 3D T1 per participant, charm reference, train / test split.

    python scripts/prep_oasis3_mri.py            # writes ROOT/selection.tsv and ROOT/queue.txt, prints the split

Selection (deterministic, before any charm run on these heads):
- participants with a 3D T1w NIfTI (voxels <= 1.2 mm, >= 150 slices on each of two axes), not in the CT-T1 pairs of
  /mnt/24tb/headmodel/oasis3/pairs.tsv (444 participants; kept clean for the CT track);
- per participant the earliest such MR session, its lowest run number;
- clinical status at that session = CDR total of the UDS B4 visit closest in days (within 365 d; else 'unknown',
  left out): 'normal' (CDR 0) or 'decline' (CDR >= 0.5);
- in each stratum, sort by sha1('r6-oasis3:' + participant) and take the first 250 (fewer if the stratum is smaller);
- test = per stratum the 50 with the smallest sha1('r6-oasis3-test:' + participant); train = the others.
queue.txt lists test first, then train (so the test heads have their reference first).
Output under ROOT (OASIS DUA: never in git, never off this machine; no face renderings).
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import nibabel as nib

SRC = Path("/mnt/24tb/datasets/oasis/oasis3")
PAIRS = Path("/mnt/24tb/headmodel/oasis3/pairs.tsv")
ROOT = Path("/mnt/24tb/headmodel/oasis3_mri")
N_PER, N_TEST = 250, 50


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def ok_t1(f: Path) -> bool:
    try:
        img = nib.load(str(f))
    except Exception:
        return False
    z = img.header.get_zooms()[:3]
    sh = img.shape[:3]
    return len(sh) == 3 and max(z) <= 1.2 and sorted(sh)[1] >= 150


def main() -> None:
    ct = {r["participant"] for r in csv.DictReader(PAIRS.open(), delimiter="\t")}
    t1s: dict[str, list] = {}
    for f in sorted((SRC / "mr").glob("OAS3*_MR_d*/anat*/NIFTI/*_T1w.nii.gz")):
        m = re.match(r"(OAS3\d{4})_MR_d(\d{4})", f.parts[-4])
        run = re.search(r"run-(\d+)", f.name)
        t1s.setdefault(m.group(1), []).append((int(m.group(2)), int(run.group(1)) if run else 0, f))
    cdr: dict[str, list] = {}
    b4 = next((SRC / "data_files/OASIS3_data_files/UDSb4/csv").glob("*.csv"))
    for r in csv.DictReader(b4.open()):
        if r["CDRTOT"] not in ("", None):
            cdr.setdefault(r["OASISID"], []).append((int(float(r["days_to_visit"])), float(r["CDRTOT"])))
    rows, skipped = [], {"ct_pair": 0, "no_3d_t1": 0, "no_cdr": 0}
    for p in sorted(t1s):
        if p in ct:
            skipped["ct_pair"] += 1
            continue
        pick = next((x for x in sorted(t1s[p]) if ok_t1(x[2])), None)
        if pick is None:
            skipped["no_3d_t1"] += 1
            continue
        day = pick[0]
        near = min(cdr.get(p, []), key=lambda v: abs(v[0] - day), default=None)
        if near is None or abs(near[0] - day) > 365:
            skipped["no_cdr"] += 1
            continue
        rows.append({"participant": p, "day": f"d{day:04d}", "t1": str(pick[2]), "cdr": near[1],
                     "stratum": "normal" if near[1] == 0 else "decline"})
    chosen = []
    for s in ("normal", "decline"):
        g = sorted((r for r in rows if r["stratum"] == s), key=lambda r: h("r6-oasis3:" + r["participant"]))[:N_PER]
        test = {r["participant"] for r in sorted(g, key=lambda r: h("r6-oasis3-test:" + r["participant"]))[:N_TEST]}
        for r in g:
            r["set"] = "test" if r["participant"] in test else "train"
        chosen += g
    ROOT.mkdir(parents=True, exist_ok=True)
    keys = ["id", "participant", "day", "stratum", "cdr", "set", "t1"]
    with (ROOT / "selection.tsv").open("w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in chosen:
            r["id"] = f"{r['participant']}_{r['day']}"
            fh.write("\t".join(str(r[k]) for k in keys) + "\n")
    order = [r["id"] for r in chosen if r["set"] == "test"] + [r["id"] for r in chosen if r["set"] == "train"]
    (ROOT / "queue.txt").write_text("\n".join(order) + "\n")
    for s in ("normal", "decline"):
        print(s, {k: sum(1 for r in chosen if r["stratum"] == s and r["set"] == k) for k in ("train", "test")},
              "eligible", sum(1 for r in rows if r["stratum"] == s))
    print("skipped", skipped, "| chosen", len(chosen))


if __name__ == "__main__":
    main()
