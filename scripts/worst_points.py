#!/usr/bin/env python3
"""Where and why the worst points are (paper 01, Figure 3 / gap G10). Reads scored outputs only; runs no model.

For R5 (the release) on: the IXI test set (r5_e2e, 4 arms), the OASIS-3 external test (r6_e2e, R5 as comparator,
4 arms) and the 57 held-out patient cases (r5_e2e compare, 1 mm). Per gate row (10 MNI targets; the 13 rTMS EEG
positions and the 4 fiducials, projected on our label skin):
  - distribution of point errors (p50 / p95 / p99 / max), points > 5 mm, cases with any;
  - which named points the > 5 mm points are;
  - for each > 5 mm EEG / fiducial point, its share:
      warp  = our warp projected on charm's skin   (proj_charm_skin)  > half the total
      skin  = charm's warp projected on our skin   (floor_label_skin) > half the total
      edge  = charm's point within 5 mm of the T1's box (fov_margin_mm < 5) or its unprojected point outside
      no-signal = the T1 has almost no signal within 6 mm of charm's point (masked / defaced / cut): no image there
      deface<10mm = charm's point lies within 10 mm of a block that charm labels as head but the T1 is exactly 0
                    (defacing / masking): charm's own skin is unreliable there (seen on 1BC019 Fz)
  - the worst points with these numbers.
The patient set is split by source (SynthRAD centres A/B/C; Sun Yat-sen PIT = non-contrast T1, others contrast).
OASIS-3 cases are numbered, never named (DUA). Output: runs/analysis/worst_points/ (gitignored).

  python3 scripts/worst_points.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
OUT = RUNS / "analysis/worst_points"
TARGET_NAMES = [f"{k}_{s}" for k in ("dlpfc_sgacc", "dlpfc_5cm", "dlpfc_ba46", "dlpfc_ba9", "m1_hand") for s in "lr"]
RTMS_EEG = ("F3", "F4", "Fz", "Fpz", "Fp1", "Fp2", "Cz", "C3", "C4", "FCz", "T7", "T8", "Oz")
SETS = {   # name: (directory with <arm>/<case>/, arms, anonymise)
    "IXI test (100)": (RUNS / "r5/e2e/test/r5", ("1mm", "ax5", "ax5gap", "axsag5"), False),
    "OASIS-3 external (100)": (RUNS / "r6/e2e/test/r5", ("1mm", "ax5", "ax5gap", "axsag5"), True),
    "patients held out (57)": (RUNS / "r5/e2e/compare/r5", ("1mm",), False),
}
BIG = 5.0
M2M_ROOTS = (Path("/mnt/24tb/headmodel/ixi"), Path("/mnt/24tb/headmodel/oasis3_mri"))
_T1: dict = {}


def m2m_of(case: str) -> Path | None:
    for r in M2M_ROOTS:
        d = r / case / f"m2m_{case}"
        if d.is_dir():
            return d
    man = REPO / "data/manifests/round1.json"
    info = json.loads(man.read_text())["cases"].get(case) if man.is_file() else None
    return Path(info["m2m"]) if info else None


_DEF: dict = {}


def deface_dist(case: str, xyz) -> float:
    """Distance (mm) from the reference point to the nearest voxel that charm labels as head but where the T1 is
    exactly 0 (a defacing / masking block or a cut edge); inf if there is no such block (< 500 voxels)."""
    if case not in _DEF:
        from scipy import ndimage
        d = m2m_of(case)
        img = nib.load(d / "T1.nii.gz")
        a = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
        lab = np.squeeze(np.asarray(nib.load(d / "final_tissues.nii.gz").dataobj)) > 0
        blk = lab & (a == 0)
        vox = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
        _DEF[case] = (img, ndimage.distance_transform_edt(~blk, sampling=vox) if blk.sum() >= 500 else None,
                      int(blk.sum()))
    img, dist, _ = _DEF[case]
    if dist is None:
        return float("inf")
    ijk = np.clip(np.round(nib.affines.apply_affine(np.linalg.inv(img.affine), np.asarray(xyz, float))).astype(int),
                  0, np.array(dist.shape) - 1)
    return float(dist[tuple(ijk)])


def no_signal(case: str, xyz, r_mm: float = 6.0) -> bool:
    """True if the T1 has (almost) no signal within r_mm of the reference point: a masked / defaced / cut region,
    where charm's point is the atlas's guess, not the image's."""
    if case not in _T1:
        d = m2m_of(case)
        img = nib.load(d / "T1.nii.gz") if d else None
        a = np.asarray(img.dataobj, dtype=np.float32) if img is not None else None
        _T1[case] = (img, a, float(np.percentile(a[a > 0], 99)) if a is not None else 0.0)
    img, a, p99 = _T1[case]
    if img is None:
        return False
    ijk = np.round(nib.affines.apply_affine(np.linalg.inv(img.affine), np.asarray(xyz, float))).astype(int)
    r = int(np.ceil(r_mm / np.sqrt((img.affine[:3, :3] ** 2).sum(0)).min()))
    lo, hi = np.maximum(ijk - r, 0), np.minimum(ijk + r + 1, a.shape[:3])
    if np.any(hi <= lo):
        return True
    return float(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max()) < 0.03 * p99


def source(case: str) -> str:
    if case.startswith("1B") and len(case) > 2:
        return f"SynthRAD-{case[2]}"
    if case.startswith("PIT"):
        return "SYSU-PIT (non-contrast)"
    if case[:2].isalpha() and not case.startswith("IXI") and not case.startswith("OAS"):
        return "SYSU-other (contrast)"
    return ""


def load_case(d: Path) -> dict | None:
    e, p = d / "eval.json", d / "projected.json"
    if not (e.is_file() and p.is_file()):
        return None
    ev, pr = json.loads(e.read_text()), json.loads(p.read_text())
    rows = {"targets": [{"name": n, "total": float(v)} for n, v in zip(TARGET_NAMES, ev["err_targets"])]}
    eeg, fid = pr["groups"]["eeg"], pr["groups"]["fiducials"]
    edge_all = np.asarray(eeg["fov_margin_mm"], float)
    for row, g, names in (("rtms_eeg", eeg, RTMS_EEG), ("fiducials", fid, tuple(fid["names"]))):
        out = []
        for n in names:
            i = g["names"].index(n)
            out.append({"name": n, "total": g["proj_label_skin"][i], "warp": g["proj_charm_skin"][i],
                        "skin": g["floor_label_skin"][i], "normal": g["unprojected_normal"][i],
                        "tangential": g["unprojected_tangential"][i], "fov": g["fov_margin_mm"][i],
                        "fov_unproj": g["fov_margin_unprojected_mm"][i], "xyz": g["charm_xyz"][i]})
        rows[row] = out
    rows["_case"] = d.name
    rows["_cap_edge_points"] = int((edge_all < BIG).sum())   # of the 80 cap points, how many lie within 5 mm of the box
    return rows


def why(p: dict) -> list[str]:
    tags = ["no-signal"] if p.get("nosig") else []
    if p.get("deface", float("inf")) < 10:
        tags.append("deface<10mm")
    if p.get("fov", 99) < BIG or p.get("fov_unproj", 99) < 0:
        tags.append("edge")
    if "skin" in p and p["skin"] > 0.5 * p["total"]:
        tags.append("skin")
    if "warp" in p and p["warp"] > 0.5 * p["total"]:
        tags.append("warp")
    return tags or ["mixed"]


def q(a, x):
    return float(np.percentile(a, x)) if len(a) else float("nan")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Worst points of R5 (the release): where and why", "",
          "Scored outputs only (no model run). Rows: 10 MNI targets (unprojected); 13 rTMS EEG positions and 4 "
          "fiducials projected on our label skin. Shares of a point > 5 mm: *warp* = our warp on charm's skin > half "
          "the total; *skin* = charm's warp on our skin > half; *edge* = charm's point within 5 mm of the T1 box "
          "(or its unprojected point outside); *no-signal* = the T1 is (almost) empty within 6 mm of charm's point (masked, defaced or cut), so the reference is the atlas's guess; *deface<10mm* = charm's point within 10 mm of a block charm labels as head but the T1 is exactly 0 (defacing or masking), where charm's skin can be cut (1BC019 Fz: charm 13 mm inside the visible skin). Generated by scripts/worst_points.py.", ""]
    res = {}
    for sname, (root, arms, anon) in SETS.items():
        cases = sorted(p.name for p in (root / arms[0]).iterdir() if p.is_dir()) if (root / arms[0]).is_dir() else []
        alias = {c: (f"O{i + 1:03d}" if anon else c) for i, c in enumerate(cases)}
        md += [f"## {sname}", ""]
        md += ["| arm | row | points | p50 | p95 | p99 | max | points > 5 mm | cases with any | > 5 mm by share |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        worst, by_name, by_src, edgecases = [], defaultdict(Counter), defaultdict(lambda: [0, 0]), []
        for arm in arms:
            data = {c: load_case(root / arm / c) for c in cases}
            data = {c: v for c, v in data.items() if v}
            for row in ("targets", "rtms_eeg", "fiducials"):
                pts = [(c, p) for c, v in data.items() for p in v[row]]
                tot = np.array([p["total"] for _, p in pts])
                big = [(c, p) for c, p in pts if p["total"] > BIG]
                for c, p in big:
                    if "xyz" in p and "nosig" not in p:
                        p["nosig"] = no_signal(c, p["xyz"])
                        p["deface"] = deface_dist(c, p["xyz"])
                shares = Counter(t for _, p in big for t in why(p)) if row != "targets" else Counter()
                md.append(f"| {arm} | {row} | {len(tot)} | {q(tot, 50):.2f} | {q(tot, 95):.2f} | {q(tot, 99):.2f} | "
                          f"{tot.max():.2f} | {len(big)} | {len({c for c, _ in big})}/{len(data)} | "
                          + (", ".join(f"{k} {v}" for k, v in shares.most_common()) or "-") + " |")
                for c, p in big:
                    by_name[f"{row}"][p["name"]] += 1
                    worst.append((p["total"], arm, alias[c], row, p))
                if arm == arms[0] and sname.startswith("patients"):
                    for c, v in data.items():
                        s = by_src[source(c)]
                        s[0] += 1 if row == "targets" else 0
                        s[1] += sum(p["total"] > BIG for p in v[row])
            if arm == arms[0]:
                for c in data:
                    deface_dist(c, [0, 0, 0])
                nblk = sum(1 for c in data if _DEF.get(c, (0, None, 0))[1] is not None)
                md_def = f"Cases ({arm}) with a defacing / masking block (charm head label on T1 = 0, >= 500 voxels): {nblk}/{len(data)}"
                edgecases = sorted(((v["_cap_edge_points"], alias[c]) for c, v in data.items()), reverse=True)
        md += ["", "Points > 5 mm by name (all arms): " + "; ".join(
            f"{row}: " + ", ".join(f"{n} {k}" for n, k in cnt.most_common(8)) for row, cnt in by_name.items()), ""]
        md += [md_def, ""]
        n_edge = sum(k > 0 for k, _ in edgecases)
        md += [f"Cases ({arms[0]}) with any of the 80 cap points within 5 mm of the image box: {n_edge}/{len(edgecases)}"
               + (f" (most: {', '.join(f'{c} {k}' for k, c in edgecases[:5] if k)})" if n_edge else ""), ""]
        if by_src:
            md += ["Patients by source (1 mm): " + "; ".join(
                f"{s or '?'}: {n} cases, {k} points > 5 mm" for s, (n, k) in sorted(by_src.items())), ""]
        worst.sort(key=lambda r: -r[0])
        md += ["Worst 12 points:", "", "| mm | arm | case | row | point | warp share | skin share | normal / tangential | "
               "charm point to image edge | to defacing block | why |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for tot, arm, c, row, p in worst[:12]:
            src = f" ({source(c)})" if source(c) else ""
            md.append(f"| {tot:.2f} | {arm} | {c}{src} | {row} | {p['name']} | "
                      + (f"{p['warp']:.2f} | {p['skin']:.2f} | {p['normal']:.2f} / {p['tangential']:.2f} | "
                         f"{p['fov']:.1f} mm | {p.get('deface', float('inf')):.0f} mm | {', '.join(why(p))} |"
                         if row != "targets" else "- | - | - | - | - | warp |"))
        md.append("")
        res[sname] = {"worst": [(t, a, c, r, p) for t, a, c, r, p in worst[:50]],
                      "by_name": {k: dict(v) for k, v in by_name.items()}}
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
