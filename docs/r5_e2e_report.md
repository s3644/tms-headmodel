# R5 end-to-end report (2026-11-05)

**Rule:** `runs/prereg/r5_e2e.json`, sealed 2026-11-03T02:31:12Z, sha256 `dfbeffe3…6eb` (ledger entry unchanged).
**Chain:** `scripts/train_r5_warp.sh`. It ended at 13:22:02 on 5 Nov with no errors.
**Sources:** `runs/r5/e2e/{pilot,test,compare}/summary.{md,json}`.
**Run:** the test set was run once, and nothing has been re-run since.

All distances are in mm, from our points to charm's. Each cell gives the mean of case means / the worst point.

**Rows** (the rTMS EEG set and the fiducials are both projected on our predicted skin):
- targets: `mni2subject_coords`, unprojected;
- rTMS EEG set;
- fiducials.

**Arms:**
- 1mm: the original T1;
- ax5: 5 mm axial;
- ax5gap: 5 mm axial with a gap;
- axsag5: 5 mm axial + sagittal.

## Decision under the sealed rule: ADOPT R5

The rule is judged on the test set's 1mm arm:

| check | R5 | release | result |
|---|---|---|---|
| no folding | 0 folded | 0 | ok |
| sum of the three row means lower | 4.15 | 5.62 | ok |
| no row mean worse by > 0.05 | all lower (−0.37, −0.74, −0.37) | | ok |
| no row worst point worse by > 1.0 | 5.47 vs 7.00; 5.09 vs 8.63; 10.94 vs 11.93 | | ok |
| navprep ≤ 120 s per case | 36 s | 20 s | ok |
| GPU peak ≤ 7.5 GB | 7.24 GB | 7.33 GB | ok |

Paired Wilcoxon tests on the 1mm arm (n = 100), mean difference R5 minus release:

| row | mean difference | p |
|---|---|---|
| targets | −0.37 | 3e-11 |
| rTMS EEG | −0.74 | 5e-18 |
| fiducials | −0.37 | 5e-10 |

Per the sealed rule, adoption does not by itself build a release; the release (Zenodo 10.5281/zenodo.24215664) was built afterwards from these models.

## WP3 gate (≤ 2 mm mean and ≤ 5 mm worst point, per row): not met on any row

### Test set (IXI, 100 heads)

| model | arm | targets | rTMS EEG | fiducials | navprep s | GPU GB | heads passing all |
|---|---|---|---|---|---|---|---|
| release | 1mm | 1.69 / 7.00 | 2.05 / 8.63 | 1.89 / 11.93 | 20 | 7.33 | 21 |
| release | ax5 | 1.81 / 7.09 | 2.37 / 14.89 | 1.87 / 11.39 | 20 | 7.23 | 10 |
| release | ax5gap | 1.99 / 8.14 | 2.56 / 15.20 | 1.90 / 11.45 | 20 | 7.24 | 5 |
| release | axsag5 | 1.78 / 8.02 | 2.13 / 8.26 | 1.95 / 11.77 | 36 | 7.50 | 17 |
| **R5** | 1mm | 1.32 / 5.47 | 1.31 / 5.09 | 1.52 / 10.94 | 36 | 7.24 | 63 |
| **R5** | ax5 | 1.35 / 5.59 | 1.39 / 5.10 | 1.53 / 11.08 | 35 | 7.23 | 62 |
| **R5** | ax5gap | 1.37 / 5.84 | 1.43 / 9.13 | 1.55 / 11.00 | 35 | 7.24 | 59 |
| **R5** | axsag5 | 1.38 / 5.71 | 1.36 / 5.54 | 1.58 / 11.46 | 53 | 7.50 | 53 |

- **All means pass.** Every mean is now under 2 mm, and R5 is nearly as good on 5 mm scans as on 1 mm.
- **The gate fails on worst points only:**
  - targets: 5.5–5.8, against the 5.0 limit;
  - rTMS EEG: 5.1 at best (9.1 on ax5gap);
  - fiducials: about 11 in both models.
- **Fiducials** are template warps in charm, not anatomy (`docs/research/learning_from_charm.md`). The planned anatomical fiducial detector, scored against the X0 raters, is the right fix, not more charm imitation.

### Compare set (57 WP3 cases, 1mm, reported only)

| row | release (M1) | R5 | change |
|---|---|---|---|
| targets | 2.21 / 10.17 (30/57 pass) | 2.11 / 11.28 (31/57) | mean −0.11, worst +1.11 |
| rTMS EEG | 3.22 / 14.76 (10/57) | 2.85 / 15.83 (19/57) | mean −0.37, worst +1.07 |
| fiducials | 3.21 / 14.40 (16/57) | 3.37 / 15.63 (14/57) | mean +0.16, worst +1.23 |
| cases passing all | 6/57 | 13/57 | |

- navprep 28 s, GPU peak 7.45 GB, no folding.
- **Caution:** on these cases R5's gain is much smaller than on IXI.
  - The fiducials get worse.
  - Every worst point is about 1.1 mm worse.
- This set does not gate adoption. Had the same tests been applied here, R5 would have failed two of them: a row worse by more than 0.05 mm (fiducials), and a worst point worse by more than 1.0 mm (all three rows).
- **Likely reason:** R5 was trained on IXI heads, so the IXI test set is in-domain, and the 57 cases are not. This is a guess, not tested.
- It matters for the release decision, because clinic MRIs look more like the 57 than like IXI.

### Pilot (5 heads, machinery only)

The pilot ran with no folding, at 32–46 s per case, with a GPU peak of 7.46 GB. It was not scored for the decision.

## Resources

- **GPU peak:**
  - 1mm, ax5 and ax5gap arms: 7.23–7.33 GB.
  - axsag5 arm (both models): exactly 7.50 GB, which is the per-process cap. That arm is at the limit of an 8 GB card, so an RTX 5060 has no headroom there. The adoption check used the 1mm arm.
- **navprep time:** R5 takes 35–36 s on single-series inputs (release 20 s), and 53 s on axsag5. That is still well under 120 s.

## What this means

- The fast route is not "within the gate" on any arm, so its output stays provisional until charm confirms it.
- The compare-set worst points are about 1 mm worse than the previous model's, while IXI improves on every row
  and arm.
