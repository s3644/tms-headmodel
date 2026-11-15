# Differences from the private repository

This public copy is a cleaned export of a private research repository. It differs from it as follows.

## Redactions in files

OASIS participant identifiers are replaced by neutral labels (caseNN) to follow the OASIS data use
agreement; LAN addresses are removed. Files changed:

- `docs/r5_e2e_report.md`
- `runs/prereg/r6_train_plan.json`
- `scripts/prep_oasis_rescan.py`
- `scripts/queue_oasis3_mri.sh`

Sealed rules among them: their SHA-256 in `runs/prereg/ledger.txt` is that of the original,
unredacted file; the original files are kept privately and can be shown to an editor or reviewer
on request. All other sealed rules are byte-identical to the originals and match the ledger.

## Left out (from 0.1.1)

- Material of other projects: internal research notes, the CT route (navprep's CT modality and its model),
  implant and fiducial-marking tools, a design note of our navigation application, job and deposit tooling,
  and the two sealed rules of the CT project (`oasis3_ct_reference*.json`; their ledger lines stay, so the
  ledger is complete). None of them is used by the MRI route or by the paper's scoring.
- `docs/r5_e2e_report.md`: its closing list of internal next steps is replaced by a short summary; the results
  are unchanged.
- Local paths to our navigation application are replaced by environment variables (`HM_LOCATOR_BACKEND` in
  `scripts/eval_register.py`, `HM_ERNIE_M2M` in `tests/test_contract.py`).
- The MRI route itself (`navprep.py`: segmentation, warp, cortex) is unchanged: the same input gives the same
  output as the private code within the GPU's run-to-run variation (about 0.002 % of label voxels).
