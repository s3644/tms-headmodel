"""R5 warp (scripts/r5_warp.py) and end-to-end scoring (scripts/r5_e2e.py): case sets, the adoption test, the
model overrides navprep and register read."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def test_warp_sets_never_hold_a_test_head(tmp_path, monkeypatch):
    import r5_warp

    man = {"cases": {"1BA001": {"split": "test"}, "1BA002": {"split": "train"}, "1BA003": {"split": "train"},
                     "IXI1": {"split": "train", "r5_set": "train"}, "IXI2": {"split": "train", "r5_set": "val"},
                     "IXI3": {"split": "test", "r5_set": "test"}}}
    (tmp_path / "m.json").write_text(json.dumps(man))
    (tmp_path / "val.json").write_text(json.dumps(["1BA003"]))
    monkeypatch.setattr(r5_warp, "MANIFEST", tmp_path / "m.json")
    monkeypatch.setattr(r5_warp, "WP3B_VAL", tmp_path / "val.json")
    train, val = r5_warp.sets()
    assert train == ["1BA002", "IXI1"] and val == ["1BA003", "IXI2"]


def _summary(t, r, f, tmax=4.0, folded=0, secs=40.0, gpu=7.0):
    row = lambda m, x: {"mean": m, "max": x}   # noqa: E731
    return {"targets": row(t, tmax), "eeg_label_skin_rtms": row(r, 4.0), "fid_label_skin": row(f, 4.0),
            "n_cases": 100, "cases_no_fold": 100 - folded, "navprep": {"total_s_mean": secs, "gpu_peak_gb": gpu},
            "gate": {"met": False}}


@pytest.mark.parametrize("r5, adopt, failing", [
    (_summary(1.8, 2.2, 2.4), True, None),
    (_summary(2.0, 2.3, 2.5), False, "sum_of_means_lower"),             # equal sum is not better
    (_summary(1.5, 2.36, 2.0), False, "no_row_mean_worse_0.05"),         # one row 0.06 worse
    (_summary(1.8, 2.2, 2.4, tmax=5.1), False, "no_row_worst_worse_1.0"),
    (_summary(1.8, 2.2, 2.4, folded=1), False, "no_folding"),
    (_summary(1.8, 2.2, 2.4, secs=130), False, "navprep_mean_le_120s"),
    (_summary(1.8, 2.2, 2.4, gpu=7.6), False, "gpu_peak_le_7.5gb"),
])
def test_adoption_test(r5, adopt, failing):
    import r5_e2e

    res = {"r5": {"1mm": r5}, "release": {"1mm": _summary(2.0, 2.3, 2.5)}}
    d = r5_e2e.decide(res)
    assert d["adopt"] is adopt
    if failing:
        assert [k for k, v in d["checks"].items() if not v] == [failing]


def test_sim_seed_is_per_case_and_stable():
    import r5_e2e

    assert r5_e2e.case_seed("IXI014-HH-1236") == r5_e2e.case_seed("IXI014-HH-1236")
    assert r5_e2e.case_seed("IXI014-HH-1236") != r5_e2e.case_seed("IXI015-HH-1258")


def test_model_overrides(monkeypatch):
    from headmodel import navprep, register

    try:
        monkeypatch.setenv("HM_NAVPREP_MR_FOLDS", "0,1,2,3,4")
        monkeypatch.setenv("HM_WARP_NET", "/x/last.pt")
        assert importlib.reload(navprep).MODELS["mr"][1] == (0, 1, 2, 3, 4)
        assert str(importlib.reload(register).ADOPTED_NET) == "/x/last.pt"
        monkeypatch.delenv("HM_NAVPREP_MR_FOLDS")
        monkeypatch.delenv("HM_WARP_NET")
        assert importlib.reload(navprep).MODELS["mr"][1] == (0,)   # the release's default
        assert importlib.reload(register).ADOPTED_NET.name == "last.pt"
        assert "warpnet_r3/e1000" in str(register.ADOPTED_NET)
    finally:
        importlib.reload(navprep)
        importlib.reload(register)
