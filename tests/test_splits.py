import json
import warnings

import pytest

from headmodel import splits


def _redirect(monkeypatch, tmp_path):
    for name in ("WP3_EVAL", "WP3_TEST_FILE", "VAL_FILE", "VAL_PROVENANCE", "WP3_TRAIN_SNAPSHOT"):
        monkeypatch.setattr(splits, name, tmp_path / f"{name}.json")
    monkeypatch.setattr(splits, "REPO", tmp_path)


def test_wp3_test_cases_frozen_once(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(splits, "N_WP3_TEST", 3)
    splits.WP3_EVAL.write_text(json.dumps({"cases": [{"case": c} for c in ("B", "A", "C")]}))
    assert splits.wp3_test_cases() == ["A", "B", "C"]
    splits.WP3_EVAL.write_text(json.dumps({"cases": [{"case": c} for c in ("A", "B", "C", "D")]}))
    assert splits.wp3_test_cases() == ["A", "B", "C"]   # the frozen file wins over a rewritten eval.json
    monkeypatch.setattr(splits, "N_WP3_TEST", 4)
    with pytest.raises(RuntimeError):
        splits.wp3_test_cases()


def test_val_cases_fixed_file_wins_and_only_warns(monkeypatch, tmp_path):
    _redirect(monkeypatch, tmp_path)
    monkeypatch.setattr(splits, "N_VAL", 2)
    snap = [f"t{i}" for i in range(6)]
    splits.WP3_TRAIN_SNAPSHOT.write_text(json.dumps({"cases": [{"case": c} for c in snap]}))
    man = {"cases": {c: {"split": "train"} for c in snap}}
    val = splits.val_cases(man)
    assert val == splits.sha1_rule(snap, 2) and json.loads(splits.VAL_PROVENANCE.read_text())["train_cases"] == snap
    grown = {"cases": {**man["cases"], **{f"n{i}": {"split": "train"} for i in range(40)}}}
    assert splits.sha1_rule(list(grown["cases"]), 2) != val
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert splits.val_cases(grown) == val
    assert any("fixed file is used" in str(x.message) for x in w)
    with pytest.raises(RuntimeError):
        splits.val_cases({"cases": {c: {"split": "test"} for c in snap}})
