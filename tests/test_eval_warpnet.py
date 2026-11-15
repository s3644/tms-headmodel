import sys
from pathlib import Path

import numpy as np
import pytest

from headmodel import register

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytestmark = pytest.mark.skipif(not (register.simnibs_dir() / "resources").is_dir(), reason="simnibs missing")


def _row(case, names, bad=None, fold=False):
    """A scored case row: 1 mm everywhere, 9 mm at the named EEG position `bad` (all three EEG variants)."""
    e = np.ones(len(names))
    if bad:
        e[names.index(bad)] = 9.0
    r = {"case": case, "targets": {"mean": 1.0, "max": 1.0}, "gpu_s": 1.0, "peak_mem_gb": 1.0,
         "round_trip_max_mm": 0.1, "jacobian_min": -0.1 if fold else 0.5, "jacobian_min_svf": 0.5,
         "jacobian_written": {"min": -0.1 if fold else 0.5}}
    for k in ("eeg", "eeg_charm_skin", "eeg_label_skin"):
        r[k] = {"mean": float(e.mean()), "max": float(e.max())}
        r[f"err_{k}"] = e.tolist()
    for k in ("fid_charm_skin", "fid_label_skin"):
        r[k] = {"mean": 1.0, "max": 1.0}
    return r


def test_gate_uses_the_rtms_set_and_ignores_the_rest_of_the_cap():
    import eval_register as ev
    import eval_warpnet as ew

    names = ev.eeg_cap()[0]
    assert set(ev.RTMS_EEG) <= set(names)
    s = ew.summarise([_row("a", names, bad="TP10"), _row("b", names)])
    assert s["eeg"]["max"] == 9.0 and s["eeg_rtms"]["max"] == 1.0
    assert s["gate_rtms"]["met"]
    s = ew.summarise([_row("a", names, bad="F3"), _row("b", names)])
    assert s["eeg_rtms"]["max"] == 9.0 and not s["gate_rtms"]["eeg_rtms"] and not s["gate_rtms"]["met"]
    s = ew.summarise([_row("a", names), _row("b", names, fold=True)])
    assert not s["gate_rtms"]["no_folding"] and not s["gate_rtms"]["met"]
