"""Tests for the benchmark collation (pure-Python, no GPU / heavy deps)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vid2scene.benchmark.table import collate, METRICS  # noqa: E402


def _write(eval_dir, scene, method, metrics):
    d = os.path.join(eval_dir, f"{scene}_{method}")
    os.makedirs(d, exist_ok=True)
    json.dump(metrics, open(os.path.join(d, f"{scene}_{method}_metrics.json"), "w"))


def test_collate_discovers_and_averages(tmp_path):
    ev = str(tmp_path / "eval")
    # two scenes, two methods; Chamfer "C-L1" averages cleanly
    _write(ev, "room0", "pgsr", {"Acc": 0.01, "Comp": 0.02, "C-L1": 0.015, "NC": 0.97, "F-score": 0.97})
    _write(ev, "room1", "pgsr", {"Acc": 0.01, "Comp": 0.02, "C-L1": 0.025, "NC": 0.95, "F-score": 0.95})
    _write(ev, "room0", "dn", {"Acc": 0.005, "Comp": 0.006, "C-L1": 0.006, "NC": 0.99, "F-score": 0.99})

    out = collate(ev, markdown=False)
    assert set(out) >= {"room0", "room1", "AVG"}
    assert "pgsr" in out["room0"] and "dn" in out["room0"]
    # pgsr Chamfer averages (0.015 + 0.025) / 2 = 0.020
    assert abs(out["AVG"]["pgsr"]["C-L1"] - 0.020) < 1e-9
    # dn only appears in room0, so its AVG equals that single value
    assert abs(out["AVG"]["dn"]["C-L1"] - 0.006) < 1e-9


def test_collate_empty_dir_is_graceful(tmp_path):
    assert collate(str(tmp_path / "nothing"), markdown=False) == {}


def test_metrics_schema():
    keys = {k for k, *_ in METRICS}
    assert keys == {"Acc", "Comp", "C-L1", "NC", "F-score"}
