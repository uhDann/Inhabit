"""Collate per-method GT-mesh metric JSONs into a comparison table.

Each eval writes a JSON like:
    {"Acc": 0.0057, "Comp": 0.0067, "C-L1": 0.0062, "NC": 0.988, "F-score": 0.997}
(distances in metres). This reads a directory of them, converts to cm, averages
across scenes, and prints a Markdown table — the artifact that goes in the
README / BENCHMARK.md.

CLI:
    python -m vid2scene benchmark --eval-dir runs/eval --markdown
"""
from __future__ import annotations

import glob
import json
import os

# (json key, display name, unit-scale, higher-is-better)
METRICS = [
    ("Acc", "Acc", 100.0, False),
    ("Comp", "Comp", 100.0, False),
    ("C-L1", "Chamfer-L1", 100.0, False),
    ("NC", "Normal-C", 1.0, True),
    ("F-score", "F-score", 1.0, True),
]


def _discover(eval_dir: str) -> dict[str, dict[str, dict]]:
    """Find metric JSONs named like <scene>_<method>/<scene>_<method>_metrics.json
    (or any *_metrics.json) -> {scene: {method: metrics}}."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(os.path.join(eval_dir, "**", "*_metrics.json"), recursive=True)):
        stem = os.path.basename(f).replace("_metrics.json", "")
        parts = stem.split("_")
        method = parts[-1]
        scene = "_".join(parts[:-1]) or "scene"
        try:
            out.setdefault(scene, {})[method] = json.load(open(f))
        except Exception:
            pass
    return out


def collate(eval_dir: str, markdown: bool = True) -> dict:
    """Return {scene: {method: metrics}} plus an 'AVG' pseudo-scene; optionally
    print a Markdown table (cm, F-score@5cm)."""
    data = _discover(eval_dir)
    if not data:
        print(f"no *_metrics.json found under {eval_dir}")
        return {}

    methods = sorted({m for s in data.values() for m in s})
    # averages across scenes per method
    avg: dict[str, dict] = {}
    for m in methods:
        vals = [data[s][m] for s in data if m in data[s]]
        if vals:
            avg[m] = {k: sum(v.get(k, 0.0) for v in vals) / len(vals) for k, *_ in
                      [(kk,) for kk, *_ in METRICS]}
    out = dict(data)
    out["AVG"] = avg

    if markdown:
        hdr = "| Scene | Method | " + " | ".join(f"{d} {'↑' if hi else '↓'}" for _, d, _, hi in METRICS) + " |"
        sep = "|" + "---|" * (2 + len(METRICS))
        print(hdr); print(sep)
        for scene in list(data) + ["AVG"]:
            for m in methods:
                if m not in out[scene]:
                    continue
                cells = []
                for k, _, scale, _ in METRICS:
                    v = out[scene][m].get(k)
                    cells.append("—" if v is None else f"{v*scale:.2f}" if scale != 1.0 else f"{v:.3f}")
                print(f"| {scene} | {m} | " + " | ".join(cells) + " |")
    return out
