"""Validation report writers (JSON + standalone HTML contact sheet)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from .frames import FrameStat


def write_reports(out_dir: Path, stats: list[FrameStat], summary: dict) -> None:
    out_dir = Path(out_dir)
    payload = {"summary": summary, "frames": [dataclasses.asdict(s) for s in stats]}
    (out_dir / "frame_report.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "report.html").write_text(_render_html(stats, summary))


def _render_html(stats: list[FrameStat], summary: dict) -> str:
    kept = [s for s in stats if s.selected]
    cards = "\n".join(
        f'<figure><img src="{s.thumb}" loading="lazy">'
        f'<figcaption>#{s.index} · {s.time_s:.2f}s · sharp {s.sharpness:.0f}</figcaption></figure>'
        for s in kept
    )
    stat_rows = "".join(
        f"<tr><td>{k}</td><td>{_fmt(v)}</td></tr>"
        for k, v in summary.items() if k != "config"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>vid2scene — ingest report</title>
<style>
 body {{ font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }}
 table {{ border-collapse: collapse; margin: 1rem 0; }}
 td {{ border: 1px solid #ddd; padding: 4px 10px; }}
 td:first-child {{ color: #555; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr));
          gap: 10px; }}
 figure {{ margin: 0; }}
 img {{ width: 100%; border-radius: 6px; display: block; }}
 figcaption {{ font-size: 11px; color: #666; padding-top: 2px; }}
</style></head><body>
<h1>vid2scene — frame validation</h1>
<p>Kept <b>{summary.get('selected_frames')}</b> of {summary.get('analysed_frames')} analysed frames
 — dropped {summary.get('dropped_blur')} for blur, {summary.get('dropped_redundant')} as redundant.</p>
<table>{stat_rows}</table>
<h2>Selected keyframes</h2>
<div class="grid">{cards}</div>
</body></html>"""


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else str(v)
