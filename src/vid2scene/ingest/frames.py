"""Frame selection and validation.

Strategy (no ground-truth poses required):
  * Pass 1 decodes the video and scores every (sampled) frame for
    - sharpness   = variance of the Laplacian (motion-blur / focus gate)
    - motion      = mean abs difference vs the previous frame on a small gray
                    image, used as a proxy for camera parallax / baseline.
  * Keyframes are spaced by *accumulated motion*, not by a fixed time stride,
    so a slow pan and a fast pan both yield well-distributed views with
    enough baseline between them. Within each motion "slot" we keep the
    sharpest frame, so blurry frames lose to a sharp neighbour.
  * Frames below an adaptive sharpness threshold are flagged; the report
    records the reason every frame was kept or dropped.

When phone poses/timestamps are available (later), the motion proxy is
replaced by real inter-frame translation/rotation — same selection logic.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameStat:
    index: int            # frame index in the source video
    time_s: float         # timestamp in seconds
    sharpness: float      # variance of Laplacian (higher = sharper)
    motion: float         # mean abs diff vs previous sampled frame
    selected: bool = False
    reason: str = ""      # human-readable keep/drop reason
    path: str | None = None       # saved keyframe filename (if selected)
    thumb: str | None = None      # saved thumbnail filename (if selected)


@dataclass
class IngestConfig:
    target_frames: int = 60       # desired number of keyframes
    sample_stride: int = 1        # analyse every Nth frame (speed knob)
    blur_rel_threshold: float = 0.6   # flag frames below this * median sharpness
    proc_width: int = 320         # width used for sharpness/motion scoring
    thumb_width: int = 256        # report thumbnail width
    jpeg_quality: int = 92


def _to_small_gray(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / float(w)
    small = cv2.resize(frame, (width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def analyze_video(video: Path, cfg: IngestConfig) -> tuple[list[FrameStat], float]:
    """Pass 1: score every sampled frame."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    stats: list[FrameStat] = []
    prev_small: np.ndarray | None = None
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % cfg.sample_stride:
            continue
        small = _to_small_gray(frame, cfg.proc_width)
        sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
        motion = 0.0 if prev_small is None else float(np.mean(cv2.absdiff(small, prev_small)))
        stats.append(FrameStat(index=idx, time_s=idx / fps, sharpness=sharp, motion=motion))
        prev_small = small
    cap.release()
    if not stats:
        raise RuntimeError(f"no frames decoded from {video}")
    return stats, fps


def select_keyframes(stats: list[FrameStat], cfg: IngestConfig) -> dict:
    """Pass 2 (in-memory): choose motion-spaced, sharp keyframes."""
    n = len(stats)
    sharp = np.array([s.sharpness for s in stats])
    motion = np.array([s.motion for s in stats])
    median_sharp = float(np.median(sharp))
    blur_thresh = cfg.blur_rel_threshold * median_sharp
    target = max(1, min(cfg.target_frames, n))
    total_motion = float(motion.sum())

    selected: set[int] = set()
    if total_motion > 1e-6 and target < n:
        # Place `target` evenly along the accumulated-motion axis, then snap
        # each to the sharpest frame in its local window.
        cum = np.cumsum(motion)
        step = total_motion / target
        half = max(1, int(0.5 * n / target))
        for k in range(target):
            target_m = (k + 0.5) * step
            center = int(np.searchsorted(cum, target_m))
            center = min(max(center, 0), n - 1)
            lo, hi = max(0, center - half), min(n, center + half + 1)
            best = max(range(lo, hi), key=lambda i: stats[i].sharpness)
            selected.add(best)
    else:
        # Static or tiny clip: fall back to even index spacing.
        for i in np.linspace(0, n - 1, target).astype(int):
            selected.add(int(i))

    for i, s in enumerate(stats):
        below = s.sharpness < blur_thresh
        if i in selected:
            s.selected = True
            s.reason = "keyframe" if not below else "keyframe (sharpest in window, still soft)"
        elif below:
            s.reason = "dropped: below sharpness threshold"
        else:
            s.reason = "dropped: redundant (low parallax)"

    return {
        "analysed_frames": n,
        "selected_frames": len(selected),
        "dropped_blur": int(sum(1 for s in stats if not s.selected and "sharpness" in s.reason)),
        "dropped_redundant": int(sum(1 for s in stats if not s.selected and "redundant" in s.reason)),
        "median_sharpness": median_sharp,
        "blur_threshold": blur_thresh,
        "total_motion": total_motion,
    }


def save_keyframes(video: Path, stats: list[FrameStat], frames_dir: Path,
                   thumbs_dir: Path, cfg: IngestConfig) -> None:
    """Pass 3: sequentially re-decode and write the selected frames + thumbs."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    by_index = {s.index: s for s in stats if s.selected}
    order = {idx: k for k, idx in enumerate(sorted(by_index))}

    cap = cv2.VideoCapture(str(video))
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        s = by_index.get(idx)
        if s is None:
            continue
        k = order[idx]
        name = f"frame_{k:04d}.jpg"
        cv2.imwrite(str(frames_dir / name), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
        h, w = frame.shape[:2]
        tw = cfg.thumb_width
        thumb = cv2.resize(frame, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumbs_dir / name), thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
        s.path = f"frames/{name}"
        s.thumb = f"thumbs/{name}"
    cap.release()


def run_ingest(video: str | Path, out_dir: str | Path,
               cfg: IngestConfig | None = None) -> dict:
    """End-to-end Stage 1. Returns a summary dict and writes:
       out_dir/frames/*.jpg, out_dir/thumbs/*.jpg,
       out_dir/frame_report.json, out_dir/report.html
    """
    from . import report as report_mod  # local import to avoid cycle

    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or IngestConfig()

    stats, fps = analyze_video(video, cfg)
    summary = select_keyframes(stats, cfg)
    save_keyframes(video, stats, out_dir / "frames", out_dir / "thumbs", cfg)

    summary.update({"video": str(video), "fps": fps,
                    "config": dataclasses.asdict(cfg)})
    report_mod.write_reports(out_dir, stats, summary)
    return summary
