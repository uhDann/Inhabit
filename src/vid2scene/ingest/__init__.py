"""Stage 1 — ingest & validation.

Turns a raw phone video into a curated set of keyframes, and emits a
validation report explaining *why* each frame was kept or dropped. This is
the "camera systems (timestamping, synchronization, validation)" surface:
sharpness gating, motion/parallax-aware sampling, and (later) pose/timestamp
ingestion for the pose-free vs pose-assisted comparison.
"""

from .frames import FrameStat, IngestConfig, run_ingest

__all__ = ["FrameStat", "IngestConfig", "run_ingest"]
