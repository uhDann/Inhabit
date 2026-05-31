"""Generate a synthetic 'room pan' video for smoke-testing Stage 1.

Pans a crop window across a textured canvas (simulating camera motion) and
injects motion blur on a handful of frames so the sharpness gate has
something to reject. Usage:  python scripts/make_sample_video.py [out.mp4]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/sample_room.mp4")
W, H = 1280, 720
CW, CH = 2400, 720          # wide canvas to pan across
N, FPS = 150, 30
BLUR_FRAMES = {20, 21, 55, 90, 91, 92, 120}

rng = np.random.default_rng(0)
canvas = np.zeros((CH, CW, 3), np.uint8)
canvas[:] = (45, 40, 38)
# vertical brightness gradient (floor->ceiling feel)
canvas += np.linspace(0, 40, CH, dtype=np.uint8)[:, None, None]
# scatter "furniture": rectangles, circles, lines + labels for texture/parallax
for _ in range(60):
    x, y = int(rng.integers(0, CW)), int(rng.integers(0, CH))
    w, h = int(rng.integers(40, 240)), int(rng.integers(40, 220))
    color = tuple(int(c) for c in rng.integers(60, 255, 3))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (20, 20, 20), thickness=2)
for _ in range(40):
    cv2.circle(canvas, (int(rng.integers(0, CW)), int(rng.integers(0, CH))),
               int(rng.integers(8, 40)), tuple(int(c) for c in rng.integers(60, 255, 3)), -1)
for x in range(0, CW, 200):
    cv2.putText(canvas, f"x={x}", (x + 6, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (240, 240, 240), 2)

# horizontal motion-blur kernel
ksize = 21
mb = np.zeros((ksize, ksize), np.float32)
mb[ksize // 2, :] = 1.0 / ksize

OUT.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for i in range(N):
    t = i / (N - 1)
    x = int(t * (CW - W))
    frame = canvas[0:H, x:x + W].copy()
    if i in BLUR_FRAMES:
        frame = cv2.filter2D(frame, -1, mb)
    writer.write(frame)
writer.release()
print(f"wrote {OUT} ({N} frames @ {FPS}fps, blur on {sorted(BLUR_FRAMES)})")
