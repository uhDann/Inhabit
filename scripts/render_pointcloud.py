"""Offline point-cloud renderer (numpy + OpenCV, no GPU/browser).

Loads our binary colored PLY, projects it orthographically from chosen
angles with a z-buffer, and writes a PNG (a 2x2 multi-angle montage by
default). Used both to inspect reconstructions and to make README turntables.

    python scripts/render_pointcloud.py runs/room0_ingest/recon_pose_assisted.ply out.png
"""

import argparse
import numpy as np
import cv2


def load_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            chunk = f.read(128)
            if not chunk:
                break
            hdr += chunk
        head_len = hdr.index(b"end_header\n") + len(b"end_header\n")
        text = hdr[:head_len].decode("ascii", "ignore")
        n = int(next(l for l in text.splitlines() if l.startswith("element vertex")).split()[-1])
        f.seek(head_len)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.fromfile(f, dtype=dt, count=n)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float64)
    cols = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.uint8)
    return pts, cols


def render(pts, cols, res, azim, elev, size, flip_y=True):
    p = pts - np.median(pts, 0)
    if flip_y:
        p = p * np.array([1.0, -1.0, 1.0])
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    pv = p @ (rx @ ry).T
    u, v, z = pv[:, 0], pv[:, 1], pv[:, 2]

    ext = max(u.max() - u.min(), v.max() - v.min())
    s = (res * 0.9) / ext
    upx = ((u - (u.min() + u.max()) / 2) * s + res / 2).astype(np.int32)
    vpx = (-(v - (v.min() + v.max()) / 2) * s + res / 2).astype(np.int32)

    order = np.argsort(-z)  # far -> near so near points overwrite
    upx, vpx, c = upx[order], vpx[order], cols[order][:, ::-1]  # RGB->BGR
    img = np.zeros((res, res, 3), np.uint8)
    for dy in range(size):
        for dx in range(size):
            yy = np.clip(vpx + dy, 0, res - 1)
            xx = np.clip(upx + dx, 0, res - 1)
            img[yy, xx] = c
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=720)
    ap.add_argument("--size", type=int, default=2)
    ap.add_argument("--azims", default="25,115,205,295")
    ap.add_argument("--elev", type=float, default=18)
    args = ap.parse_args()

    pts, cols = load_ply(args.ply)
    print(f"loaded {len(pts)} points")
    azims = [float(a) for a in args.azims.split(",")]
    tiles = [render(pts, cols, args.res, az, args.elev, args.size) for az in azims]
    # label each tile with its azimuth
    for t, az in zip(tiles, azims):
        cv2.putText(t, f"azim {az:.0f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4]) if len(tiles) > 2 else top
    montage = np.vstack([top, bot]) if len(tiles) > 2 else top
    cv2.imwrite(args.out, montage)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
