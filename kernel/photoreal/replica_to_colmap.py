"""Convert a Replica scene -> COLMAP text format, to feed surface-aligned splatting (PGSR /
2DGS) which gives a cleaner mesh than our lossy TSDF-of-expected-depth handoff.

PGSR mesh on office0 (full-res held-out): FID 74->61, LPIPS 0.234->0.203 vs our mesh-from-
splat -- and it propagates downstream (separation -> physics-completion -> 82% stable, mean
drift 8.2->2.9 cm). Then: PGSR train.py -s <out> -m <model> -r 2 ; render.py -m <model>.

Run: python -m photoreal.replica_to_colmap <scene_dir> <out_dir> [stride]
"""
from __future__ import annotations
import sys, os, glob, shutil
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as iio


def convert(rep, out, stride=4, W=1200, H=680, fx=600.0, fy=600.0, cx=599.5, cy=339.5):
    os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/sparse/0", exist_ok=True)
    traj = np.loadtxt(f"{rep}/traj.txt").reshape(-1, 4, 4)
    frames = sorted(glob.glob(f"{rep}/results/frame*.jpg")); deps = sorted(glob.glob(f"{rep}/results/depth*.png"))
    idx = list(range(0, len(frames), stride))
    with open(f"{out}/sparse/0/cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    lines = []
    for k, i in enumerate(idx, 1):
        w2c = np.linalg.inv(traj[i].astype(np.float64)); R = w2c[:3, :3]; t = w2c[:3, 3]
        q = Rotation.from_matrix(R).as_quat()            # x,y,z,w
        name = f"frame{i:06d}.jpg"; shutil.copy(frames[i], f"{out}/images/{name}")
        lines.append(f"{k} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} 1 {name}\n\n")
    open(f"{out}/sparse/0/images.txt", "w").writelines(lines)
    # init point cloud: back-project GT depth, subsample
    pts, cols = [], []
    for i in idx[::2]:
        d = iio.imread(deps[i]).astype(np.float32) / 6553.5
        rgb = np.asarray(Image.open(frames[i]).convert("RGB"), np.float32)
        ys = np.arange(0, H, 12); xs = np.arange(0, W, 12); gx, gy = np.meshgrid(xs, ys)
        z = d[gy, gx].reshape(-1); m = (z > 0.1) & (z < 8)
        X = (gx.reshape(-1) - cx) / fx * z; Y = (gy.reshape(-1) - cy) / fy * z
        wp = (traj[i] @ np.stack([X, Y, z, np.ones_like(z)], 1).T).T[:, :3]
        pts.append(wp[m]); cols.append(rgb[gy, gx].reshape(-1, 3)[m])
    P = np.concatenate(pts, 0); C = np.concatenate(cols, 0)
    sel = np.random.choice(len(P), min(60000, len(P)), replace=False); P = P[sel]; C = C[sel].astype(int)
    with open(f"{out}/sparse/0/points3D.txt", "w") as f:
        for j, (p, c) in enumerate(zip(P, C), 1):
            f.write(f"{j} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 1.0\n")
    # PGSR reads the model from sparse/ (not sparse/0/)
    for fn in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy(f"{out}/sparse/0/{fn}", f"{out}/sparse/{fn}")
    print(f"COLMAP {os.path.basename(rep)}: {len(idx)} images, {len(P)} init points -> {out}")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
