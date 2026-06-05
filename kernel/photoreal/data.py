"""Posed-frame dataset for photoreal-mesh training (Replica-style or generic folder).

A scene provides: a list of (image[H,W,3] float 0-1, pose_c2w[4,4], K=(fx,fy,cx,cy)).
Train/test split = held-out every Nth frame (NerfBaselines convention) so NVS is
evaluated on views the appearance optimisation never saw.
"""
from __future__ import annotations
import glob, os
import numpy as np
from PIL import Image


REPLICA_DEPTH_SCALE = 6553.5


def _m8(x):
    """nvdiffrast's CUDA rasterizer requires resolution divisible by 8."""
    return max(8, int(round(x / 8.0)) * 8)


def load_replica(scene_dir, scale=0.5, stride=1, holdout=8):
    """Replica scene dir (.../room0) with results/frame*.jpg + traj.txt.
    Returns dict with train/test lists of (rgb, pose, K) and W,H."""
    W0, H0 = 1200, 680
    W, H = _m8(W0 * scale), _m8(H0 * scale)
    sx, sy = W / W0, H / H0                       # anisotropic: keep intrinsics consistent
    K = (600 * sx, 600 * sy, 599.5 * sx, 339.5 * sy)
    traj = np.loadtxt(f"{scene_dir}/traj.txt").reshape(-1, 4, 4)
    rgbs = sorted(glob.glob(f"{scene_dir}/results/frame*.jpg"))
    idx = list(range(0, len(rgbs), stride))
    train, test = [], []
    for k, i in enumerate(idx):
        rgb = np.asarray(Image.open(rgbs[i]).resize((W, H), Image.BILINEAR))[:, :, :3].astype(np.float32) / 255.0
        item = (rgb, traj[i].astype(np.float32), K)
        (test if (k % holdout == 0) else train).append(item)
    return dict(train=train, test=test, W=W, H=H, K=K)


def load_folder(img_dir, poses_npy, K, scale=1.0, holdout=8):
    """Generic: a folder of images + an [N,4,4] cam->world poses .npy + K=(fx,fy,cx,cy).
    K is given at full image resolution; scaled here by `scale`."""
    poses = np.load(poses_npy).astype(np.float32)
    paths = sorted(glob.glob(f"{img_dir}/*.jpg") + glob.glob(f"{img_dir}/*.png"))
    assert len(paths) == len(poses), "image/pose count mismatch"
    im0 = Image.open(paths[0]); W0, H0 = im0.size
    W, H = _m8(W0 * scale), _m8(H0 * scale)       # multiple of 8 for nvdiffrast
    sx, sy = W / W0, H / H0
    fx, fy, cx, cy = K
    Ks = (fx * sx, fy * sy, cx * sx, cy * sy)
    train, test = [], []
    for k, (p, pose) in enumerate(zip(paths, poses)):
        rgb = np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR)).astype(np.float32) / 255.0
        (test if (k % holdout == 0) else train).append((rgb, pose, Ks))
    return dict(train=train, test=test, W=W, H=H, K=Ks)
