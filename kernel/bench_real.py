"""Real-data benchmark: feed-forward predicted depth vs GT depth, both fused by the
from-scratch kernel, scored against the Replica GT mesh.

Isolates the front-end swap: same scene, same poses, same protocol; only the depth
SOURCE changes (Replica GT depth  vs  Depth-Anything-V2-metric predicted depth).
"""
from __future__ import annotations
import argparse, glob, time, os
import numpy as np
import torch
from scipy.spatial import cKDTree

from kernel import InhabitKernel
from bench import metrics, visibility_cull, sample_mesh, run_open3d

REP = "/cs/student/projects3/2023/dkozlov/datasets/replica/Replica"
DEPTH_SCALE = 6553.5


def load_frames(scene, stride, s):
    import imageio.v2 as iio
    from PIL import Image
    d = f"{REP}/{scene}"
    traj = np.loadtxt(f"{d}/traj.txt").reshape(-1, 4, 4)
    rgbs = sorted(glob.glob(f"{d}/results/frame*.jpg"))
    deps = sorted(glob.glob(f"{d}/results/depth*.png"))
    W0, H0 = 1200, 680
    W, H = int(W0 * s), int(H0 * s)
    K = (600.0 * s, 600.0 * s, 599.5 * s, 339.5 * s)
    idx = list(range(0, len(rgbs), stride))
    poses, rgb_list, gtd_list = [], [], []
    for i in idx:
        poses.append(traj[i].astype(np.float32))
        im = np.asarray(Image.open(rgbs[i]).resize((W, H), Image.BILINEAR))[:, :, :3]
        rgb_list.append(im.copy())
        dd = iio.imread(deps[i]).astype(np.float32) / DEPTH_SCALE
        dd = np.asarray(Image.fromarray(dd).resize((W, H), Image.NEAREST), np.float32)
        gtd_list.append(torch.from_numpy(dd))
    return poses, rgb_list, gtd_list, K, W, H


def fuse_kernel(depths, poses, K, bounds, voxel):
    lo, hi = bounds
    ker = InhabitKernel(lo, hi, voxel=voxel, trunc_vox=3.0,
                        device="cuda" if torch.cuda.is_available() else "cpu", robust=True)
    t0 = time.perf_counter()
    for dep, p in zip(depths, poses):
        ker.integrate(dep, torch.from_numpy(p), K)
    v, f, _ = ker.extract_mesh()
    return v, f, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--samples", type=int, default=300_000)
    ap.add_argument("--sources", default="gt,pred")
    a = ap.parse_args()
    import trimesh

    poses, rgbs, gtd, K, W, H = load_frames(a.scene, a.stride, a.scale)
    print(f"{a.scene}: {len(poses)} frames @ {W}x{H}, voxel {a.voxel}")

    gtm = trimesh.load(f"{REP}/{a.scene}_mesh.ply", process=False)
    lo = gtm.bounds[0] - 0.15; hi = gtm.bounds[1] + 0.15
    gt_pts = np.asarray(trimesh.sample.sample_surface(gtm, 500_000, seed=0)[0], np.float32)
    gt_pts = visibility_cull(gt_pts, gtd, poses, K, W, H, tol=0.05)
    print(f"GT points (visibility-culled): {len(gt_pts):,}")

    # predicted depth
    pred = None
    if "pred" in a.sources:
        from front_end import predict_depth
        t0 = time.perf_counter()
        pred = [torch.from_numpy(predict_depth(im)) for im in rgbs]
        print(f"feed-forward depth: {len(pred)} frames in {time.perf_counter()-t0:.1f}s")

    cols = ["pipeline", "Chamfer_cm", "acc_cm", "comp_cm", "F@2cm", "F@5cm", "secs"]
    w = [28, 11, 8, 9, 7, 7, 7]
    print("\n" + "".join(c.ljust(wi) for c, wi in zip(cols, w)))
    print("-" * sum(w))

    def report(name, v, f, dt):
        pp = sample_mesh(v, f, a.samples)
        pp = visibility_cull(pp, gtd, poses, K, W, H, tol=0.05)
        m = metrics(pp, gt_pts)
        vals = [name, f"{m['chamfer_cm']:.2f}", f"{m['acc_cm']:.2f}", f"{m['comp_cm']:.2f}",
                f"{m['F@2cm']:.3f}", f"{m['F@5cm']:.3f}", f"{dt:.2f}"]
        print("".join(str(x).ljust(wi) for x, wi in zip(vals, w)))

    trunc = 4.0 * a.voxel
    if "gt" in a.sources:
        v, f, dt = run_open3d(gtd, poses, K, W, H, a.voxel, trunc)
        report("Open3D + GT depth", v, f, dt)
        v, f, dt = fuse_kernel(gtd, poses, K, (lo, hi), a.voxel)
        report("ours TSDF kernel + GT depth", v, f, dt)
        from surfel import backproject, poisson_mesh
        import time as _t
        t0 = _t.perf_counter()
        P, Nrm, _ = backproject(gtd, poses, K, sub=0.35, device="cuda")
        v, f = poisson_mesh(P.cpu().numpy(), Nrm.cpu().numpy())
        report("ours surfel + GT depth", v, f, _t.perf_counter() - t0)
    if pred is not None:
        v, f, dt = fuse_kernel(pred, poses, K, (lo, hi), a.voxel)
        report("ours kernel + feed-forward depth", v, f, dt)


if __name__ == "__main__":
    main()
