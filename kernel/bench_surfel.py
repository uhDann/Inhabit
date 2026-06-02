"""Does the surfel core close the clean-precision gap that the voxel TSDF can't?
Compares Open3D TSDF, our TSDF kernel, and our surfel core (back-project + Poisson,
and + differentiable refinement) on the synthetic scene, clean and noisy.
"""
from __future__ import annotations
import time
import numpy as np
import torch

import scene as S
from kernel import InhabitKernel
from bench import metrics, visibility_cull, sample_mesh, run_open3d, add_noise
from surfel import backproject, refine, poisson_mesh


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H, ncam, voxel = 256, 192, 48, 0.02
    sc = S.Scene(device=dev); K = S.intrinsics(W, H); poses = S.ring_poses(sc, n=ncam)
    depths_clean = [sc.render(torch.from_numpy(p).to(dev), K, W, H)[0].cpu() for p in poses]
    gt, _ = sc.gt_points(n=500_000)
    gt = visibility_cull(gt, depths_clean, poses, K, W, H)
    print(f"GT points: {len(gt):,}")
    trunc = 4 * voxel
    pad = 0.15
    cols = ["noise", "method", "Chamfer", "F@2cm", "F@5cm", "secs"]
    w = [7, 26, 9, 7, 7, 7]
    print("".join(c.ljust(wi) for c, wi in zip(cols, w))); print("-" * sum(w))

    def row(tag, name, v, f, dt):
        pp = sample_mesh(v, f, 200_000)
        pp = visibility_cull(pp, depths_clean, poses, K, W, H)
        m = metrics(pp, gt)
        print("".join(str(x).ljust(wi) for x, wi in zip(
            [tag, name, f"{m['chamfer_cm']:.2f}", f"{m['F@2cm']:.3f}",
             f"{m['F@5cm']:.3f}", f"{dt:.2f}"], w)))

    gen = torch.Generator().manual_seed(0)
    for nf, dp in [(0.0, 0.0), (0.05, 0.08)]:
        tag = "clean" if nf == 0 else f"{int(nf*100)}%"
        depths = [add_noise(d, nf, dp, gen) for d in depths_clean]
        v, f, dt = run_open3d(depths, poses, K, W, H, voxel, trunc)
        row(tag, "Open3D TSDF", v, f, dt)
        ker = InhabitKernel([-pad]*3, [4+pad, 2.6+pad, 5+pad], voxel=voxel, device=dev, robust=True)
        t0 = time.perf_counter()
        for d, p in zip(depths, poses):
            ker.integrate(d, torch.from_numpy(p), K)
        v, f, _ = ker.extract_mesh(); row(tag, "ours TSDF kernel", v, f, time.perf_counter()-t0)
        # surfel: back-project + Poisson (depth-derived normals -> fast)
        t0 = time.perf_counter()
        P, Nrm, src = backproject(depths, poses, K, sub=0.25, device=dev)
        v, f = poisson_mesh(P.cpu().numpy(), Nrm.cpu().numpy())
        row(tag, "ours surfel (Poisson)", v, f, time.perf_counter()-t0)
        # surfel + differentiable refinement
        t0 = time.perf_counter()
        P2 = refine(P, src, depths, poses, K, iters=30, device=dev)
        v, f = poisson_mesh(P2.cpu().numpy(), Nrm.cpu().numpy())
        row(tag, "ours surfel + refine", v, f, time.perf_counter()-t0)
        print()


if __name__ == "__main__":
    main()
