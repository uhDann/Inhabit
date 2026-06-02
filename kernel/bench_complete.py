"""Benchmark amodal completion: convex-hull (front-shell) vs visual-hull vs GT,
per object, on the synthetic scene where true volumes are known.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import trimesh

import scene as S
from kernel import InhabitKernel
from bench import metrics, visibility_cull, sample_mesh
from complete import poisson_close

GT_VOL = {1: 4.0 / 3.0 * math.pi * 0.5 ** 3,     # sphere r=0.5
          2: 1.0 * 0.9 * 0.8}                    # box 1.0 x 0.9 x 0.8
NAMES = {1: "sphere", 2: "inner_box"}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W, H, ncam, voxel = 256, 192, 48, 0.02
    sc = S.Scene(device=dev); K = S.intrinsics(W, H); poses = S.ring_poses(sc, n=ncam)
    depths, objids = [], []
    for p in poses:
        d, o = sc.render(torch.from_numpy(p).to(dev), K, W, H)
        depths.append(d.cpu()); objids.append(o.cpu())

    pad = 0.15
    ker = InhabitKernel([-pad, -pad, -pad], [4.0 + pad, 2.6 + pad, 5.0 + pad],
                        voxel=voxel, trunc_vox=3.0, device=dev, robust=True, n_labels=4)
    for d, p, o in zip(depths, poses, objids):
        ker.integrate(d, torch.from_numpy(p), K, labels=o)
    verts, faces, vlab = ker.extract_mesh()
    gtP, gtL = sc.gt_points(n=500_000)

    lf = vlab[faces]; a, b, c = lf[:, 0], lf[:, 1], lf[:, 2]
    face_lab = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))

    cols = ["object", "GT_vol_L", "convex_L", "convex_err", "poisson_L", "poisson_err",
            "poisson_Chamfer", "F@2cm"]
    w = [11, 9, 9, 11, 10, 12, 16, 7]
    print("=== Amodal completion: convex hull (front-shell) vs Poisson closure ===")
    print("".join(x.ljust(wi) for x, wi in zip(cols, w)))
    print("-" * sum(w))
    for oid in (1, 2):
        gtv = GT_VOL[oid] * 1000.0
        # convex hull of the fused front-shell
        sub = faces[face_lab == oid]
        used = np.unique(sub); remap = {int(o): i for i, o in enumerate(used)}
        sm = trimesh.Trimesh(vertices=verts[used],
                             faces=np.vectorize(remap.get)(sub), process=False)
        cvol = abs(float(sm.convex_hull.volume)) * 1000.0
        # Poisson closure of the observed object surface
        opts = np.asarray(trimesh.sample.sample_surface(sm, 60_000, seed=0)[0], np.float32)
        vv, vf, vol = poisson_close(opts)
        vvol = vol * 1000.0
        pp = sample_mesh(vv, vf, 200_000)
        pp = visibility_cull(pp, depths, poses, K, W, H)
        gt_o = visibility_cull(gtP[gtL == oid], depths, poses, K, W, H)
        m = metrics(pp, gt_o)
        vals = [NAMES[oid], f"{gtv:.0f}", f"{cvol:.0f}", f"{100*(cvol-gtv)/gtv:+.0f}%",
                f"{vvol:.0f}", f"{100*(vvol-gtv)/gtv:+.0f}%",
                f"{m['chamfer_cm']:.2f}", f"{m['F@2cm']:.3f}"]
        print("".join(str(x).ljust(wi) for x, wi in zip(vals, w)))


if __name__ == "__main__":
    main()
