"""Benchmark: inhabit-kernel (from scratch, GPU) vs our current Open3D TSDF.

Same synthetic room, same posed depth (clean or noisy), same iso-surface mesher
and same visibility-culled Chamfer/F-score protocol we use in the real pipeline.
Reports quality (Chamfer-L1 cm, F-score@2cm/5cm) and speed (fusion+extract wall-clock).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
from scipy.spatial import cKDTree

import scene as S
from kernel import InhabitKernel


def add_noise(depth, frac, dropout, gen):
    d = depth.clone()
    m = d > 0
    sigma = frac * d
    d = d + torch.randn(d.shape, generator=gen).to(d) * sigma * m
    if dropout > 0:
        drop = (torch.rand(d.shape, generator=gen).to(d) < dropout) & m
        d[drop] = 0.0
    d[~m] = 0.0
    return d.clamp(min=0)


def sample_mesh(verts, faces, n, seed=0):
    import trimesh
    if len(faces) == 0:
        return np.zeros((0, 3), np.float32)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, np.float32)


def metrics(pred_pts, gt_pts, taus=(0.02, 0.05)):
    if len(pred_pts) == 0:
        return dict(chamfer_cm=float("nan"), acc_cm=float("nan"), comp_cm=float("nan"),
                    **{f"F@{int(t*100)}cm": 0.0 for t in taus})
    tg = cKDTree(gt_pts); tp = cKDTree(pred_pts)
    d_p2g, _ = tg.query(pred_pts, workers=-1)     # accuracy
    d_g2p, _ = tp.query(gt_pts, workers=-1)       # completion
    acc, comp = float(d_p2g.mean()), float(d_g2p.mean())
    out = dict(chamfer_cm=100 * 0.5 * (acc + comp), acc_cm=100 * acc, comp_cm=100 * comp)
    for t in taus:
        prec = float((d_p2g < t).mean()); rec = float((d_g2p < t).mean())
        out[f"F@{int(t*100)}cm"] = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return out


def visibility_cull(gt_pts, depths, poses, K, W, H, tol=0.03):
    """Keep GT points seen as the first surface by >=1 camera (standard protocol)."""
    fx, fy, cx, cy = K
    P = torch.from_numpy(gt_pts)
    seen = torch.zeros(P.shape[0], dtype=torch.bool)
    for dep, pose in zip(depths, poses):
        T = torch.from_numpy(pose)
        R = T[:3, :3]; t = T[:3, 3]
        Xc = (P - t) @ R                          # world->cam (row vectors)
        zc = Xc[:, 2]
        u = (fx * Xc[:, 0] / zc + cx).round().long()
        v = (fy * Xc[:, 1] / zc + cy).round().long()
        ok = (zc > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uu = u.clamp(0, W - 1); vv = v.clamp(0, H - 1)
        dm = dep[vv, uu]
        hit = ok & (dm > 0) & ((zc - dm).abs() < tol)
        seen |= hit
    return gt_pts[seen.numpy()]


def run_open3d(depths, poses, K, W, H, voxel, trunc):
    import open3d as o3d
    fx, fy, cx, cy = K
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    t0 = time.perf_counter()
    for dep, pose in zip(depths, poses):
        d = (dep.cpu().numpy()).astype(np.float32)
        col = np.zeros((H, W, 3), np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(col), o3d.geometry.Image(d),
            depth_scale=1.0, depth_trunc=20.0, convert_rgb_to_intensity=False)
        extr = np.linalg.inv(pose).astype(np.float64)
        vol.integrate(rgbd, intr, extr)
    mesh = vol.extract_triangle_mesh()
    dt = time.perf_counter() - t0
    v = np.asarray(mesh.vertices, np.float32); f = np.asarray(mesh.triangles, np.int64)
    return v, f, dt


def run_kernel(depths, poses, K, voxel, dev, robust):
    pad = 0.15
    ker = InhabitKernel([-pad, -pad, -pad], [4.0 + pad, 2.6 + pad, 5.0 + pad],
                        voxel=voxel, trunc_vox=3.0, device=dev, robust=robust)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for dep, pose in zip(depths, poses):
        ker.integrate(dep, torch.from_numpy(pose), K)
    if dev == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    v, f, _ = ker.extract_mesh()
    if dev == "cuda":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    run_kernel.last = (t1 - t0, t2 - t1)                  # (fuse, extract)
    return v, f, t2 - t0


def decompose_report(sc, depths_clean, objids, poses, K, voxel, dev, W, H, samples):
    """Phase 1: fuse with object votes, split the mesh into a room shell + one mesh
    per object, and score each separately against its own ground truth."""
    import trimesh
    pad = 0.15
    ker = InhabitKernel([-pad, -pad, -pad], [4.0 + pad, 2.6 + pad, 5.0 + pad],
                        voxel=voxel, trunc_vox=3.0, device=dev, robust=True, n_labels=4)
    for dep, pose, obj in zip(depths_clean, poses, objids):
        ker.integrate(dep, torch.from_numpy(pose), K, labels=obj)
    verts, faces, vlab = ker.extract_mesh()
    gtP, gtL = sc.gt_points(n=500_000)
    lf = vlab[faces]
    a, b, c = lf[:, 0], lf[:, 1], lf[:, 2]
    face_lab = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))

    names = {0: "room shell", 1: "sphere", 2: "inner box"}
    cols = ["object", "verts", "Chamfer_cm", "F@2cm", "watertight", "vol_L", "mass_kg"]
    w = [12, 8, 11, 8, 11, 8, 8]
    print("\n=== Phase 1+2: separable per-object meshes, made physics-ready ===")
    print("(per-object accuracy vs its own GT; then filled to watertight + mass props "
          "at 400 kg/m^3)")
    print("".join(c.ljust(wi) for c, wi in zip(cols, w)))
    print("-" * sum(w))
    for oid, name in names.items():
        fm = face_lab == oid
        sub_f = faces[fm]
        if len(sub_f) == 0:
            print(f"{name.ljust(12)}{'0'.ljust(8)}(no surface)"); continue
        used = np.unique(sub_f)
        remap = {int(o): i for i, o in enumerate(used)}
        nf = np.vectorize(remap.get)(sub_f)
        sm = trimesh.Trimesh(vertices=verts[used], faces=nf, process=False)
        pp = np.asarray(trimesh.sample.sample_surface(sm, samples, seed=0)[0], np.float32)
        pp = visibility_cull(pp, depths_clean, poses, K, W, H)
        gt_o = visibility_cull(gtP[gtL == oid], depths_clean, poses, K, W, H)
        m = metrics(pp, gt_o)
        # Phase 2: physics-ready. The room is the static environment mesh; movable
        # objects get a convex-hull collision proxy (watertight -> mass/inertia).
        if oid == 0:
            wt_s, vol_s, mass_s = "static env", "-", "-"
        else:
            hull = sm.convex_hull
            vol = abs(float(hull.volume))
            wt_s = "yes (hull)" if hull.is_watertight else "no"
            vol_s = f"{vol * 1000:.1f}"; mass_s = f"{vol * 400:.1f}"   # L, kg @ 400 kg/m^3
        vals = [name, len(used), f"{m['chamfer_cm']:.2f}", f"{m['F@2cm']:.3f}",
                wt_s, vol_s, mass_s]
        print("".join(str(x).ljust(wi) for x, wi in zip(vals, w)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--W", type=int, default=256); ap.add_argument("--H", type=int, default=192)
    ap.add_argument("--ncam", type=int, default=48)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--samples", type=int, default=200_000)
    args = ap.parse_args()
    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"device={dev}  res={args.W}x{args.H}  ncam={args.ncam}  voxel={args.voxel}")

    sc = S.Scene(device=dev)
    K = S.intrinsics(args.W, args.H)
    poses = S.ring_poses(sc, n=args.ncam)

    depths_clean, objids = [], []
    for pose in poses:
        dep, obj = sc.render(torch.from_numpy(pose).to(dev), K, args.W, args.H)
        depths_clean.append(dep.cpu()); objids.append(obj.cpu())

    gt, _ = sc.gt_points(n=500_000)
    gt = visibility_cull(gt, depths_clean, poses, K, args.W, args.H)
    print(f"GT points (visibility-culled): {len(gt):,}\n")

    trunc = 4.0 * args.voxel
    sweep = [(0.0, 0.0), (0.02, 0.03), (0.05, 0.08)]   # (depth-noise frac, dropout)

    cols = ["noise", "method", "chamfer_cm", "acc_cm", "comp_cm", "F@2cm", "F@5cm", "secs"]
    w = [8, 22, 11, 8, 9, 7, 7, 7]
    print("".join(c.ljust(wi) for c, wi in zip(cols, w)))
    print("-" * sum(w))

    def row(tag, name, v, f, dt):
        pp = sample_mesh(v, f, args.samples)
        pp = visibility_cull(pp, depths_clean, poses, K, args.W, args.H)
        m = metrics(pp, gt)
        vals = [tag, name, f"{m['chamfer_cm']:.2f}", f"{m['acc_cm']:.2f}", f"{m['comp_cm']:.2f}",
                f"{m['F@2cm']:.3f}", f"{m['F@5cm']:.3f}", f"{dt:.2f}"]
        print("".join(str(x).ljust(wi) for x, wi in zip(vals, w)))
        return m

    gen = torch.Generator().manual_seed(0)
    for nf, dp in sweep:
        tag = "clean" if nf == 0 else f"{int(nf*100)}%"
        depths = [add_noise(d, nf, dp, gen) for d in depths_clean]
        try:
            v, f, dt = run_open3d(depths, poses, K, args.W, args.H, args.voxel, trunc)
            row(tag, "Open3D TSDF (current)", v, f, dt)
        except Exception as e:
            print(f"{tag}  open3d failed: {e}")
        v, f, dt = run_kernel(depths, poses, K, args.voxel, dev, robust=False)
        row(tag, "ours (uniform)", v, f, dt)
        v, f, dt = run_kernel(depths, poses, K, args.voxel, dev, robust=True)
        row(tag, "ours (robust kernel)", v, f, dt)
        print(f"         [ours fuse {run_kernel.last[0]:.2f}s + MC extract {run_kernel.last[1]:.2f}s]")
        print()

    decompose_report(sc, depths_clean, objids, poses, K, args.voxel, dev,
                     args.W, args.H, args.samples)


if __name__ == "__main__":
    main()
