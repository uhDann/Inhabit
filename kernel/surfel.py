"""Surfel core (#12): a non-voxel surface representation.

A TSDF quantises the surface to the voxel grid, which caps fine precision (F@2cm).
Surfels place the surface at the true sub-voxel depth. This module:
  1. back-projects posed depth into an oriented surfel cloud (position + normal +
     confidence) -- no grid, so sub-voxel by construction;
  2. (optionally) differentiably refines surfel positions for multi-view consistency
     so depth noise is averaged out at sub-voxel scale (the surfel analogue of the
     kernel's robust TSDF weighting);
  3. extracts a watertight mesh by screened Poisson over the oriented surfels.

Frozen-model-free, from scratch in PyTorch + Open3D's Poisson solver for meshing.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def backproject(depths, poses, K, confs=None, sub=1.0, device="cuda"):
    """World-space surfel positions [M,3], camera-oriented normals [M,3], and a
    per-point camera id [M]. Normals come from the depth-map cross product (already
    oriented toward the camera), so no slow normal-orientation pass is needed."""
    fx, fy, cx, cy = K
    P, Nn, src = [], [], []
    for ci, (dep, pose) in enumerate(zip(depths, poses)):
        dep = dep.to(device)
        H, W = dep.shape
        R = torch.as_tensor(pose[:3, :3], dtype=torch.float32, device=device)
        t = torch.as_tensor(pose[:3, 3], dtype=torch.float32, device=device)
        vv, uu = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                indexing="ij")
        x = (uu.float() - cx) / fx * dep; y = (vv.float() - cy) / fy * dep
        Xc = torch.stack([x, y, dep], -1)               # [H,W,3] cam coords
        Xw = Xc @ R.T + t                               # [H,W,3] world
        # normal from neighbour differences, oriented toward the camera
        du = Xw[:, 2:] - Xw[:, :-2]; dv = Xw[2:, :] - Xw[:-2, :]
        nrm = torch.zeros_like(Xw)
        nrm[1:-1, 1:-1] = torch.cross(du[1:-1], dv[:, 1:-1], dim=-1)
        view = t - Xw                                   # toward camera
        flip = (nrm * view).sum(-1, keepdim=True) < 0
        nrm = torch.where(flip, -nrm, nrm)
        nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-9)
        valid = (dep > 0) & (nrm.norm(dim=-1) > 0.1)
        valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
        if confs is not None:
            c = confs[ci].to(device); valid = valid & (c > c.median() * 0.3)
        if sub < 1.0:
            valid = valid & (torch.rand(H, W, device=device) < sub)
        P.append(Xw[valid]); Nn.append(nrm[valid])
        src.append(torch.full((int(valid.sum()),), ci, device=device))
    return torch.cat(P, 0), torch.cat(Nn, 0), torch.cat(src, 0)


@torch.no_grad()
def refine(points, src, depths, poses, K, iters=40, lr=0.4, knn=8, device="cuda"):
    """Slide each surfel along its viewing rays toward multi-view-consistent depth,
    averaging out per-frame noise at sub-voxel scale. Plain gradient steps on a
    reprojection-consistency + local-smoothness objective."""
    from scipy.spatial import cKDTree
    fx, fy, cx, cy = K
    Rs = [torch.as_tensor(p[:3, :3], dtype=torch.float32, device=device) for p in poses]
    ts = [torch.as_tensor(p[:3, 3], dtype=torch.float32, device=device) for p in poses]
    dep = [d.to(device) for d in depths]
    X = points.clone()
    # static knn graph for smoothness (recomputed once)
    tree = cKDTree(X.cpu().numpy())
    _, nbr = tree.query(X.cpu().numpy(), k=knn + 1, workers=-1)
    nbr = torch.as_tensor(nbr[:, 1:], device=device)
    for it in range(iters):
        grad = torch.zeros_like(X)
        wsum = torch.zeros(X.shape[0], device=device)
        for ci, (R, t, d) in enumerate(zip(Rs, ts, dep)):
            H, W = d.shape
            Xc = (X - t) @ R                            # world->cam
            zc = Xc[:, 2]
            u = (fx * Xc[:, 0] / zc + cx); v = (fy * Xc[:, 1] / zc + cy)
            inb = (zc > 1e-3) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
            uu = u.clamp(0, W - 1).long(); vv = v.clamp(0, H - 1).long()
            dm = d[vv, uu]
            ok = inb & (dm > 0) & ((dm - zc).abs() < 0.15)
            # move the surfel along the camera ray so its depth matches the measurement
            ray = (R[:, 2])                             # camera viewing axis in world
            delta = (dm - zc)
            grad += ok[:, None].float() * (delta[:, None] * ray[None, :])
            wsum += ok.float()
        step = grad / wsum.clamp(min=1)[:, None]
        X = X + lr * step
        # local smoothing
        X = 0.8 * X + 0.2 * X[nbr].mean(1)
    return X


def poisson_mesh(points, normals=None, depth=9, trim=0.02, normals_knn=16):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    if normals is not None:                            # depth-derived normals: skip the
        pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, np.float64))  # slow
    else:                                              # consistent-orientation pass
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(normals_knn))
        pcd.orient_normals_consistent_tangent_plane(normals_knn)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int64)
