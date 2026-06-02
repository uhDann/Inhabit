"""Amodal completion by from-scratch visual hull / space carving.

A separated object mesh from fusion is a front-shell (only the observed surface), so
its convex hull under-counts volume and its centroid is biased. The visual hull is
the maximal solid consistent with every camera's object silhouette: a voxel is solid
if it projects inside the object's 2D mask in (almost) every view that sees it. For
convex / multi-view-observed objects this recovers the full watertight solid, the
unseen back included, with the correct volume -- no generative prior needed.

This is the geometry-faithful baseline completion; a generative completer
(DP-Recon / Amodal3R) would handle concave/occluded objects beyond the silhouette.
"""
from __future__ import annotations
import numpy as np
import torch


def poisson_close(points, depth=8, trim=0.04):
    """Watertight closure of an observed object surface via screened Poisson.
    A full camera ring observes most of an object's surface, so closing the small
    remaining gaps recovers an accurate watertight solid (unlike a convex hull, which
    bridges concavities). Returns (verts, faces, volume_m3)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, np.float64))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(24))
    pcd.orient_normals_consistent_tangent_plane(24)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, trim))
    mesh.remove_degenerate_triangles(); mesh.remove_unreferenced_vertices()
    V = np.asarray(mesh.vertices, np.float32); F = np.asarray(mesh.triangles, np.int64)
    if mesh.is_watertight():
        vol = abs(float(mesh.get_volume()))
    else:                                          # trimmed surface is open: bound the
        import trimesh                             # volume by the convex hull of the
        vol = abs(float(trimesh.Trimesh(V, F, process=False).convex_hull.volume))
    return V, F, vol


@torch.no_grad()
def visual_hull(kernel, objids, poses, K, oid, sil_frac=0.85, min_views=2,
                dilate=1, chunk=4_000_000):
    """Return a watertight per-object mesh (verts, faces) and its volume (m^3)."""
    dev = kernel.device
    fx, fy, cx, cy = K
    fov = torch.zeros(kernel.N, device=dev)
    sil = torch.zeros(kernel.N, device=dev)
    # precompute dilated silhouette masks
    masks = []
    for obj in objids:
        m = (obj == oid).to(dev).float()[None, None]
        if dilate > 0:
            m = torch.nn.functional.max_pool2d(m, 2 * dilate + 1, 1, dilate)
        masks.append(m[0, 0] > 0.5)
    nx, ny, nz = kernel.dims
    org = kernel.origin.to(dev); vx = kernel.voxel
    for s in range(0, kernel.N, chunk):
        lin = torch.arange(s, min(s + chunk, kernel.N), device=dev)
        ii = lin // (ny * nz); jj = (lin // nz) % ny; kk = lin % nz
        X = (torch.stack([ii, jj, kk], 1).float() + 0.5) * vx + org
        for obj, pose, mk in zip(objids, poses, masks):
            P = torch.from_numpy(np.asarray(pose, np.float32)).to(dev)
            R = P[:3, :3]; t = P[:3, 3]
            Xc = (X - t) @ R
            zc = Xc[:, 2]
            front = zc > 1e-4
            H, W = mk.shape
            u = (fx * Xc[:, 0] / zc + cx).round().long()
            v = (fy * Xc[:, 1] / zc + cy).round().long()
            inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            uu = u.clamp(0, W - 1); vv = v.clamp(0, H - 1)
            hit = mk[vv, uu] & inb
            fov[s:s + chunk] += inb.float()
            sil[s:s + chunk] += hit.float()
    occ = (fov >= min_views) & (sil >= sil_frac * fov.clamp(min=1))
    field = torch.where(occ, torch.tensor(-1.0, device=dev), torch.tensor(1.0, device=dev))
    field = field.reshape(kernel.dims)
    field = kernel._denoise(field, bilateral=False)            # smooth the binary edge
    verts, faces, _ = kernel._surface_nets(field)
    vol = float(occ.sum().item()) * (kernel.voxel ** 3)
    return verts.cpu().numpy().astype(np.float32), faces.cpu().numpy().astype(np.int64), vol
