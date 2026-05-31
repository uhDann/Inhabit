"""Stage: point cloud -> watertight-ish triangle mesh (Open3D Poisson).

The splat is rendering-only; physics/collision + a Habitat navmesh need a real
surface mesh. We build one from the MapAnything point cloud: voxel-downsample,
estimate+orient normals, Poisson reconstruction, crop low-density (anti-balloon),
keep the largest connected component, decimate, export GLB (+OBJ fallback).

    python -m vid2scene.geometry.mesh --ply recon.ply --out room.glb
"""

from __future__ import annotations

import argparse
import numpy as np
import open3d as o3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True, help="output mesh (.glb / .obj / .ply)")
    ap.add_argument("--poisson-depth", type=int, default=10)
    ap.add_argument("--density-quantile", type=float, default=0.04, help="drop lowest-density verts")
    ap.add_argument("--voxel", type=float, default=0.0, help="downsample voxel size (0 = auto from bbox)")
    ap.add_argument("--target-tris", type=int, default=250_000)
    args = ap.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"loaded {len(pcd.points)} points", flush=True)

    ext = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    diag = float(np.linalg.norm(ext))
    voxel = args.voxel if args.voxel > 0 else diag / 600.0
    pcd = pcd.voxel_down_sample(voxel)
    print(f"downsampled to {len(pcd.points)} points (voxel {voxel:.4f}, scene diag {diag:.2f})", flush=True)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(15)

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, args.density_quantile))
    for fn in (mesh.remove_unreferenced_vertices, mesh.remove_degenerate_triangles,
               mesh.remove_duplicated_triangles, mesh.remove_duplicated_vertices):
        fn()

    # keep only the largest connected component (drops floating fragments)
    idx, counts, _ = mesh.cluster_connected_triangles()
    idx, counts = np.asarray(idx), np.asarray(counts)
    if len(counts):
        mesh.remove_triangles_by_mask(idx != int(counts.argmax()))
        mesh.remove_unreferenced_vertices()

    if args.target_tris and len(mesh.triangles) > args.target_tris:
        mesh = mesh.simplify_quadric_decimation(args.target_tris)
    mesh.compute_vertex_normals()

    ok = o3d.io.write_triangle_mesh(args.out, mesh)
    print(f"wrote {args.out} ok={ok}  verts={len(mesh.vertices)} tris={len(mesh.triangles)}", flush=True)
    # always also drop an OBJ as a robust fallback for importers
    obj = args.out.rsplit(".", 1)[0] + ".obj"
    o3d.io.write_triangle_mesh(obj, mesh)
    print(f"wrote {obj}", flush=True)


if __name__ == "__main__":
    main()
