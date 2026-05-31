#!/usr/bin/env python3
"""
Prepare a reconstructed room mesh as a Genesis static collider.

Steps:
  1. Load the DN-Splatter Replica room0 TSDF mesh (metric, real reconstruction).
  2. Find the floor plane via RANSAC, restricted to near-horizontal planes at the
     bottom of the scene (the up-axis is the mesh's smallest-extent axis = Z here).
  3. Build a rotation that maps the floor normal -> +Z, translate so the floor
     sits at z = 0. This gravity-aligns the mesh (floor horizontal, room above).
  4. Export OBJ (+ a decimated OBJ) for Genesis ingestion.

Run with the `splat`/`genesis` env python (open3d 0.19 available there).
"""
import argparse
import numpy as np
import open3d as o3d


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def rotation_aligning(a, b):
    """Rotation matrix R such that R @ a == b (both unit vectors)."""
    a = unit(a); b = unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        # parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180 deg: pick any orthogonal axis
        ortho = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = unit(np.cross(a, ortho))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/cs/student/projects3/2023/dkozlov/work/out/dn_room0/mesh/Open3dTSDFfusion_mesh.ply")
    ap.add_argument("--out", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned.obj")
    ap.add_argument("--out_decim", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned_decim.obj")
    ap.add_argument("--decim_tris", type=int, default=200000)
    args = ap.parse_args()

    m = o3d.io.read_triangle_mesh(args.inp)
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    v = np.asarray(m.vertices)
    ext = v.max(0) - v.min(0)
    up_axis = int(np.argmin(ext))  # smallest extent = vertical for a room
    print(f"[prep] verts={len(v)} tris={len(np.asarray(m.triangles))}")
    print(f"[prep] extent={ext}  guessed up-axis={'xyz'[up_axis]}")

    # RANSAC plane fit on the lower portion of the up-axis (the floor region).
    lo = v[:, up_axis].min()
    hi = v[:, up_axis].max()
    floor_band = lo + 0.25 * (hi - lo)
    pcd = o3d.geometry.PointCloud()
    floor_pts = v[v[:, up_axis] < floor_band]
    if len(floor_pts) < 1000:
        floor_pts = v
    pcd.points = o3d.utility.Vector3dVector(floor_pts)
    plane, inliers = pcd.segment_plane(distance_threshold=0.03,
                                       ransac_n=3, num_iterations=2000)
    a, b, c, d = plane
    normal = unit(np.array([a, b, c]))
    # Orient normal to point "up" along the guessed up-axis (away from below).
    if normal[up_axis] < 0:
        normal = -normal
        d = -d
    print(f"[prep] floor plane normal={normal}  d={d:.4f}  inliers={len(inliers)}")

    # Rotate floor normal -> +Z.
    R = rotation_aligning(normal, np.array([0.0, 0.0, 1.0]))
    v_rot = v @ R.T
    # Translate so floor sits at z=0 (use the rotated floor inlier centroid).
    floor_world = floor_pts[inliers] @ R.T
    z_floor = np.median(floor_world[:, 2])
    v_rot[:, 2] -= z_floor

    m.vertices = o3d.utility.Vector3dVector(v_rot)
    m.compute_vertex_normals()

    vmin = v_rot.min(0); vmax = v_rot.max(0)
    print(f"[prep] after align: z-range=[{vmin[2]:.3f}, {vmax[2]:.3f}]  "
          f"floor near z=0, ceiling near z={vmax[2]:.2f}")
    print(f"[prep] xy footprint: x[{vmin[0]:.2f},{vmax[0]:.2f}] y[{vmin[1]:.2f},{vmax[1]:.2f}]")
    print(f"[prep] suggested sphere drop center: x={(vmin[0]+vmax[0])/2:.2f} "
          f"y={(vmin[1]+vmax[1])/2:.2f} z=1.0")

    o3d.io.write_triangle_mesh(args.out, m)
    print(f"[prep] wrote full mesh -> {args.out}")

    md = m.simplify_quadric_decimation(args.decim_tris)
    md.compute_vertex_normals()
    o3d.io.write_triangle_mesh(args.out_decim, md)
    print(f"[prep] wrote decimated ({len(np.asarray(md.triangles))} tris) -> {args.out_decim}")


if __name__ == "__main__":
    main()
