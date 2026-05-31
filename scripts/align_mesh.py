"""Rotate a reconstructed mesh to be Y-up (gravity-aligned) for Habitat.

MapAnything's world frame is arbitrarily oriented, so the floor isn't horizontal
and Habitat's Recast navmesh fails. We recover "up" from the camera poses: the
photographer held the phone roughly upright, so the mean camera up-vector ≈
world up. We rotate the mesh so that direction maps to +Y and drop the floor to y≈0.

    python scripts/align_mesh.py --mesh room_mesh.glb --cameras cameras.json --out room_aligned.glb
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import open3d as o3d


def rotation_aligning(a, b):
    """Rotation matrix mapping unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:               # opposite: 180° about any perpendicular axis
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, perp); axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cams = json.load(open(args.cameras))["views"]
    ups = []
    for v in cams:
        c2w = np.array(v["c2w"], dtype=np.float64)
        ups.append(-c2w[:3, 1])      # OpenCV: world-up ≈ -Y_cam axis
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    print(f"estimated gravity-up (world): {np.round(up,3)}", flush=True)

    R = rotation_aligning(up, np.array([0.0, 1.0, 0.0]))
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.rotate(R, center=(0, 0, 0))
    # drop floor to y≈0
    v = np.asarray(mesh.vertices)
    v[:, 1] -= np.percentile(v[:, 1], 1.0)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()

    lo = v.min(0); hi = v.max(0)
    print(f"after align: y-range [{lo[1]:.2f}, {hi[1]:.2f}]  xz extent {hi[0]-lo[0]:.2f} x {hi[2]-lo[2]:.2f}", flush=True)
    o3d.io.write_triangle_mesh(args.out, mesh)
    o3d.io.write_triangle_mesh(args.out.rsplit(".", 1)[0] + ".obj", mesh)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
