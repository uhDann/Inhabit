"""Render a mesh from a Replica camera trajectory (interior fly-through).
Produces the same in-room viewpoints as the input video, so a side-by-side with
the original frames reads as reconstruction-vs-original.

Usage:
    python render_mesh_traj.py MESH.ply SCENE_DIR OUT_PREFIX [--stride 16]
"""
from __future__ import annotations
import argparse, glob
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("scene_dir")               # .../Replica/room0  (has traj.txt)
    ap.add_argument("out_prefix")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.5)  # render at this fraction of native res
    a = ap.parse_args()

    import open3d as o3d

    W, H = int(1200 * a.scale), int(680 * a.scale)
    fx = fy = 600.0 * a.scale
    cx, cy = 599.5 * a.scale, 339.5 * a.scale
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix

    m = o3d.io.read_triangle_mesh(a.mesh)
    m.compute_vertex_normals()
    traj = np.loadtxt(f"{a.scene_dir}/traj.txt").reshape(-1, 4, 4)  # camera-to-world (OpenCV)

    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
    r.scene.add_geometry("m", m, mat)
    r.scene.scene.set_sun_light([0, 1, 0], [1, 1, 1], 60000)
    r.scene.scene.enable_sun_light(True)

    n = 0
    for i in range(0, len(traj), a.stride):
        extrinsic = np.linalg.inv(traj[i])     # world-to-camera (OpenCV)
        r.setup_camera(K, extrinsic, W, H)
        o3d.io.write_image(f"{a.out_prefix}_{n:04d}.png", r.render_to_image(), 9)
        n += 1
    print(f"wrote {n} frames ({W}x{H}) to {a.out_prefix}_*.png", flush=True)
    print("MESHTRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
