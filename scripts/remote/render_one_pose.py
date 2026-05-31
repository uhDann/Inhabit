"""Render a mesh (or its vertices as points, for GT) from one Replica-trajectory
camera pose. Used to build a side-by-side qualitative method comparison.

Usage:
    python render_one_pose.py MESH.ply SCENE_DIR FRAME_IDX OUT.png [--points]
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("scene_dir"); ap.add_argument("idx", type=int)
    ap.add_argument("out")
    ap.add_argument("--points", action="store_true")
    ap.add_argument("--scale", type=float, default=0.6)
    a = ap.parse_args()

    import open3d as o3d
    W, H = int(1200 * a.scale), int(680 * a.scale)
    fx = fy = 600.0 * a.scale; cx, cy = 599.5 * a.scale, 339.5 * a.scale
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix

    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    if a.points:
        src = o3d.io.read_triangle_mesh(a.mesh)
        pc = o3d.geometry.PointCloud(); pc.points = src.vertices
        if src.has_vertex_colors(): pc.colors = src.vertex_colors
        mat.shader = "defaultUnlit"; mat.point_size = 4.0
        r.scene.add_geometry("g", pc, mat)
    else:
        m = o3d.io.read_triangle_mesh(a.mesh); m.compute_vertex_normals()
        mat.shader = "defaultLit"
        r.scene.add_geometry("g", m, mat)
        r.scene.scene.set_sun_light([0, 1, 0], [1, 1, 1], 60000)
        r.scene.scene.enable_sun_light(True)

    traj = np.loadtxt(f"{a.scene_dir}/traj.txt").reshape(-1, 4, 4)
    r.setup_camera(K, np.linalg.inv(traj[a.idx]), W, H)
    o3d.io.write_image(a.out, r.render_to_image(), 9)
    print("ONEPOSE_DONE", flush=True)


if __name__ == "__main__":
    main()
