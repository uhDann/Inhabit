"""Render an orbiting turntable of a mesh to PNG frames (Open3D EGL headless).
Used to build side-by-side reconstruction-vs-GT comparison videos.

Usage:
    python render_turntable.py MESH.ply OUT_PREFIX [--frames 72] [--elev 28] [--up auto|x|y|z]
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_prefix")
    ap.add_argument("--frames", type=int, default=72)
    ap.add_argument("-W", type=int, default=720)
    ap.add_argument("-H", type=int, default=720)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--radius-mult", type=float, default=1.7)
    ap.add_argument("--up", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--neutral", type=float, nargs=3, default=[0.80, 0.80, 0.83])
    ap.add_argument("--points", action="store_true",
                    help="render the mesh's vertices as a point cloud (for GT PLYs whose faces Open3D won't read)")
    ap.add_argument("--point-size", type=float, default=4.0)
    a = ap.parse_args()

    import open3d as o3d

    if a.points:
        src = o3d.io.read_triangle_mesh(a.mesh)
        m = o3d.geometry.PointCloud()
        m.points = src.vertices
        if src.has_vertex_colors():
            m.colors = src.vertex_colors
        else:
            m.paint_uniform_color(a.neutral)
    else:
        m = o3d.io.read_triangle_mesh(a.mesh)
        m.compute_vertex_normals()
        if not m.has_vertex_colors():
            m.paint_uniform_color(a.neutral)
    bb = m.get_axis_aligned_bounding_box()
    c = np.asarray(bb.get_center())
    ext = np.asarray(bb.get_extent())

    # vertical axis: 'auto' -> smallest-extent axis (room height < floor dims)
    up_idx = int(np.argmin(ext)) if a.up == "auto" else {"x": 0, "y": 1, "z": 2}[a.up]
    up = np.zeros(3); up[up_idx] = 1.0
    horiz = [i for i in range(3) if i != up_idx]
    r = a.radius_mult * 0.5 * max(ext[horiz[0]], ext[horiz[1]])
    h = np.tan(np.radians(a.elev)) * r

    r_obj = o3d.visualization.rendering.OffscreenRenderer(a.W, a.H)
    r_obj.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    if a.points:
        mat.shader = "defaultUnlit"; mat.point_size = a.point_size
    else:
        mat.shader = "defaultLit"
    r_obj.scene.add_geometry("m", m, mat)
    r_obj.scene.scene.set_sun_light((-up).tolist(), [1, 1, 1], 75000)
    r_obj.scene.scene.enable_sun_light(True)

    for f in range(a.frames):
        ang = 2 * np.pi * f / a.frames
        eye = c.copy().astype(float)
        eye[horiz[0]] += r * np.cos(ang)
        eye[horiz[1]] += r * np.sin(ang)
        eye[up_idx] += h
        r_obj.setup_camera(60.0, c.tolist(), eye.tolist(), up.tolist())
        img = r_obj.render_to_image()
        o3d.io.write_image(f"{a.out_prefix}_{f:03d}.png", img, 9)
    print(f"wrote {a.frames} frames to {a.out_prefix}_*.png  (up-axis={up_idx})", flush=True)
    print("TURNTABLE_DONE", flush=True)


if __name__ == "__main__":
    main()
