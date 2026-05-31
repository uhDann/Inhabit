"""Render a closed mesh from cameras placed INSIDE it (so we see interior
walls/floor instead of the exterior shell). Tries Open3D's Filament offscreen
renderer first; if Filament fails (headless EGL issues), falls back to a
matplotlib CPU triangle render.

Usage:
    python render_interior.py <mesh.ply> <out_prefix> [--scale-eye 0.30]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np


def make_cams(center, ext):
    """4 interior views (toward each wall), eye placed at a fraction of half-extent
    in one axis so we stand inside the room and look across to the opposite wall."""
    e = ext * 0.45  # how far from center to place the eye (still inside)
    views = []
    views.append(("look_+X", center + np.array([-e[0], 0, 0]), center + np.array([e[0], 0, 0])))
    views.append(("look_-X", center + np.array([ e[0], 0, 0]), center + np.array([-e[0], 0, 0])))
    views.append(("look_+Z", center + np.array([0, 0, -e[2]]), center + np.array([0, 0,  e[2]])))
    views.append(("look_-Z", center + np.array([0, 0,  e[2]]), center + np.array([0, 0, -e[2]])))
    return views


def try_open3d(ply, out_prefix, W, H):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(ply)
    if len(m.triangles) == 0:
        raise RuntimeError("empty mesh")
    m.compute_vertex_normals()
    # paint light grey if the mesh has no vertex colors (MonoSDF mesh is uncolored)
    if not m.has_vertex_colors():
        m.paint_uniform_color([0.78, 0.80, 0.85])
    bb = m.get_axis_aligned_bounding_box()
    c = np.asarray(bb.get_center()); ext = np.asarray(bb.get_extent())
    print(f"bbox center {c.round(3)} extent {ext.round(3)}", flush=True)
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    r.scene.add_geometry("m", m, mat)
    # one light from "up"
    r.scene.scene.set_sun_light([0.0, -1.0, 0.0], [1, 1, 1], 75000)
    r.scene.scene.enable_sun_light(True)
    for name, eye, look in make_cams(c, ext):
        r.setup_camera(75.0, look, eye, np.array([0.0, 1.0, 0.0]))
        img = r.render_to_image()
        out = f"{out_prefix}_{name}.png"
        o3d.io.write_image(out, img, 9)
        print("wrote", out, flush=True)


def try_matplotlib(ply, out_prefix, W, H):
    """CPU fallback. Decimates aggressively and renders the back-facing
    interior surfaces with simple Lambert shading from inside cameras."""
    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    m = o3d.io.read_triangle_mesh(ply)
    m.compute_vertex_normals()
    # decimate hard so matplotlib survives
    target = 120000
    if len(m.triangles) > target:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target)
        print(f"decimated to {len(m.triangles)} tris", flush=True)
    V = np.asarray(m.vertices); T = np.asarray(m.triangles); N = np.asarray(m.vertex_normals)
    bb = m.get_axis_aligned_bounding_box()
    c = np.asarray(bb.get_center()); ext = np.asarray(bb.get_extent())
    print(f"bbox center {c.round(3)} extent {ext.round(3)}", flush=True)
    for name, eye, look in make_cams(c, ext):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        tri = V[T]
        # backface cull from interior cam
        f_normals = N[T].mean(1)
        view_dir = (look - eye); view_dir /= np.linalg.norm(view_dir) + 1e-9
        keep = (f_normals @ view_dir) > 0  # only faces pointing AWAY from cam (interior surfaces toward us)
        tri = tri[keep]
        shade = np.clip(f_normals[keep] @ -view_dir, 0.15, 1.0)
        colors = np.stack([shade * 0.75, shade * 0.78, shade * 0.85, np.ones_like(shade)], 1)
        ax.add_collection3d(Poly3DCollection(tri, facecolors=colors, edgecolors="none", linewidths=0))
        # set the view from the interior camera
        ax.set_xlim(c[0] - ext[0] / 2, c[0] + ext[0] / 2)
        ax.set_ylim(c[1] - ext[1] / 2, c[1] + ext[1] / 2)
        ax.set_zlim(c[2] - ext[2] / 2, c[2] + ext[2] / 2)
        ax.set_axis_off()
        ax.set_box_aspect((ext[0], ext[1], ext[2]))
        # convert eye->look into elev/azim for matplotlib (rough)
        d = eye - c
        ax.azim = np.degrees(np.arctan2(d[2], d[0]))
        ax.elev = -np.degrees(np.arcsin(d[1] / (np.linalg.norm(d) + 1e-9)))
        out = f"{out_prefix}_{name}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig)
        print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply")
    ap.add_argument("out_prefix")
    ap.add_argument("-W", type=int, default=1280)
    ap.add_argument("-H", type=int, default=720)
    args = ap.parse_args()
    try:
        try_open3d(args.ply, args.out_prefix, args.W, args.H)
        print("OPEN3D_OK", flush=True)
    except Exception as e:
        print(f"open3d failed ({e!r}); falling back to matplotlib", flush=True)
        try_matplotlib(args.ply, args.out_prefix, args.W, args.H)
        print("MPL_OK", flush=True)


if __name__ == "__main__":
    main()
