"""Render a reconstructed mesh along the MuSHRoom iPhone capture trajectory
(interior fly-through), so we get a real-capture reconstruction video like the
Replica fly-throughs.

Reads transformations_colmap.json (nerfstudio-style, OpenGL c2w) + its intrinsics.
Renders the mesh from every Nth posed camera. The mesh is assumed to be in the
COLMAP/OpenCV world frame (PGSR TSDF output); we convert each c2w from OpenGL
camera convention to OpenCV for Open3D's setup_camera (world-to-camera OpenCV).

Usage:
  python render_mesh_mushroom.py MESH.ply LONG_CAPTURE_DIR OUT_PREFIX [--stride 6] [--scale 1.0]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("long_capture")           # .../coffee_room/iphone/long_capture
    ap.add_argument("out_prefix")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)  # render-res fraction of native
    a = ap.parse_args()

    import open3d as o3d

    LC = Path(a.long_capture)
    meta = json.load(open(LC / "transformations_colmap.json"))
    W = int(round(int(meta["w"]) * a.scale))
    H = int(round(int(meta["h"]) * a.scale))
    fx = float(meta["fl_x"]) * a.scale
    fy = float(meta["fl_y"]) * a.scale
    cx = float(meta["cx"]) * a.scale
    cy = float(meta["cy"]) * a.scale
    # Open3D requires cx ~ W/2-0.5; clamp the principal point into the valid window.
    cx = min(max(cx, 0.5), W - 1.5)
    cy = min(max(cy, 0.5), H - 1.5)
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix

    frames = sorted(meta["frames"], key=lambda f: f["file_path"])

    m = o3d.io.read_triangle_mesh(a.mesh)
    m.compute_vertex_normals()

    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
    r.scene.add_geometry("m", m, mat)
    r.scene.scene.set_sun_light([0.3, -1, 0.3], [1, 1, 1], 75000)
    r.scene.scene.enable_sun_light(True)

    n = 0
    for i in range(0, len(frames), a.stride):
        c2w = np.array(frames[i]["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1                      # OpenGL -> OpenCV camera axes
        extrinsic = np.linalg.inv(c2w)          # world-to-camera (OpenCV)
        r.setup_camera(K, extrinsic, W, H)
        o3d.io.write_image(f"{a.out_prefix}_{n:04d}.png", r.render_to_image(), 9)
        n += 1
    print(f"wrote {n} frames ({W}x{H}) to {a.out_prefix}_*.png", flush=True)
    print("MESHTRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
