"""Render a mesh from the Habitat agent's recorded navigation path.
Replaces the splat-based agent walkthrough (which is dark/smeary because of
floaters) with a clean mesh render of what the agent sees as it walks.

Maps each Habitat camera pose into the mesh's source frame using M_s2h from
scene.json, following scripts/gsplat_render_path.py:
    c2w_s     = inv(M_s2h) @ c2w_Habitat
    c2w_s_cv  = c2w_s @ diag(1,-1,-1,1)        # OpenGL(Habitat) -> OpenCV
    extrinsic = inv(c2w_s_cv)                  # world-to-camera for Open3D

Usage:
    python render_mesh_habitat_path.py MESH.ply path.json scene.json OUT_PREFIX [--stride 4]
"""
from __future__ import annotations
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("path_json"); ap.add_argument("scene_json")
    ap.add_argument("out_prefix")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    import open3d as o3d

    path = json.load(open(a.path_json))
    poses = np.asarray(path["poses"])                      # (N,4,4) Habitat c2w
    hfov, W, H = path["hfov"], int(path["width"]), int(path["height"])
    W, H = int(W * a.scale), int(H * a.scale)
    fx = (W / 2) / np.tan(np.radians(hfov) / 2); fy = fx
    cx, cy = W / 2, H / 2
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix

    M = np.asarray(json.load(open(a.scene_json))["M_s2h"])  # source -> Habitat
    Minv = np.linalg.inv(M)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])

    m = o3d.io.read_triangle_mesh(a.mesh); m.compute_vertex_normals()
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
    r.scene.add_geometry("m", m, mat)
    r.scene.scene.set_sun_light([0, -1, 0], [1, 1, 1], 60000)
    r.scene.scene.enable_sun_light(True)

    n = 0
    for i in range(0, len(poses), a.stride):
        c2w_s = Minv @ poses[i]
        c2w_s_cv = c2w_s @ flip
        extrinsic = np.linalg.inv(c2w_s_cv)
        r.setup_camera(K, extrinsic, W, H)
        o3d.io.write_image(f"{a.out_prefix}_{n:04d}.png", r.render_to_image(), 9)
        n += 1
    print(f"wrote {n} frames ({W}x{H})", flush=True)
    print("HABPATH_DONE", flush=True)


if __name__ == "__main__":
    main()
