"""Render a reconstructed mesh along a SMOOTHED MuSHRoom iPhone trajectory.

The raw handheld capture path is jittery, and sampling it sparsely looks fast and
disorienting. Here we downsample the posed cameras to keyframes, fit a smooth cubic
spline through the keyframe positions and a slerp through their rotations, then
render many evenly-spaced frames along that smooth path. The result is a calm
interior fly-through of the same real capture.

Usage:
  python render_mesh_mushroom_smooth.py MESH.ply LONG_CAPTURE_DIR OUT_PREFIX \
      [--key-stride 12] [--frames 150] [--scale 1.0] [--trim A B]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("long_capture")
    ap.add_argument("out_prefix")
    ap.add_argument("--key-stride", type=int, default=12)   # keep every Nth pose as a keyframe
    ap.add_argument("--frames", type=int, default=150)       # rendered (interpolated) frames
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--trim", type=float, nargs=2, default=[0.0, 1.0])  # use this fraction of the path
    a = ap.parse_args()

    import open3d as o3d
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp

    LC = Path(a.long_capture)
    meta = json.load(open(LC / "transformations_colmap.json"))
    W = int(round(int(meta["w"]) * a.scale)); H = int(round(int(meta["h"]) * a.scale))
    fx = float(meta["fl_x"]) * a.scale; fy = float(meta["fl_y"]) * a.scale
    cx = min(max(float(meta["cx"]) * a.scale, 0.5), W - 1.5)
    cy = min(max(float(meta["cy"]) * a.scale, 0.5), H - 1.5)
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix

    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)  # OpenGL c2w

    # trim to a contiguous fraction of the capture, then keep keyframes
    lo, hi = int(a.trim[0] * len(c2w)), int(a.trim[1] * len(c2w))
    c2w = c2w[lo:hi][:: a.key_stride]
    pos = c2w[:, :3, 3]
    rots = Rotation.from_matrix(c2w[:, :3, :3])           # OpenGL camera rotations

    # smooth, arc-length-ish parameterisation
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t /= t[-1]
    csx = [CubicSpline(t, pos[:, k]) for k in range(3)]
    slerp = Slerp(t, rots)

    tt = np.linspace(0, 1, a.frames)
    sm_pos = np.stack([cs(tt) for cs in csx], 1)
    sm_rot = slerp(tt).as_matrix()

    m = o3d.io.read_triangle_mesh(a.mesh); m.compute_vertex_normals()
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultLit"
    r.scene.add_geometry("m", m, mat)
    r.scene.scene.set_sun_light([0.3, -1, 0.3], [1, 1, 1], 75000)
    r.scene.scene.enable_sun_light(True)

    for i in range(a.frames):
        g = np.eye(4); g[:3, :3] = sm_rot[i]; g[:3, 3] = sm_pos[i]   # smoothed c2w (OpenGL)
        g[:3, 1:3] *= -1                                              # OpenGL -> OpenCV
        r.setup_camera(K, np.linalg.inv(g), W, H)
        o3d.io.write_image(f"{a.out_prefix}_{i:04d}.png", r.render_to_image(), 9)
    print(f"wrote {a.frames} smoothed frames ({W}x{H}) to {a.out_prefix}_*.png", flush=True)
    print("MESHTRAJ_DONE", flush=True)


if __name__ == "__main__":
    main()
