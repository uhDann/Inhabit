"""Colour a predicted mesh by per-vertex distance to the GT mesh and render it
from a Replica-trajectory pose. The standard surface-reconstruction error
visualization (distance-to-GT colormapped, clamped, with a colorbar built later).

Usage:
    python render_chamfer_heatmap.py PRED.ply GT.ply SCENE_DIR FRAME_IDX OUT.png [--clamp 0.05]
"""
from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred"); ap.add_argument("gt"); ap.add_argument("scene_dir")
    ap.add_argument("idx", type=int); ap.add_argument("out")
    ap.add_argument("--clamp", type=float, default=0.05)   # 5 cm
    ap.add_argument("--scale", type=float, default=0.6)
    a = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree
    import matplotlib.cm as cm

    pred = o3d.io.read_triangle_mesh(a.pred); pred.compute_vertex_normals()
    gtm = o3d.io.read_triangle_mesh(a.gt)
    gt_pts = np.asarray(gtm.vertices)
    V = np.asarray(pred.vertices)

    tree = cKDTree(gt_pts)
    dist, _ = tree.query(V, k=1, workers=-1)          # per-vertex distance to GT (metres)
    err = np.clip(dist / a.clamp, 0, 1)               # normalise to [0,1] over [0, clamp]
    colors = cm.turbo(err)[:, :3]
    pred.vertex_colors = o3d.utility.Vector3dVector(colors)
    print(f"err median={np.median(dist)*100:.2f}cm mean={np.mean(dist)*100:.2f}cm "
          f"p90={np.percentile(dist,90)*100:.2f}cm", flush=True)

    W, H = int(1200 * a.scale), int(680 * a.scale)
    fx = fy = 600.0 * a.scale; cx, cy = 599.5 * a.scale, 339.5 * a.scale
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy).intrinsic_matrix
    r = o3d.visualization.rendering.OffscreenRenderer(W, H)
    r.scene.set_background([1, 1, 1, 1])
    mat = o3d.visualization.rendering.MaterialRecord(); mat.shader = "defaultUnlit"
    r.scene.add_geometry("m", pred, mat)
    traj = np.loadtxt(f"{a.scene_dir}/traj.txt").reshape(-1, 4, 4)
    r.setup_camera(K, np.linalg.inv(traj[a.idx]), W, H)
    o3d.io.write_image(a.out, r.render_to_image(), 9)
    print("HEATMAP_DONE", flush=True)


if __name__ == "__main__":
    main()
