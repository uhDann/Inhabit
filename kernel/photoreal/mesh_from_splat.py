"""Splat -> mesh: render the trained 3DGS colour+depth at the training poses and colour
TSDF-fuse into a mesh whose geometry matches the splat appearance (GS2Mesh/2DGS lineage).
The most complete geometry we produce -> best base for texturing AND separation.

Run: python -m photoreal.mesh_from_splat --replica <scene_dir> --teacher runs/gs/gs_teacher.pt --out viz/mesh.ply
"""
from __future__ import annotations
import argparse, collections
import numpy as np, torch, open3d as o3d
from gsplat import rasterization
from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--voxel", type=float, default=0.012); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = "cuda"
    g = torch.load(a.teacher, map_location=dev)
    means, log_scales, quats = g["means"].to(dev), g["log_scales"].to(dev), g["quats"].to(dev)
    op_raw, col_raw = g["op_raw"].to(dev), g["col_raw"].to(dev)
    ds = D.load_replica(a.replica, a.scale, stride=a.stride); W, H = ds["W"], ds["H"]

    def Kmat(K):
        fx, fy, cx, cy = K; m = torch.eye(3, device=dev); m[0, 0] = fx; m[1, 1] = fy; m[0, 2] = cx; m[1, 2] = cy; return m

    def rgbd(pose, K):
        vm = torch.as_tensor(np.linalg.inv(pose.astype(np.float32)), device=dev)[None]
        out, _, _ = rasterization(means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(log_scales),
            torch.sigmoid(op_raw), torch.sigmoid(col_raw), vm, Kmat(K)[None], W, H,
            render_mode="RGB+ED", backgrounds=torch.zeros(1, 3, device=dev))
        o = out[0]; return o[..., :3].clamp(0, 1).cpu().numpy(), o[..., 3].cpu().numpy()

    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=a.voxel, sdf_trunc=4 * a.voxel,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    fx, fy, cx, cy = ds["train"][0][2]
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    for rgb, pose, K in ds["train"]:
        col, dep = rgbd(pose, K)
        c = o3d.geometry.Image(np.ascontiguousarray((col * 255).astype(np.uint8)))
        d = o3d.geometry.Image(np.ascontiguousarray(dep.astype(np.float32)))
        rgbd_im = o3d.geometry.RGBDImage.create_from_color_and_depth(c, d, depth_scale=1.0, depth_trunc=8.0, convert_rgb_to_intensity=False)
        vol.integrate(rgbd_im, intr, np.linalg.inv(pose.astype(np.float64)))
    mesh = vol.extract_triangle_mesh(); mesh.compute_vertex_normals()
    tri_cl, _, _ = mesh.cluster_connected_triangles(); tri_cl = np.asarray(tri_cl)
    big = collections.Counter(tri_cl).most_common(1)[0][0]
    mesh.remove_triangles_by_mask(tri_cl != big); mesh.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(a.out, mesh, write_ascii=False)
    print(f"mesh-from-splat verts {len(mesh.vertices)} faces {len(mesh.triangles)} -> {a.out}", flush=True)
    print("MESH_FROM_SPLAT_DONE", flush=True)


if __name__ == "__main__":
    main()
