"""Build a mesh in the SAME frame as a trained gsplat splat, by rendering the
splat's own depth at the (COLMAP) training poses and TSDF-fusing the posed RGB-D.
Co-registration is automatic (mesh and splat share the world frame). Also emits a
gravity rotation (R_up) + scene.json so a Habitat agent path can later be mapped
back to the splat frame for photoreal rendering.

Run in the `splat` env (gsplat + open3d + gsplat-src/examples on path):
    python scripts/gsplat_tsdf_mesh.py --room-dir <mip360/room> --ckpt ckpt.pt \
        --gsplat-examples ~/gsplat-src/examples --out-mesh room_tsdf.glb --scene-json scene.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys

import numpy as np


def rotation_aligning(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, perp); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gsplat-examples", required=True)
    ap.add_argument("--out-mesh", required=True, help="aligned (Y-up) mesh for Habitat")
    ap.add_argument("--scene-json", required=True)
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--max-cams", type=int, default=80)
    ap.add_argument("--alpha-thresh", type=float, default=0.5, help="drop depth where coverage below this")
    ap.add_argument("--sdf-trunc-mult", type=float, default=4.0)
    ap.add_argument("--voxel-div", type=float, default=512.0)
    ap.add_argument("--fill-holes", type=float, default=0.0, help="fill holes up to this size (world units); 0=off")
    ap.add_argument("--poisson", action="store_true", help="Poisson-reconstruct from the fused points to fill gaps (continuous floor for navmesh)")
    ap.add_argument("--poisson-depth", dest="poisson_depth_p", type=int, default=9)
    ap.add_argument("--sanity", default="", help="optional: save splat render of cam 0 here")
    args = ap.parse_args()

    sys.path.insert(0, args.gsplat_examples)
    import torch
    import open3d as o3d
    from gsplat import rasterization
    from datasets.colmap import Parser

    parser = Parser(data_dir=args.room_dir, factor=args.factor, normalize=True, test_every=10**9)
    c2ws = np.asarray(parser.camtoworlds)
    N = len(c2ws)
    print(f"{N} cameras", flush=True)

    ck = torch.load(args.ckpt, map_location="cuda")
    sp = ck["splats"] if "splats" in ck else ck
    device = "cuda"
    means = sp["means"].to(device)
    scales = torch.exp(sp["scales"]).to(device)
    quats = sp["quats"].to(device)
    op = sp["opacities"].to(device)
    opacities = torch.sigmoid(op.squeeze(-1) if op.dim() > 1 else op)
    colors = torch.cat([sp["sh0"], sp["shN"]], dim=1).to(device)

    pts = means.detach().cpu().numpy()
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    extent = float(np.linalg.norm(hi - lo))
    voxel = extent / args.voxel_div
    print(f"scene extent {extent:.3f}  voxel {voxel:.4f}", flush=True)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=voxel * args.sdf_trunc_mult,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    idx = np.unique(np.linspace(0, N - 1, min(args.max_cams, N)).astype(int))
    for n, j in enumerate(idx):
        c2w = c2ws[j].astype(np.float64)
        cid = parser.camera_ids[j]
        K = np.asarray(parser.Ks_dict[cid], dtype=np.float64)
        W, H = parser.imsize_dict[cid]
        W, H = int(W), int(H)
        viewmat = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)[None]
        Kt = torch.from_numpy(K).float().to(device)[None]
        out, alphas, _ = rasterization(
            means, quats, scales, opacities, colors, viewmat, Kt, W, H,
            sh_degree=args.sh_degree, render_mode="RGB+ED",
            near_plane=0.01, far_plane=1e10, packed=True, rasterize_mode="classic")
        rgb = (out[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth = out[0, :, :, 3].cpu().numpy().astype(np.float32)
        a = alphas[0, :, :, 0].cpu().numpy()
        depth[(a < args.alpha_thresh) | (depth > extent)] = 0.0
        if args.sanity and n == 0:
            o3d.io.write_image(args.sanity, o3d.geometry.Image(np.ascontiguousarray(rgb)))
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0, depth_trunc=extent, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(c2w))
    print("fused; extracting mesh", flush=True)

    mesh = vol.extract_triangle_mesh()
    mesh.remove_unreferenced_vertices()
    ci, cnt, _ = mesh.cluster_connected_triangles()
    ci, cnt = np.asarray(ci), np.asarray(cnt)
    if len(cnt):
        mesh.remove_triangles_by_mask(ci != int(cnt.argmax()))
        mesh.remove_unreferenced_vertices()
    if args.poisson:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        pcd.normals = mesh.vertex_normals
        if mesh.has_vertex_colors():
            pcd.colors = mesh.vertex_colors
        pmesh, pdens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.poisson_depth_p)
        pdens = np.asarray(pdens)
        pmesh.remove_vertices_by_mask(pdens < np.quantile(pdens, 0.02))
        ci2, cnt2, _ = pmesh.cluster_connected_triangles()
        ci2, cnt2 = np.asarray(ci2), np.asarray(cnt2)
        if len(cnt2):
            pmesh.remove_triangles_by_mask(ci2 != int(cnt2.argmax()))
            pmesh.remove_unreferenced_vertices()
        mesh = pmesh
        print(f"poisson-filled: verts={len(mesh.vertices)} tris={len(mesh.triangles)}", flush=True)

    if args.fill_holes > 0:
        try:
            tm = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            tm = tm.fill_holes(hole_size=args.fill_holes)
            mesh = tm.to_legacy()
            mesh.remove_unreferenced_vertices()
            print(f"filled holes up to {args.fill_holes:.3f}", flush=True)
        except Exception as e:
            print(f"fill_holes skipped: {repr(e)[:120]}", flush=True)
    if len(mesh.triangles) > 400000:
        mesh = mesh.simplify_quadric_decimation(400000)
    mesh.compute_vertex_normals()
    print(f"mesh verts={len(mesh.vertices)} tris={len(mesh.triangles)}", flush=True)

    up = np.mean([-c2ws[j][:3, 1] for j in range(N)], axis=0)
    up /= np.linalg.norm(up)
    R3 = rotation_aligning(up, np.array([0.0, 1.0, 0.0]))
    R4 = np.eye(4); R4[:3, :3] = R3
    print(f"gravity up {np.round(up,3)}", flush=True)

    # aligned = T @ R applied to the source-frame mesh; M_s2h maps source -> Habitat world
    aligned = copy.deepcopy(mesh)
    aligned.transform(R4)
    drop = float(np.percentile(np.asarray(aligned.vertices)[:, 1], 1.0))
    T4 = np.eye(4); T4[1, 3] = -drop
    aligned.transform(T4)
    aligned.compute_vertex_normals()
    M_s2h = T4 @ R4
    o3d.io.write_triangle_mesh(args.out_mesh, aligned)
    o3d.io.write_triangle_mesh(args.out_mesh.rsplit(".", 1)[0] + ".obj", aligned)

    json.dump({"M_s2h": M_s2h.tolist(), "extent": extent, "floor_drop": drop},
              open(args.scene_json, "w"))
    print(f"wrote {args.out_mesh} + scene.json (drop={drop:.3f})", flush=True)


if __name__ == "__main__":
    main()
