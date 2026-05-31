"""Render two meshes (MonoSDF watertight + DN-Splatter textured) from a SHARED
COLMAP camera pose, so they pixel-align and we can overlay them.

MonoSDF mesh is in COLMAP world. DN-Splatter mesh is in nerfstudio's frame —
we apply the inverse of the dataparser transform (a pure axis swap for the
Mip-NeRF 360 room: x_nerf = (x, z, -y) of COLMAP) to bring it back to COLMAP.

Outputs (per camera): <prefix>_mono.png, <prefix>_dn.png. Local-side ffmpeg
composites the alpha-blended overlay so we keep shader logic simple.

Usage:
    python render_aligned.py MONO_PLY DN_PLY DN_TRANSFORMS COLMAP_ROOT GSPLAT_EXAMPLES OUT_PREFIX [CAM_INDICES...]
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mono_ply")
    ap.add_argument("dn_ply")
    ap.add_argument("dn_transforms")
    ap.add_argument("colmap_root")
    ap.add_argument("gsplat_examples")
    ap.add_argument("out_prefix")
    ap.add_argument("--cams", type=int, nargs="+", default=[40, 120, 200])
    ap.add_argument("--factor", type=int, default=4)
    args = ap.parse_args()

    import open3d as o3d

    sys.path.insert(0, args.gsplat_examples)
    from datasets.colmap import Parser

    # ---- meshes ----
    mono = o3d.io.read_triangle_mesh(args.mono_ply)
    mono.compute_vertex_normals()
    if not mono.has_vertex_colors():
        mono.paint_uniform_color([0.85, 0.55, 0.55])  # warm gray-red so it reads through overlay
    dn = o3d.io.read_triangle_mesh(args.dn_ply)
    dn.compute_vertex_normals()

    # ---- DN-Splatter -> COLMAP world (apply inverse of nerfstudio transform) ----
    with open(args.dn_transforms) as f:
        dt = json.load(f)
    T34 = np.asarray(dt["transform"], dtype=np.float64)  # 3x4 forward (colmap->nerf)
    scale = float(dt.get("scale", 1.0))
    T4 = np.eye(4, dtype=np.float64); T4[:3] = T34
    # forward: x_nerf = scale * T4 @ x_colmap   =>   x_colmap = inv(T4) @ (x_nerf / scale)
    inv = np.linalg.inv(T4)
    if scale != 1.0:
        inv = inv @ np.diag([1.0 / scale, 1.0 / scale, 1.0 / scale, 1.0])
    dn.transform(inv)
    print(f"applied inverse(nerfstudio transform), scale={scale}", flush=True)

    # ---- COLMAP cameras (raw poses, normalize=False to keep COLMAP world frame) ----
    parser = Parser(data_dir=args.colmap_root, factor=args.factor, normalize=False, test_every=10**9)
    print(f"loaded {len(parser.camtoworlds)} colmap cameras at factor {args.factor}", flush=True)
    c2ws = np.asarray(parser.camtoworlds)

    # ---- render ----
    mat_dn = o3d.visualization.rendering.MaterialRecord(); mat_dn.shader = "defaultLit"
    mat_mono = o3d.visualization.rendering.MaterialRecord(); mat_mono.shader = "defaultLit"

    for cam_idx in args.cams:
        if cam_idx >= len(c2ws):
            print(f"skip cam_idx {cam_idx} (out of range)"); continue
        cid = parser.camera_ids[cam_idx]
        K = np.asarray(parser.Ks_dict[cid], dtype=np.float64)
        W, H = parser.imsize_dict[cid]
        W, H = int(W), int(H)
        # crop to even (paranoia) and ensure within Open3D limits
        c2w = c2ws[cam_idx].astype(np.float64)
        extrinsic = np.linalg.inv(c2w)  # world->cam (OpenCV convention)

        # one renderer per cam (in case resolution differs)
        r = o3d.visualization.rendering.OffscreenRenderer(W, H)
        r.scene.set_background([1, 1, 1, 1])
        # downward sun light so floors read
        r.scene.scene.set_sun_light([0.0, 1.0, 0.0], [1, 1, 1], 75000)
        r.scene.scene.enable_sun_light(True)

        # DN render
        r.scene.add_geometry("dn", dn, mat_dn)
        r.setup_camera(o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]).intrinsic_matrix,
                       extrinsic, W, H)
        img_dn = r.render_to_image()
        out_dn = f"{args.out_prefix}_cam{cam_idx}_dn.png"
        o3d.io.write_image(out_dn, img_dn, 9)
        print(f"wrote {out_dn}", flush=True)

        # MONO render (replace dn with mono)
        r.scene.remove_geometry("dn")
        r.scene.add_geometry("mono", mono, mat_mono)
        r.setup_camera(o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2]).intrinsic_matrix,
                       extrinsic, W, H)
        img_mono = r.render_to_image()
        out_mono = f"{args.out_prefix}_cam{cam_idx}_mono.png"
        o3d.io.write_image(out_mono, img_mono, 9)
        print(f"wrote {out_mono}", flush=True)

        # Both together (for a "geometric union" visual)
        r.scene.add_geometry("dn", dn, mat_dn)
        img_both = r.render_to_image()
        out_both = f"{args.out_prefix}_cam{cam_idx}_both.png"
        o3d.io.write_image(out_both, img_both, 9)
        print(f"wrote {out_both}", flush=True)

        del r


if __name__ == "__main__":
    main()
