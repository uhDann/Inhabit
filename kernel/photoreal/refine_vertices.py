"""Differentiable vertex refinement (MILo / nvdiffrec lineage), tuned for our pipeline.

Optimise per-vertex position offsets so the mesh's rendered DEPTH matches the splat (dense
geometry oracle) and its rendered COLOUR matches the real frames (silhouette gradients via
nvdiffrast antialias), with a STRONG Laplacian smoothness prior.

The key non-obvious finding: this method is regularisation-gated. Weak smoothness (LAP~8)
overfits train views with rough deformation and makes FID WORSE (office0: FID 74->98);
strong smoothness (LAP~40) keeps the deformation smooth so the alignment gain comes WITHOUT
the realism penalty (office0: PSNR 26.7->35.6, FID 74->61). A naive reimplementation that
skips this is actively harmful.

Run: python -m photoreal.refine_vertices --scene <dir> --mesh mfs.ply --teacher gs.pt \
        --lap 40 --out viz/refined.ply
"""
from __future__ import annotations
import argparse, time
import numpy as np, torch, torch.nn as nn, trimesh, open3d as o3d
import nvdiffrast.torch as dr
from gsplat import rasterization
from .core import mvp_from_pose
from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--mesh", required=True); ap.add_argument("--teacher", required=True)
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1200); ap.add_argument("--lap", type=float, default=40.0,
                    help="Laplacian smoothness weight (CRITICAL; <~20 hurts FID)")
    ap.add_argument("--decimate", type=int, default=350000); ap.add_argument("--lpips", type=float, default=0.0,
                    help="optional perceptual term weight on the rendered colour")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = "cuda"
    ds = D.load_replica(a.scene, a.scale, stride=a.stride); W, H = ds["W"], ds["H"]
    om = o3d.io.read_triangle_mesh(a.mesh)
    if len(om.triangles) > a.decimate: om = om.simplify_quadric_decimation(a.decimate)
    V0 = np.asarray(om.vertices, np.float32); F = np.asarray(om.triangles, np.int32); N = len(V0)
    print(f"opt mesh {N} verts {len(F)} faces  @ {W}x{H}", flush=True)
    base = torch.tensor(V0, device=dev); dv = nn.Parameter(torch.zeros(N, 3, device=dev))
    col = nn.Parameter(torch.full((N, 3), 0.5, device=dev))
    faces = torch.tensor(F, device=dev); glctx = dr.RasterizeCudaContext()
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0); e = np.unique(np.sort(e, 1), axis=0); e = np.concatenate([e, e[:, ::-1]], 0)
    e0 = torch.tensor(e[:, 0], device=dev).long(); e1 = torch.tensor(e[:, 1], device=dev).long()
    deg = torch.zeros(N, device=dev); deg.index_add_(0, e0, torch.ones(len(e0), device=dev)); deg = deg.clamp(min=1)
    g = torch.load(a.teacher, map_location=dev)
    gm, gls, gq, gop, gc = g["means"].to(dev), g["log_scales"].to(dev), g["quats"].to(dev), g["op_raw"].to(dev), g["col_raw"].to(dev)
    perc = None
    if a.lpips > 0:
        import lpips; perc = lpips.LPIPS(net="vgg").to(dev)

    def Kmat(K):
        fx, fy, cx, cy = K; m = torch.eye(3, device=dev); m[0, 0] = fx; m[1, 1] = fy; m[0, 2] = cx; m[1, 2] = cy; return m

    def splat(pose, K):
        vm = torch.as_tensor(np.linalg.inv(pose.astype(np.float32)), device=dev)[None]
        out, al, _ = rasterization(gm, torch.nn.functional.normalize(gq, dim=-1), torch.exp(gls), torch.sigmoid(gop),
            torch.sigmoid(gc), vm, Kmat(K)[None], W, H, render_mode="RGB+ED", backgrounds=torch.zeros(1, 3, device=dev))
        o = out[0]; return o[..., 3], al[0, ..., 0]

    def render(pose, K):
        vw = base + dv; vh = torch.cat([vw, torch.ones(N, 1, device=dev)], 1)
        mvp, _ = mvp_from_pose(pose, K, W, H); clip = vh @ torch.as_tensor(mvp.T, device=dev)
        w2c = torch.as_tensor(np.linalg.inv(pose.astype(np.float32)), device=dev); camz = (vh @ w2c.T)[:, 2]
        rast, _ = dr.rasterize(glctx, clip[None], faces, (H, W))
        cimg, _ = dr.interpolate(col[None], rast, faces); cimg = dr.antialias(cimg, rast, clip[None], faces)[0]
        dimg, _ = dr.interpolate(camz[None, :, None].contiguous(), rast, faces)
        return cimg, dimg[0, ..., 0], (rast[0, ..., 3] > 0)

    opt = torch.optim.Adam([{"params": [dv], "lr": 1e-3}, {"params": [col], "lr": 1e-2}]); t0 = time.time()
    for it in range(a.iters):
        rgb, pose, K = ds["train"][np.random.randint(len(ds["train"]))]; gt = torch.as_tensor(rgb, device=dev)
        sd, sa = splat(pose, K); sd = sd.detach(); sa = sa.detach()
        cimg, dimg, vis = render(pose, K)
        m = vis & (sa > 0.9) & (sd > 0.1)
        Lc = (cimg[vis] - gt[vis]).abs().mean()
        Ld = (dimg[m] - sd[m]).abs().mean() if m.any() else torch.tensor(0., device=dev)
        nb = torch.zeros(N, 3, device=dev); nb.index_add_(0, e0, dv[e1]); Llap = ((dv - nb / deg[:, None]) ** 2).mean()
        loss = Lc + 0.5 * Ld + a.lap * Llap + 0.05 * (dv ** 2).mean()
        if perc is not None:
            loss = loss + a.lpips * perc(cimg.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 300 == 0:
            print(f"it{it} Lc {Lc.item():.4f} Ld {Ld.item():.4f} |dv| {dv.detach().norm(dim=1).mean().item()*1000:.1f}mm ({time.time()-t0:.0f}s)", flush=True)
    trimesh.Trimesh((base + dv).detach().cpu().numpy(), F, process=False).export(a.out)
    print(f"refined mesh -> {a.out}  mean offset {dv.detach().norm(dim=1).mean().item()*1000:.1f}mm", flush=True)
    print("REFINE_DONE", flush=True)


if __name__ == "__main__":
    main()
