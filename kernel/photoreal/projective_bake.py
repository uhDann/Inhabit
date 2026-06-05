"""Photoreal texture via deterministic PROJECTIVE per-vertex bake (+ optional view-dependent
residual). Beats optimisation (which collapses to muddy mean colour). Reuses the validated
dr.interpolate path -- no UV/atlas convention to get wrong.

IMPORTANT: relies on the y-flipped projection in core.opengl_projection (nvdiffrast is
bottom-left origin; without the flip every render is upside-down and PSNR caps ~15 dB).

Run: python -m photoreal.projective_bake --replica <scene_dir> --mesh mesh.ply --out runs/tex
"""
from __future__ import annotations
import argparse, os
import numpy as np, torch, torch.nn as nn, trimesh, imageio.v2 as iio
import nvdiffrast.torch as dr
from skimage.metrics import structural_similarity as ssim_fn
from .core import load_mesh, mvp_from_pose, _posenc
from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica", required=True); ap.add_argument("--mesh", required=True)
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--view_dep", action="store_true", help="add a regularised view-dependent residual")
    ap.add_argument("--vd_iters", type=int, default=900); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--out", default="runs/tex"); ap.add_argument("--save_mesh")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    ds = D.load_replica(a.replica, a.scale, stride=a.stride); W, H = ds["W"], ds["H"]
    V, F, _ = load_mesh(a.mesh); N = len(V)
    verts = torch.tensor(V, device=dev); faces = torch.tensor(F.astype(np.int32), device=dev)
    vn = torch.tensor(np.asarray(trimesh.Trimesh(V, F, process=False).vertex_normals, np.float32), device=dev)
    vh = torch.cat([verts, torch.ones(N, 1, device=dev)], 1); glctx = dr.RasterizeCudaContext()
    import lpips; perc = lpips.LPIPS(net="vgg").to(dev)

    # projective bake (cos^4/z^2 view weighting, depth-occluded)
    acc = torch.zeros(N, 3, device=dev); wsum = torch.zeros(N, 1, device=dev)
    for rgb, pose, K in ds["train"]:
        fx, fy, cx, cy = K; gt = torch.as_tensor(rgb, device=dev)
        w2c = torch.as_tensor(np.linalg.inv(pose.astype(np.float32)), device=dev); cam = (vh @ w2c.T)[:, :3]; z = cam[:, 2]
        x = fx * cam[:, 0] / z + cx; y = fy * cam[:, 1] / z + cy
        cpw = torch.as_tensor(pose[:3, 3].astype(np.float32), device=dev); vd = cpw - verts
        vd = vd / (vd.norm(dim=-1, keepdim=True) + 1e-9); cosv = (vd * vn).sum(-1).abs().clamp(0, 1)
        mvp, _ = mvp_from_pose(pose, K, W, H); clip = vh @ torch.as_tensor(mvp.T, device=dev)
        rast, _ = dr.rasterize(glctx, clip[None], faces, (H, W))
        zbuf, _ = dr.interpolate(z[None, :, None].contiguous(), rast, faces); zbuf = zbuf[0, ..., 0]
        pixvis = (rast[0, ..., 3] > 0); xi = x.round().long(); yi = y.round().long()
        inb = (z > 0) & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H); xi2 = xi.clamp(0, W - 1); yi2 = yi.clamp(0, H - 1)
        vis = inb & pixvis[yi2, xi2] & (z <= zbuf[yi2, xi2] + a.tol); w = (cosv ** 4) / (z.clamp(min=0.1) ** 2) * vis.float()
        acc += gt[yi2, xi2] * w[:, None]; wsum += w[:, None]
    col = torch.where(wsum > 0, acc / wsum.clamp(min=1e-6), torch.full_like(acc, 0.5)); seen = (wsum[:, 0] > 0)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0); e = np.unique(np.sort(e, 1), axis=0); e = np.concatenate([e, e[:, ::-1]], 0)
    e0 = torch.tensor(e[:, 0], device=dev).long(); e1 = torch.tensor(e[:, 1], device=dev).long(); filled = seen.clone()
    for _ in range(80):
        ns = torch.zeros(N, 3, device=dev); ns.index_add_(0, e0, col[e1] * filled[e1, None].float())
        nc = torch.zeros(N, device=dev); nc.index_add_(0, e0, filled[e1].float()); upd = (~filled) & (nc > 0)
        col[upd] = ns[upd] / nc[upd, None]; filled[upd] = True
        if filled.all(): break

    feat = mlp = None
    if a.view_dep:
        feat = nn.Parameter(torch.zeros(N, 8, device=dev))
        mlp = nn.Sequential(nn.Linear(8 + 3 * 2 * 4, 64), nn.ReLU(True), nn.Linear(64, 64), nn.ReLU(True), nn.Linear(64, 3)).to(dev)
        nn.init.zeros_(mlp[-1].weight); nn.init.zeros_(mlp[-1].bias)
        opt = torch.optim.Adam([{"params": [feat], "lr": 3e-3}, {"params": mlp.parameters(), "lr": 1e-3}])
        for it in range(a.vd_iters):
            rgb, pose, K = ds["train"][np.random.randint(len(ds["train"]))]; gt = torch.as_tensor(rgb, device=dev)
            pred = render(dev, vh, faces, verts, col, feat, mlp, pose, K, W, H, glctx)
            l1 = (pred - gt).abs().mean(); lap = ((feat[e0] - feat[e1]) ** 2).mean()
            (l1 + 0.5 * lap).backward(); opt.step(); opt.zero_grad()

    rows = []
    for j, (rgb, pose, K) in enumerate(ds["test"]):
        with torch.no_grad(): pred = render(dev, vh, faces, verts, col, feat, mlp, pose, K, W, H, glctx).cpu().numpy()
        lp = perc(torch.as_tensor(pred, device=dev).permute(2, 0, 1)[None] * 2 - 1, torch.as_tensor(rgb, device=dev).permute(2, 0, 1)[None] * 2 - 1).item()
        rows.append((-10 * np.log10(((pred - rgb) ** 2).mean() + 1e-10), ssim_fn(pred, rgb, channel_axis=2, data_range=1.0), lp))
        if j % 8 == 0: iio.imwrite(f"{a.out}/cmp_{j:02d}.png", (np.concatenate([rgb, pred], 1) * 255).astype(np.uint8))
    m = np.array(rows); print(f"[texture] PSNR {m[:,0].mean():.2f} SSIM {m[:,1].mean():.3f} LPIPS {m[:,2].mean():.3f}", flush=True)
    if a.save_mesh:
        mm = trimesh.Trimesh(V, F, process=False); mm.visual.vertex_colors = (col.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        mm.export(a.save_mesh)
    print("TEXTURE_DONE", flush=True)


def render(dev, vh, faces, verts, col, feat, mlp, pose, K, W, H, glctx):
    mvp, cp = mvp_from_pose(pose, K, W, H); clip = vh @ torch.as_tensor(mvp.T, device=dev)
    rast, _ = dr.rasterize(glctx, clip[None], faces, (H, W)); base, _ = dr.interpolate(col[None], rast, faces); out = base[0]
    if mlp is not None:
        fi, _ = dr.interpolate(feat[None], rast, faces); pos_i, _ = dr.interpolate(verts[None], rast, faces)
        view = torch.as_tensor(cp, device=dev) - pos_i[0]; view = view / (view.norm(dim=-1, keepdim=True) + 1e-9)
        out = out + 0.3 * torch.tanh(mlp(torch.cat([fi[0], _posenc(view)], -1)))
    vis = (rast[0, ..., 3:4] > 0).float()
    return (out * vis + 1.0 * (1 - vis)).clamp(0, 1)


if __name__ == "__main__":
    main()
