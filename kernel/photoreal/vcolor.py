"""Per-vertex appearance on a FIXED mesh -- the robust path.

Reuses the EXACT dr.interpolate path that already renders geometry/normals correctly
(validated), so there is no UV-atlas / texture-convention to get wrong. Our mesh is dense
(hundreds of k vertices) and each vertex is seen by many posed frames, so per-vertex
colour is well-constrained and converges cleanly under L1.

Stage 1: optimise a per-vertex diffuse colour (L1, smooth).
Stage 2 (optional): a per-vertex feature + tiny MLP adds view-dependent residual (the
"splat look"), decoded from interpolated feature + posenc(view) + posenc(normal).

Trains AND evaluates held-out NVS (PSNR/SSIM/LPIPS) in one run.
Run: python -m photoreal.vcolor --mesh mesh.ply --replica <dir> --out runs/vc
"""
from __future__ import annotations
import argparse, os, glob, time
import numpy as np
import torch
import torch.nn as nn
import trimesh
import imageio.v2 as iio

from .core import load_mesh, mvp_from_pose, _posenc
from . import data as D


def _save(path, img):
    iio.imwrite(path, (np.clip(img, 0, 1) * 255).astype(np.uint8))


def psnr(a, b):
    return -10.0 * np.log10(np.mean((a - b) ** 2) + 1e-10)


class VertexAppearance(nn.Module):
    def __init__(self, V, feat_ch=8, hidden=64, view_dep=True):
        super().__init__()
        self.col = nn.Parameter(torch.full((V, 3), 0.5))
        self.view_dep = view_dep
        if view_dep:
            self.feat = nn.Parameter(torch.zeros(V, feat_ch))
            din = feat_ch + 3 * 2 * 4 + 3 * 2 * 4
            self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(True),
                                     nn.Linear(hidden, hidden), nn.ReLU(True),
                                     nn.Linear(hidden, 3))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--replica"); ap.add_argument("--img_dir"); ap.add_argument("--poses")
    ap.add_argument("--K", nargs=4, type=float); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=600, help="diffuse-only (+ no view-dep) iters")
    ap.add_argument("--lpips_from", type=int, default=1200)
    ap.add_argument("--out", default="runs/vc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeCudaContext()

    ds = (D.load_replica(a.replica, a.scale, stride=a.stride) if a.replica
          else D.load_folder(a.img_dir, a.poses, a.K, a.scale))
    W, H = ds["W"], ds["H"]
    print(f"train {len(ds['train'])}  test {len(ds['test'])}  @ {W}x{H}", flush=True)

    v, f, n = load_mesh(a.mesh)
    verts = torch.tensor(v, device=dev); faces = torch.tensor(f.astype(np.int32), device=dev)
    nrm = torch.tensor(n, device=dev)
    vh = torch.cat([verts, torch.ones(len(verts), 1, device=dev)], 1)
    app = VertexAppearance(len(v)).to(dev)

    def render(mvp, cp, train=True):
        clip = vh @ torch.as_tensor(mvp.T, device=dev)
        rast, _ = dr.rasterize(glctx, clip[None], faces, (H, W))
        base, _ = dr.interpolate(app.col[None], rast, faces)
        col = base[0]
        if app.view_dep and (not train or app._use_vd):
            feat_i, _ = dr.interpolate(app.feat[None], rast, faces)
            nrm_i, _ = dr.interpolate(nrm[None], rast, faces)
            pos_i, _ = dr.interpolate(verts[None], rast, faces)
            view = torch.as_tensor(cp, device=dev) - pos_i[0]
            view = view / (view.norm(dim=-1, keepdim=True) + 1e-9)
            nn_ = nrm_i[0] / (nrm_i[0].norm(dim=-1, keepdim=True) + 1e-9)
            x = torch.cat([feat_i[0], _posenc(view), _posenc(nn_)], -1)
            col = col + app.mlp(x)
        vis = (rast[0, ..., 3:4] > 0).float()
        col = col * vis + 1.0 * (1 - vis)
        col = dr.antialias(col.clamp(0, 1)[None], rast, clip[None], faces)[0]
        return col

    import lpips
    perc = lpips.LPIPS(net="vgg").to(dev)
    params = [{"params": [app.col], "lr": 2e-2}]
    if app.view_dep:
        params.append({"params": [app.feat, *app.mlp.parameters()], "lr": 5e-3})
    opt = torch.optim.Adam(params)
    app._use_vd = False
    train = ds["train"]
    t0 = time.time()
    for it in range(a.iters):
        app._use_vd = (it >= a.warmup)
        rgb, pose, K = train[np.random.randint(len(train))]
        gt = torch.as_tensor(rgb, device=dev)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        pred = render(mvp, cp, train=True)
        loss = (pred - gt).abs().mean()
        if it >= a.lpips_from:
            loss = loss + 0.1 * perc(pred.permute(2, 0, 1)[None] * 2 - 1,
                                     gt.permute(2, 0, 1)[None] * 2 - 1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            print(f"it {it}  L1 {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
            _save(f"{a.out}/train_{it:05d}.png", pred.detach().cpu().numpy())
    torch.save({"col": app.col.detach().cpu(), "state": app.state_dict()}, f"{a.out}/vcolor.pt")

    # --- held-out eval ---
    app._use_vd = app.view_dep
    from skimage.metrics import structural_similarity as ssim_fn
    rows = []
    for j, (rgb, pose, K) in enumerate(ds["test"]):
        mvp, cp = mvp_from_pose(pose, K, W, H)
        with torch.no_grad():
            pred = render(mvp, cp, train=False).cpu().numpy()
        lp = perc(torch.as_tensor(pred, device=dev).permute(2, 0, 1)[None] * 2 - 1,
                  torch.as_tensor(rgb, device=dev).permute(2, 0, 1)[None] * 2 - 1).item()
        ss = ssim_fn(pred, rgb, channel_axis=2, data_range=1.0)
        rows.append((psnr(pred, rgb), ss, lp))
        _save(f"{a.out}/cmp_{j:03d}.png", np.concatenate([rgb, pred], 1))
    arr = np.array(rows)
    print(f"\n=== held-out NVS (n={len(rows)}) ===")
    print(f"PSNR  {arr[:,0].mean():.2f}")
    print(f"SSIM  {arr[:,1].mean():.3f}")
    print(f"LPIPS {arr[:,2].mean():.3f}")
    np.save(f"{a.out}/metrics.npy", arr)
    print("VCOLOR_DONE", flush=True)


if __name__ == "__main__":
    main()
