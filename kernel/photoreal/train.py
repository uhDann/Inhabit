"""Train view-dependent appearance on a FIXED mesh against posed frames.
Geometry frozen (our kernel's mesh is the asset). Two stages:
  1. BAKE a diffuse atlas by projecting the GT frames onto the mesh (multi-view
     texturing) -- a stable, room-like starting point (our mesh has no usable colour).
  2. REFINE the diffuse atlas + a small view-dependent MLP under L1 (+ light LPIPS after
     a warmup) with TV regularisation, so it sharpens without diverging into texel noise.

Run (GPU): python -m photoreal.train --mesh mesh.ply --replica <dir> --out runs/photoreal
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
import trimesh
import imageio.v2 as iio

from .core import Renderer, DeferredAppearance, load_mesh, uv_unwrap, mvp_from_pose, bake_diffuse
from . import data as D


def _save(path, img):
    iio.imwrite(path, (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--replica", help="Replica scene dir")
    ap.add_argument("--img_dir"); ap.add_argument("--poses"); ap.add_argument("--K", nargs=4, type=float)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1, help="subsample frames (Replica has 2000)")
    ap.add_argument("--atlas", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--warmup", type=int, default=400, help="L1-only iters before LPIPS")
    ap.add_argument("--out", default="runs/photoreal")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    ds = (D.load_replica(a.replica, a.scale, stride=a.stride) if a.replica
          else D.load_folder(a.img_dir, a.poses, a.K, a.scale))
    W, H = ds["W"], ds["H"]
    print(f"train {len(ds['train'])}  test {len(ds['test'])}  @ {W}x{H}", flush=True)

    v, f, _ = load_mesh(a.mesh)
    vx, fx, uv, _ = uv_unwrap(v, f)
    nx = np.asarray(trimesh.Trimesh(vx, fx, process=False).vertex_normals, np.float32)
    verts = torch.tensor(vx, device=dev); faces = torch.tensor(fx, device=dev)
    uvs = torch.tensor(uv, device=dev); nrm = torch.tensor(nx, device=dev)

    R = Renderer(dev)
    app = DeferredAppearance(atlas=a.atlas).to(dev)

    # --- stage 1: bake diffuse from the training views ---
    print("baking diffuse atlas from", len(ds["train"]), "views...", flush=True)
    t0 = time.time()
    baked = bake_diffuse(R, verts, faces, uvs, nrm, ds["train"], a.atlas, W, H, dev)
    app.diffuse.data.copy_(baked[None])
    print(f"baked in {time.time()-t0:.1f}s", flush=True)
    # bake-only preview on a held-out view (no learned appearance yet)
    rgb0, pose0, K0 = ds["test"][0]
    mvp0, cp0 = mvp_from_pose(pose0, K0, W, H)
    with torch.no_grad():
        _save(f"{a.out}/bake_preview.png", R.render(verts, faces, uvs, nrm, app, mvp0, cp0, W, H))
        _save(f"{a.out}/bake_gt.png", torch.as_tensor(rgb0))

    # --- stage 2: refine (stable: split LR, L1 warmup, light LPIPS, TV reg) ---
    import lpips
    perc = lpips.LPIPS(net="vgg").to(dev)
    opt = torch.optim.Adam([
        {"params": [app.diffuse], "lr": 5e-3},
        {"params": [app.feat, *app.mlp.parameters()], "lr": 1e-3},
    ])
    train = ds["train"]
    for it in range(a.iters):
        rgb, pose, K = train[np.random.randint(len(train))]
        gt = torch.as_tensor(rgb, device=dev)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        pred = R.render(verts, faces, uvs, nrm, app, mvp, cp, W, H)
        l1 = (pred - gt).abs().mean()
        loss = l1
        if it >= a.warmup:
            lp = perc(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()
            loss = loss + 0.15 * lp
        # total-variation reg on the diffuse atlas (kills per-texel noise)
        d = app.diffuse[0]
        tv = (d[1:] - d[:-1]).abs().mean() + (d[:, 1:] - d[:, :-1]).abs().mean()
        loss = loss + 1e-4 * tv
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 250 == 0:
            print(f"it {it}  L1 {l1.item():.4f}" + (f"  LPIPS {lp.item():.4f}" if it >= a.warmup else ""), flush=True)
            _save(f"{a.out}/train_{it:05d}.png", pred)
    torch.save({"state": app.state_dict(), "atlas": a.atlas}, f"{a.out}/appearance.pt")
    _save(f"{a.out}/diffuse_atlas.png", app.diffuse[0])
    print("saved", f"{a.out}/appearance.pt", flush=True)


if __name__ == "__main__":
    main()
