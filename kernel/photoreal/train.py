"""Train view-dependent appearance on a FIXED mesh against posed frames.
Geometry frozen (our kernel's mesh is the asset); we optimise only the appearance
atlas + deferred MLP under an L1 + LPIPS photometric loss.

Run (GPU): python -m photoreal.train --scene <replica_dir or folder> --mesh mesh.ply --out runs/photoreal
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch
import trimesh
import imageio.v2 as iio

from .core import Renderer, DeferredAppearance, load_mesh, uv_unwrap, mvp_from_pose
from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--replica", help="Replica scene dir")
    ap.add_argument("--img_dir"); ap.add_argument("--poses"); ap.add_argument("--K", nargs=4, type=float)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--atlas", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--out", default="runs/photoreal")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    ds = (D.load_replica(a.replica, a.scale) if a.replica
          else D.load_folder(a.img_dir, a.poses, a.K, a.scale))
    W, H = ds["W"], ds["H"]
    print(f"train {len(ds['train'])}  test {len(ds['test'])}  @ {W}x{H}")

    v, f, _ = load_mesh(a.mesh)
    vx, fx, uv, _ = uv_unwrap(v, f)
    nx = np.asarray(trimesh.Trimesh(vx, fx, process=False).vertex_normals, np.float32)
    verts = torch.tensor(vx, device=dev); faces = torch.tensor(fx, device=dev)
    uvs = torch.tensor(uv, device=dev); nrm = torch.tensor(nx, device=dev)

    R = Renderer(dev)
    app = DeferredAppearance(atlas=a.atlas).to(dev)
    import lpips
    perc = lpips.LPIPS(net="vgg").to(dev)
    opt = torch.optim.Adam(app.parameters(), lr=1e-2)

    train = ds["train"]
    for it in range(a.iters):
        rgb, pose, K = train[np.random.randint(len(train))]
        gt = torch.tensor(rgb, device=dev)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        pred = R.render(verts, faces, uvs, nrm, app, mvp, cp, W, H)
        l1 = (pred - gt).abs().mean()
        # LPIPS wants [1,3,H,W] in [-1,1]
        lp = perc(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()
        loss = l1 + 0.5 * lp
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            print(f"it {it}  L1 {l1.item():.4f}  LPIPS {lp.item():.4f}", flush=True)
            iio.imwrite(f"{a.out}/train_{it:05d}.png",
                        (pred.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    torch.save({"state": app.state_dict(), "atlas": a.atlas}, f"{a.out}/appearance.pt")
    # also dump the optimised diffuse atlas as a texture image
    iio.imwrite(f"{a.out}/diffuse_atlas.png",
                (app.diffuse.detach()[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    print("saved", f"{a.out}/appearance.pt")


if __name__ == "__main__":
    main()
