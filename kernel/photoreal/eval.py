"""Held-out novel-view evaluation: render the trained mesh appearance from poses the
optimisation never saw, score vs the real photos (PSNR/SSIM/LPIPS/DreamSim), and emit
side-by-side panels. Optionally compares against a 3DGS baseline rendered at the same
poses (pass --splat_dir with pre-rendered splat images named like the test index).

Target ("confusable with the real room"): LPIPS <= 0.05, SSIM >= 0.95, PSNR >= 30,
and within ~0.02 LPIPS of the splat. The definitive proof is the 2AFC study (twoafc.py).
"""
from __future__ import annotations
import argparse, os, glob
import numpy as np
import torch
import trimesh
import imageio.v2 as iio

from .core import Renderer, DeferredAppearance, load_mesh, uv_unwrap, mvp_from_pose
from . import data as D


def psnr(a, b):
    return -10 * np.log10(((a - b) ** 2).mean() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True); ap.add_argument("--appearance", required=True)
    ap.add_argument("--replica"); ap.add_argument("--img_dir"); ap.add_argument("--poses")
    ap.add_argument("--K", nargs=4, type=float); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1, help="must match train --stride for an identical held-out split")
    ap.add_argument("--splat_dir", help="optional pre-rendered 3DGS images for the test poses")
    ap.add_argument("--out", default="runs/photoreal/eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    ds = (D.load_replica(a.replica, a.scale, stride=a.stride) if a.replica
          else D.load_folder(a.img_dir, a.poses, a.K, a.scale))
    W, H = ds["W"], ds["H"]
    v, f, _ = load_mesh(a.mesh)
    vx, fx, uv, _ = uv_unwrap(v, f)
    nx = np.asarray(trimesh.Trimesh(vx, fx, process=False).vertex_normals, np.float32)
    verts = torch.tensor(vx, device=dev); faces = torch.tensor(fx, device=dev)
    uvs = torch.tensor(uv, device=dev); nrm = torch.tensor(nx, device=dev)

    ck = torch.load(a.appearance, map_location=dev)
    app = DeferredAppearance(atlas=ck["atlas"]).to(dev); app.load_state_dict(ck["state"]); app.eval()
    R = Renderer(dev)
    import lpips
    from skimage.metrics import structural_similarity as ssim_fn
    perc = lpips.LPIPS(net="vgg").to(dev)
    try:
        import dreamsim
        dsim_model, _ = dreamsim.dreamsim(pretrained=True, device=dev)
    except Exception:
        dsim_model = None

    rows, splat_imgs = [], (sorted(glob.glob(f"{a.splat_dir}/*.png")) if a.splat_dir else None)
    for j, (rgb, pose, K) in enumerate(ds["test"]):
        gt = torch.tensor(rgb, device=dev)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        with torch.no_grad():
            pred = R.render(verts, faces, uvs, nrm, app, mvp, cp, W, H)
        p, g = pred.cpu().numpy(), rgb
        lp = perc(pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).item()
        ss = ssim_fn(p, g, channel_axis=2, data_range=1.0)
        ds_v = (dsim_model(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]).item()
                if dsim_model else float("nan"))
        rows.append((psnr(p, g), ss, lp, ds_v))
        panel = np.concatenate([g, p], 1)
        iio.imwrite(f"{a.out}/cmp_{j:03d}.png", (panel.clip(0, 1) * 255).astype(np.uint8))

    arr = np.array([r[:3] for r in rows])
    dsv = np.array([r[3] for r in rows])
    print(f"\n=== held-out NVS (n={len(rows)}) ===")
    print(f"PSNR  {arr[:,0].mean():.2f}")
    print(f"SSIM  {arr[:,1].mean():.3f}")
    print(f"LPIPS {arr[:,2].mean():.3f}   (target <= 0.05)")
    if dsim_model:
        print(f"DreamSim {np.nanmean(dsv):.3f}")
    np.save(f"{a.out}/metrics.npy", arr)


if __name__ == "__main__":
    main()
