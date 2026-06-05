"""3DGS appearance teacher (gsplat), initialised from a mesh's vertices + (optional)
baked vertex colours. Doubles as the mesh-vs-splat baseline and the distillation teacher.
No densification (mesh init is already dense).

Run: python -m photoreal.splat_teacher --replica <scene_dir> --mesh mesh.ply --out runs/gs
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import imageio.v2 as iio
from gsplat import rasterization
from skimage.metrics import structural_similarity as ssim_fn
from . import data as D


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica", required=True)
    ap.add_argument("--mesh", required=True, help="init geometry (vertices) for the gaussians")
    ap.add_argument("--colors", help="optional textured .ply to init gaussian colours")
    ap.add_argument("--scale", type=float, default=0.5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--iters", type=int, default=4000); ap.add_argument("--out", default="runs/gs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    import trimesh
    v = np.asarray(trimesh.load(a.mesh, process=False).vertices, np.float32)
    c0 = np.full((len(v), 3), 0.5, np.float32)
    if a.colors:
        m = trimesh.load(a.colors, process=False)
        cc = np.asarray(m.visual.vertex_colors, np.float32)[:, :3] / 255.0
        if len(cc) == len(v):
            c0 = cc
    ds = D.load_replica(a.replica, a.scale, stride=a.stride); W, H = ds["W"], ds["H"]
    print(f"gaussians {len(v)}  train {len(ds['train'])}  test {len(ds['test'])}", flush=True)

    means = torch.tensor(v, device=dev, requires_grad=True)
    log_scales = torch.full((len(v), 3), np.log(0.02), device=dev, requires_grad=True)
    quats = torch.zeros(len(v), 4, device=dev); quats[:, 0] = 1; quats.requires_grad_(True)
    op_raw = torch.full((len(v),), _logit(0.8), device=dev, requires_grad=True)
    col_raw = torch.tensor(_logit(c0), device=dev, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [means], "lr": 1.6e-4}, {"params": [log_scales], "lr": 5e-3},
        {"params": [quats], "lr": 1e-3}, {"params": [op_raw], "lr": 5e-2},
        {"params": [col_raw], "lr": 5e-3}])

    def Kmat(K):
        fx, fy, cx, cy = K; m = torch.eye(3, device=dev); m[0, 0] = fx; m[1, 1] = fy; m[0, 2] = cx; m[1, 2] = cy; return m

    def render(pose, K, mode="RGB"):
        vm = torch.as_tensor(np.linalg.inv(pose.astype(np.float32)), device=dev)[None]
        out, _, _ = rasterization(means, torch.nn.functional.normalize(quats, dim=-1), torch.exp(log_scales),
            torch.sigmoid(op_raw), torch.sigmoid(col_raw), vm, Kmat(K)[None], W, H,
            render_mode=mode, backgrounds=torch.zeros(1, 3, device=dev) if "ED" in mode else torch.ones(1, 3, device=dev))
        return out[0]

    import lpips; perc = lpips.LPIPS(net="vgg").to(dev)
    tr = ds["train"]
    for it in range(a.iters):
        rgb, pose, K = tr[np.random.randint(len(tr))]; gt = torch.as_tensor(rgb, device=dev)
        pred = render(pose, K)[..., :3].clamp(0, 1)
        loss = (pred - gt).abs().mean()
        loss.backward(); opt.step(); opt.zero_grad()
        if it % 500 == 0:
            print(f"gs it{it} L1 {loss.item():.4f}", flush=True)
    rows = []
    for j, (rgb, pose, K) in enumerate(ds["test"]):
        with torch.no_grad(): pred = render(pose, K)[..., :3].clamp(0, 1).cpu().numpy()
        lp = perc(torch.as_tensor(pred, device=dev).permute(2, 0, 1)[None] * 2 - 1,
                  torch.as_tensor(rgb, device=dev).permute(2, 0, 1)[None] * 2 - 1).item()
        rows.append((-10 * np.log10(((pred - rgb) ** 2).mean() + 1e-10),
                     ssim_fn(pred, rgb, channel_axis=2, data_range=1.0), lp))
    a_ = np.array(rows)
    print(f"[SPLAT] PSNR {a_[:,0].mean():.2f} SSIM {a_[:,1].mean():.3f} LPIPS {a_[:,2].mean():.3f}", flush=True)
    torch.save({k: t.detach().cpu() for k, t in
                {"means": means, "log_scales": log_scales, "quats": quats, "op_raw": op_raw, "col_raw": col_raw}.items()},
               f"{a.out}/gs_teacher.pt")
    print("SPLAT_TEACHER_DONE", flush=True)


if __name__ == "__main__":
    main()
