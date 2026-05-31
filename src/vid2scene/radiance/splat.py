"""Stage 3 — Gaussian Splatting (gsplat) with adaptive densification.

Seeds Gaussians from the MapAnything point cloud, then runs real 3DGS training
(gsplat DefaultStrategy: clone/split high-gradient gaussians, prune low-opacity)
with an L1+SSIM photometric loss against the posed keyframes. Exports a standard
3DGS .ply (SuperSplat-compatible) and renders novel views with gsplat.

Runs in the `splat` env (py3.10 / torch2.2+cu121 / prebuilt gsplat):
    python -m vid2scene.radiance.splat --frames-dir ... --report ... --poses ...
        --init-ply ... --out-ply ... --render-dir ... --steps 7000 --num-points 100000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

SH_C0 = 0.28209479177387814


def _ssim_window(ch, ws, sigma, device):
    c = torch.arange(ws, device=device) - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    w = (g[:, None] * g[None, :])
    return w.expand(ch, 1, ws, ws).contiguous()


def ssim(x, y, window):
    pad, ch = window.shape[-1] // 2, x.shape[1]
    mux = F.conv2d(x, window, padding=pad, groups=ch)
    muy = F.conv2d(y, window, padding=pad, groups=ch)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = F.conv2d(x * x, window, padding=pad, groups=ch) - mux2
    sy = F.conv2d(y * y, window, padding=pad, groups=ch) - muy2
    sxy = F.conv2d(x * y, window, padding=pad, groups=ch) - muxy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * muxy + c1) * (2 * sxy + c2)) / ((mux2 + muy2 + c1) * (sx + sy + c2))
    return s.mean()


def arkit_to_opencv_c2w(p16):
    m = np.array(p16, np.float64).reshape(4, 4).T
    m[:3, :3] = m[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return m.astype(np.float32)


def load_ply_points(path, max_points):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header\n" not in hdr:
            hdr += f.read(128)
        head_len = hdr.index(b"end_header\n") + len(b"end_header\n")
        n = int(next(l for l in hdr[:head_len].decode("ascii", "ignore").splitlines()
                     if l.startswith("element vertex")).split()[-1])
        f.seek(head_len)
        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec = np.fromfile(f, dtype=dt, count=n)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], 1).astype(np.float32)
    cols = np.stack([rec["r"], rec["g"], rec["b"]], 1).astype(np.float32) / 255.0
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts, cols


def load_views(frames_dir, report, poses, device):
    rep = json.load(open(report))
    sel = sorted(f["index"] for f in rep["frames"] if f["selected"])
    pdata = json.load(open(poses))
    fx, fy, cx, cy = pdata["intrinsics"]
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)
    frames_dir = Path(frames_dir)
    views = []
    for k, src in enumerate(sel):
        p = frames_dir / f"frame_{k:04d}.jpg"
        if not p.exists():
            continue
        img = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0).to(device)
        c2w = torch.from_numpy(arkit_to_opencv_c2w(pdata["frames"][src]["cameraPose"])).to(device)
        views.append((img, K, torch.linalg.inv(c2w)))
    return views


def load_views_from_cameras(cameras_json, device):
    """Views from a recon_folder cameras.json (image + estimated K + OpenCV cam2world)."""
    import torch
    from PIL import Image
    data = json.load(open(cameras_json))
    views = []
    for v in data["views"]:
        img = torch.from_numpy(np.asarray(Image.open(v["image"]).convert("RGB"), np.float32) / 255.0)  # CPU (may be hi-res)
        K = torch.tensor(v["K"], dtype=torch.float32, device=device)
        c2w = torch.tensor(v["c2w"], dtype=torch.float32, device=device)
        views.append((img, K, torch.linalg.inv(c2w)))
    return views


def rasterize(splats, w2c, K, H, W):
    return rasterization(
        splats["means"], F.normalize(splats["quats"], dim=-1), torch.exp(splats["scales"]),
        torch.sigmoid(splats["opacities"]), splats["colors"], w2c[None], K[None], W, H,
        sh_degree=None, packed=False,
    )


def export_3dgs_ply(path, splats):
    means = splats["means"].detach().cpu().numpy()
    quats = F.normalize(splats["quats"], dim=-1).detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    opac = splats["opacities"].detach().cpu().numpy().reshape(-1, 1)
    fdc = (splats["colors"].detach().cpu().numpy().clip(0, 1) - 0.5) / SH_C0
    n = len(means)
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rec = np.zeros(n, dtype=[(p, "<f4") for p in props])
    rec["x"], rec["y"], rec["z"] = means.T
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = fdc.T
    rec["opacity"] = opac[:, 0]
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = scales.T
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = quats.T
    header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); rec.tofile(f)
    print(f"wrote {path} ({n} gaussians)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir")
    ap.add_argument("--report")
    ap.add_argument("--poses")
    ap.add_argument("--cameras-json", help="recon_folder cameras.json (alternative to frames/report/poses)")
    ap.add_argument("--init-ply", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--render-dir", required=True)
    ap.add_argument("--steps", type=int, default=7000)
    ap.add_argument("--num-points", type=int, default=100_000)
    args = ap.parse_args()

    device = "cuda"
    if args.cameras_json:
        views = load_views_from_cameras(args.cameras_json, device)
    else:
        views = load_views(args.frames_dir, args.report, args.poses, device)
    print(f"{len(views)} training views", flush=True)
    pts, cols = load_ply_points(args.init_ply, args.num_points)
    means = torch.tensor(pts, device=device)
    n = len(means)
    center = means.mean(0)
    scene_scale = float((means - center).norm(dim=1).mean().item())
    extent = float((means.max(0).values - means.min(0).values).max().item())
    init_scale = max(extent / (n ** (1.0 / 3.0)), 5e-4)
    print(f"init {n} gaussians  scene_scale {scene_scale:.3f}  init_scale {init_scale:.4f}", flush=True)

    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(init_scale), device=device)),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]], device=device).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.full((n,), math.log(0.1 / 0.9), device=device)),
        "colors": torch.nn.Parameter(torch.tensor(cols, device=device)),
    }).to(device)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": [splats[k]], "lr": lrs[k]}], eps=1e-15)
                  for k in splats}

    strategy = DefaultStrategy(
        refine_start_iter=500, refine_stop_iter=int(args.steps * 0.85),
        reset_every=3000, refine_every=100, pause_refine_after_reset=len(views),
        verbose=True,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    window = _ssim_window(3, 11, 1.5, device)
    rng = np.random.default_rng(0)
    for step in range(args.steps):
        img_gt, K, w2c = views[int(rng.integers(len(views)))]
        img_gt = img_gt.to(device)            # move per-step (images may live on CPU)
        H, W = img_gt.shape[:2]
        out, _, info = rasterize(splats, w2c, K, H, W)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        pred = out[0].clamp(0, 1)
        x, y = pred.permute(2, 0, 1)[None], img_gt.permute(2, 0, 1)[None]
        loss = 0.8 * F.l1_loss(pred, img_gt) + 0.2 * (1.0 - ssim(x, y, window))
        for o in optimizers.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in optimizers.values():
            o.step()
        strategy.step_post_backward(splats, optimizers, state, step, info)
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  loss {loss.item():.4f}  gaussians {len(splats['means'])}", flush=True)

    export_3dgs_ply(args.out_ply, splats)

    rd = Path(args.render_dir); rd.mkdir(parents=True, exist_ok=True)
    pick = np.linspace(0, len(views) - 1, 8).astype(int)
    tiles = []
    with torch.no_grad():
        for j, vi in enumerate(pick):
            img_gt, K, w2c = views[vi]
            H, W = img_gt.shape[:2]
            out, _, _ = rasterize(splats, w2c, K, H, W)
            arr = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(rd / f"render_{j:02d}.png")
            tiles.append(arr)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]
    Image.fromarray(np.vstack(rows)).save(rd / "montage.png")
    print(f"wrote renders to {rd}", flush=True)


if __name__ == "__main__":
    main()
