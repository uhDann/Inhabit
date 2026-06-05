"""Splat-appearance distillation (the novel contribution).

A Gaussian splat is the best appearance model we can fit, but it is not a true mesh.
Here we use a trained 3DGS purely as an APPEARANCE TEACHER: render it from many poses,
including NOVEL poses between the captured ones, and supervise our mesh's appearance to
match. The mesh thus inherits the splat's photorealism (view-dependent highlights, soft
look) while remaining true, editable, physics-ready geometry.

Loss = direct photo loss on real frames  +  distillation loss on novel poses
       (mesh-render vs splat-render).

Teacher: a trained gsplat model (a .ply of Gaussians or a gsplat checkpoint). Train it
with the existing gsplat pipeline (or gsplat.simple_trainer) first; this module only
consumes its renderings. Requires: gsplat, torch, nvdiffrast, lpips.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import trimesh
import imageio.v2 as iio

from .core import Renderer, DeferredAppearance, load_mesh, uv_unwrap, mvp_from_pose, _CV2GL
from . import data as D


class SplatTeacher:
    """Wraps a trained gsplat model and renders it from an OpenCV cam->world pose."""
    def __init__(self, ply_or_ckpt, device="cuda"):
        import gsplat
        self.gsplat = gsplat
        self.device = device
        g = torch.load(ply_or_ckpt, map_location=device) if ply_or_ckpt.endswith(".pt") \
            else _load_gaussian_ply(ply_or_ckpt, device)
        self.means = g["means"]; self.quats = g["quats"]; self.scales = g["scales"]
        self.opac = g["opacities"]; self.colors = g["colors"]   # colors precomputed RGB or SH0

    @torch.no_grad()
    def render(self, pose_c2w, K, W, H):
        fx, fy, cx, cy = K
        viewmat = torch.tensor(np.linalg.inv(pose_c2w), dtype=torch.float32, device=self.device)
        Kmat = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=self.device)
        out, _, _ = self.gsplat.rasterization(
            self.means, self.quats, self.scales, torch.sigmoid(self.opac),
            self.colors, viewmat[None], Kmat[None], W, H)
        return out[0].clamp(0, 1)                    # [H,W,3]


def _load_gaussian_ply(path, device):
    from plyfile import PlyData
    p = PlyData.read(path)["vertex"]
    f = lambda *ns: torch.tensor(np.stack([p[n] for n in ns], 1), dtype=torch.float32, device=device)
    means = f("x", "y", "z")
    scales = f("scale_0", "scale_1", "scale_2")
    quats = f("rot_0", "rot_1", "rot_2", "rot_3")
    opac = f("opacity")[:, 0:1]
    # SH degree-0 color -> RGB
    sh0 = f("f_dc_0", "f_dc_1", "f_dc_2")
    colors = (0.2820948 * sh0 + 0.5)
    return dict(means=means, quats=torch.exp(scales) * 0 + quats, scales=torch.exp(scales),
                opacities=opac, colors=colors)


def sample_novel_poses(train_poses, n, jitter=0.15):
    """Interpolate between random pairs of train poses + small translation jitter."""
    out = []
    for _ in range(n):
        i, j = np.random.randint(len(train_poses), size=2)
        t = np.random.rand()
        A, B = train_poses[i], train_poses[j]
        P = A.copy()
        P[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3] + np.random.randn(3) * jitter
        # slerp-ish: just take A's rotation (small novel baseline); good enough for distill
        out.append(P.astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True); ap.add_argument("--teacher", required=True,
                    help="trained gsplat .ply or .pt")
    ap.add_argument("--replica"); ap.add_argument("--img_dir"); ap.add_argument("--poses")
    ap.add_argument("--K", nargs=4, type=float); ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=1, help="must match eval --stride for an identical held-out split")
    ap.add_argument("--atlas", type=int, default=1024); ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--out", default="runs/photoreal_distill")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"

    ds = (D.load_replica(a.replica, a.scale, stride=a.stride) if a.replica
          else D.load_folder(a.img_dir, a.poses, a.K, a.scale))
    W, H, K = ds["W"], ds["H"], ds["K"]
    v, f, _ = load_mesh(a.mesh)
    vx, fx, uv, _ = uv_unwrap(v, f)
    nx = np.asarray(trimesh.Trimesh(vx, fx, process=False).vertex_normals, np.float32)
    verts = torch.tensor(vx, device=dev); faces = torch.tensor(fx, device=dev)
    uvs = torch.tensor(uv, device=dev); nrm = torch.tensor(nx, device=dev)

    R = Renderer(dev); app = DeferredAppearance(atlas=a.atlas).to(dev)
    teacher = SplatTeacher(a.teacher, dev)
    import lpips
    perc = lpips.LPIPS(net="vgg").to(dev)
    opt = torch.optim.Adam(app.parameters(), lr=1e-2)
    train = ds["train"]; train_poses = [p for _, p, _ in train]

    def photo_loss(pred, gt):
        return (pred - gt).abs().mean() + 0.5 * perc(
            pred.permute(2, 0, 1)[None] * 2 - 1, gt.permute(2, 0, 1)[None] * 2 - 1).mean()

    for it in range(a.iters):
        if it % 2 == 0:                               # real-frame supervision
            rgb, pose, K = train[np.random.randint(len(train))]
            gt = torch.tensor(rgb, device=dev)
        else:                                         # novel-pose distillation
            pose = sample_novel_poses(train_poses, 1)[0]
            gt = teacher.render(pose, K, W, H)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        pred = R.render(verts, faces, uvs, nrm, app, mvp, cp, W, H)
        loss = photo_loss(pred, gt)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 200 == 0:
            print(f"it {it}  loss {loss.item():.4f}", flush=True)
    torch.save({"state": app.state_dict(), "atlas": a.atlas}, f"{a.out}/appearance.pt")
    print("saved", f"{a.out}/appearance.pt")


if __name__ == "__main__":
    main()
