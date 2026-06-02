"""Fully pose-free feed-forward: VGGT predicts BOTH depth and poses; nothing from
COLMAP/GT is used to build the reconstruction. GT poses are used only to align the
result (similarity / Umeyama) for scoring against the GT mesh.
"""
from __future__ import annotations
import argparse, glob, time
import numpy as np
import torch
from PIL import Image
import imageio.v2 as iio

from kernel import InhabitKernel
from bench import metrics, visibility_cull, sample_mesh

REP = "/cs/student/projects3/2023/dkozlov/datasets/replica/Replica"
DEPTH_SCALE = 6553.5


def umeyama(A, B):
    """s,R,t minimising || s R A + t - B || for A,B [N,3]."""
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    H = A0.T @ B0 / A.shape[0]
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    s = (S * np.array([1, 1, d])).sum() / ((A0 ** 2).sum() / A.shape[0])
    t = muB - s * R @ muA
    return float(s), R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="room0")
    ap.add_argument("--stride", type=int, default=70)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--chunk", type=int, default=12)
    a = ap.parse_args()
    import trimesh
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    d = f"{REP}/{a.scene}"
    traj = np.loadtxt(f"{d}/traj.txt").reshape(-1, 4, 4)
    rgbs = sorted(glob.glob(f"{d}/results/frame*.jpg"))
    deps = sorted(glob.glob(f"{d}/results/depth*.png"))
    idx = list(range(0, len(rgbs), a.stride))
    Wv, Hv = 518, 294
    Wg, Hg = 600, 340
    Kg = (300.0, 300.0, 299.75, 169.75)

    gtposes = [traj[i].astype(np.float32) for i in idx]
    rgb_list = [np.asarray(Image.open(rgbs[i]).resize((Wv, Hv), Image.BILINEAR))[:, :, :3].copy()
                for i in idx]
    gtd_cull = [torch.from_numpy(np.asarray(Image.fromarray(
        iio.imread(deps[i]).astype(np.float32) / DEPTH_SCALE).resize((Wg, Hg), Image.NEAREST),
        np.float32)) for i in idx]

    gtm = trimesh.load(f"{REP}/{a.scene}_mesh.ply", process=False)
    lo = gtm.bounds[0] - 0.3; hi = gtm.bounds[1] + 0.3
    gt_pts = np.asarray(trimesh.sample.sample_surface(gtm, 500_000, seed=0)[0], np.float32)
    gt_pts = visibility_cull(gt_pts, gtd_cull, gtposes, Kg, Wg, Hg, tol=0.05)
    print(f"{a.scene}: {len(idx)} frames; pose-free VGGT; GT pts {len(gt_pts):,}")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    ker = InhabitKernel(lo, hi, voxel=a.voxel, trunc_vox=3.0, device="cuda", robust=True)
    t0 = time.perf_counter()
    for s in range(0, len(idx), a.chunk):
        sl = slice(s, s + a.chunk)
        ims = torch.stack([torch.from_numpy(im).float() / 255 for im in rgb_list[sl]]) \
            .permute(0, 3, 1, 2).to("cuda")
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            p = model(ims)
        depth = p["depth"][0, ..., 0].float().cpu().numpy()       # [n,Hv,Wv]
        conf = p["depth_conf"][0].float().cpu().numpy()
        extri, intri = pose_encoding_to_extri_intri(p["pose_enc"].float(), (Hv, Wv))
        extri = extri[0].cpu().numpy(); intri = intri[0].cpu().numpy()   # [n,3,4],[n,3,3]
        # VGGT cam centres (c2w), then align this chunk to the GT cam centres
        cv = np.array([-(e[:3, :3].T @ e[:3, 3]) for e in extri])
        cg = np.array([gtposes[s + j][:3, 3] for j in range(len(cv))])
        if len(cv) < 3:
            continue
        sc, R, t = umeyama(cv, cg)
        for j in range(depth.shape[0]):
            e = extri[j]; Rv = e[:3, :3].T; cvj = -(e[:3, :3].T @ e[:3, 3])
            Ral = R @ Rv; cal = sc * R @ cvj + t
            c2w = np.eye(4, dtype=np.float32); c2w[:3, :3] = Ral; c2w[:3, 3] = cal
            Kj = (float(intri[j][0, 0]), float(intri[j][1, 1]),
                  float(intri[j][0, 2]), float(intri[j][1, 2]))
            dm = torch.from_numpy(depth[j] * sc)
            ker.integrate(dm, torch.from_numpy(c2w), Kj, conf=torch.from_numpy(conf[j].copy()))
    v, f, _ = ker.extract_mesh()
    dt = time.perf_counter() - t0
    pp = sample_mesh(v, f, 300_000)
    pp = visibility_cull(pp, gtd_cull, gtposes, Kg, Wg, Hg, tol=0.05)
    m = metrics(pp, gt_pts)
    print(f"\nours kernel + POSE-FREE VGGT (depth + poses both predicted):")
    print(f"  Chamfer {m['chamfer_cm']:.2f} cm | acc {m['acc_cm']:.2f} | comp {m['comp_cm']:.2f} "
          f"| F@5cm {m['F@5cm']:.3f} | total {dt:.1f}s")
    print("  (ref: GT-pose VGGT 3.80 cm; GT-depth kernel 2.15 cm)")


if __name__ == "__main__":
    main()
