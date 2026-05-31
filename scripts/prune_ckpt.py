"""Prune floaters from a gsplat checkpoint: drop low-opacity, oversized, needle,
out-of-bbox, and statistically-isolated gaussians. Saves a cleaned checkpoint
(same format) for rendering, and an optional 3DGS .ply for the web viewer.

Run in the `splat` env:
    python scripts/prune_ckpt.py --ckpt ckpt.pt --out-ckpt ckpt_clean.pt --out-ply clean.ply
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

SH_C0 = 0.28209479177387814


def write_3dgs_ply(path, sp):
    from plyfile import PlyData, PlyElement
    means = sp["means"].cpu().numpy()
    quats = torch.nn.functional.normalize(sp["quats"], dim=-1).cpu().numpy()
    scales = sp["scales"].cpu().numpy()
    op = sp["opacities"].cpu().numpy().reshape(-1, 1)
    fdc = sp["sh0"].reshape(len(means), -1).cpu().numpy()
    frest = sp["shN"].permute(0, 2, 1).reshape(len(means), -1).cpu().numpy()  # channel-major
    n = len(means)
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] \
        + [f"f_rest_{i}" for i in range(frest.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                            "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([means, np.zeros((n, 3), np.float32), fdc, frest, op, scales, quats], 1).astype(np.float32)
    rec = np.empty(n, dtype=[(x, "<f4") for x in names])
    for i, x in enumerate(names):
        rec[x] = data[:, i]
    PlyData([PlyElement.describe(rec, "vertex")]).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--out-ply", default="")
    ap.add_argument("--opacity", type=float, default=0.1)
    ap.add_argument("--scale-q", type=float, default=0.99)
    ap.add_argument("--aniso", type=float, default=25.0)
    ap.add_argument("--bbox-lo", type=float, default=0.5)
    ap.add_argument("--bbox-hi", type=float, default=99.5)
    ap.add_argument("--nb-neighbors", type=int, default=16)
    ap.add_argument("--std-ratio", type=float, default=2.0)
    args = ap.parse_args()

    import open3d as o3d
    ck = torch.load(args.ckpt, map_location="cpu")
    sp = ck["splats"] if "splats" in ck else ck
    op = sp["opacities"]
    opac = torch.sigmoid(op.squeeze(-1) if op.dim() > 1 else op)
    slin = torch.exp(sp["scales"])
    smax = slin.max(1).values
    smin = slin.min(1).values.clamp_min(1e-9)
    means = sp["means"].numpy()

    keep = (opac > args.opacity) & (smax < torch.quantile(smax, args.scale_q)) & ((smax / smin) < args.aniso)
    lo, hi = np.percentile(means, args.bbox_lo, 0), np.percentile(means, args.bbox_hi, 0)
    keep = keep & torch.from_numpy(np.all((means > lo) & (means < hi), 1))
    idx = torch.nonzero(keep).squeeze(1)
    print(f"after attr/bbox prune: {len(idx)} / {len(opac)}", flush=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means[idx.numpy()])
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    final = idx[torch.tensor(ind, dtype=torch.long)]
    print(f"after outlier removal: {len(final)} gaussians", flush=True)

    out = {k: sp[k][final].contiguous() for k in ["means", "scales", "quats", "opacities", "sh0", "shN"]}
    torch.save({"splats": out}, args.out_ckpt)
    print(f"wrote {args.out_ckpt}", flush=True)
    if args.out_ply:
        write_3dgs_ply(args.out_ply, out)
        print(f"wrote {args.out_ply}", flush=True)


if __name__ == "__main__":
    main()
