"""Separation driver: real SAM2 masks -> kernel vote-fusion -> superpoint pooling ->
room shell + per-object meshes, on a real posed-depth capture.

This is the Tier-0+Tier-1 fix from RESEARCH_ROADMAP.md, wired to our existing kernel
(which already accumulates per-voxel object votes during fusion). The only new inputs
are (a) real, view-consistent SAM2 masks and (b) superpoint pooling of the votes.

Requires (GPU): the kernel (kernel.py), sam2, torch, trimesh, numpy.
Inputs: posed depth maps + poses + intrinsics (any source: GT, sensor, or our
feed-forward front end) + a frames dir for SAM2.
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import torch
import trimesh

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # kernel/
from kernel import InhabitKernel
from separation.seg import sam2_video_masks
from separation.superpoints import superpoints, pool_votes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="jpg frames for SAM2 (temporal order)")
    ap.add_argument("--depth_glob", required=True, help="glob of depth .npy/.png aligned to frames")
    ap.add_argument("--poses", required=True, help="[N,4,4] cam->world .npy")
    ap.add_argument("--K", nargs=4, type=float, required=True)
    ap.add_argument("--bounds", nargs=6, type=float, required=True, help="xmin ymin zmin xmax ymax zmax")
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--sam2_cfg", required=True); ap.add_argument("--sam2_ckpt", required=True)
    ap.add_argument("--out", default="runs/separation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    K = tuple(a.K)

    # 1. real view-consistent masks
    print("running SAM2 video masks...", flush=True)
    masks = sam2_video_masks(a.frames_dir, a.sam2_cfg, a.sam2_ckpt, device=dev)
    n_labels = int(max(m.max() for m in masks)) + 1
    print(f"{len(masks)} frames, {n_labels-1} tracked objects", flush=True)

    # 2. depth + poses
    poses = np.load(a.poses).astype(np.float32)
    dpaths = sorted(glob.glob(a.depth_glob))
    def load_depth(p):
        if p.endswith(".npy"):
            return np.load(p).astype(np.float32)
        import imageio.v2 as iio
        return iio.imread(p).astype(np.float32) / 6553.5     # Replica scale; adapt as needed

    # 3. fuse with the kernel, voting with the SAM2 masks
    lo = a.bounds[:3]; hi = a.bounds[3:]
    ker = InhabitKernel(lo, hi, voxel=a.voxel, trunc_vox=3.0, device=dev, robust=True,
                        n_labels=max(n_labels, 4))
    for i, (dp, pose, lab) in enumerate(zip(dpaths, poses, masks)):
        d = torch.from_numpy(load_depth(dp))
        ker.integrate(d, torch.from_numpy(pose), K, labels=torch.from_numpy(lab))
    verts, faces, _ = ker.extract_mesh()
    print(f"fused mesh: {len(verts):,} verts", flush=True)

    # 4. per-vertex VOTE VECTORS (not just argmax) from the kernel grid
    nx, ny, nz = ker.dims
    V = torch.from_numpy(verts).to(dev)
    idx = ((V - ker.origin.to(dev)) / ker.voxel).floor().long()
    idx[:, 0].clamp_(0, nx - 1); idx[:, 1].clamp_(0, ny - 1); idx[:, 2].clamp_(0, nz - 1)
    lin = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
    vert_votes = ker.votes.view(ker.N, ker.n_labels)[lin].cpu().numpy()   # [V, n_labels]

    # 5. superpoint pooling -> clean per-vertex object labels (shell = 0)
    nrm = np.asarray(trimesh.Trimesh(verts, faces, process=False).vertex_normals, np.float32)
    sp, n_sp = superpoints(verts, faces, nrm)
    vlab = pool_votes(sp, n_sp, vert_votes, ker.n_labels)
    print(f"{n_sp} superpoints, {len(np.unique(vlab))-1} objects after pooling", flush=True)

    # 6. split into room shell + per-object meshes
    lf = vlab[faces]; a3, b3, c3 = lf[:, 0], lf[:, 1], lf[:, 2]
    face_lab = np.where(a3 == b3, a3, np.where(a3 == c3, a3, np.where(b3 == c3, b3, 0)))
    for oid in np.unique(face_lab):
        sub = faces[face_lab == oid]
        if len(sub) < 50:
            continue
        used = np.unique(sub); remap = {int(o): k for k, o in enumerate(used)}
        m = trimesh.Trimesh(verts[used], np.vectorize(remap.get)(sub), process=False)
        name = "room_shell" if oid == 0 else f"object_{int(oid):02d}"
        m.export(f"{a.out}/{name}.ply")
    print("wrote per-object meshes to", a.out)


if __name__ == "__main__":
    main()
