"""inhabit-kernel: a from-scratch, unified, GPU-vectorised volumetric fusion core,
now with object-aware fusion (Phase 1: separate objects from the room).

Fusion (Phase 0):
  * forward ray-sampling integration, TSDF at the voxel centre -> no grazing holes;
  * confidence-weighted, robust (Huber) updates -> rejects depth noise;
  * free-space carving -> removes floaters;
  * GPU field denoise before meshing -> fast meshing + crisp surface;
  * unseen voxels default to occupied -> no back-shell.

Decomposition (Phase 1):
  * per-voxel object votes are accumulated from per-pixel instance ids during fusion;
  * extract_mesh returns a per-vertex object label, so the single fused mesh can be
    split into a room shell + one mesh per object, each independently meshed.
"""
from __future__ import annotations
import numpy as np
import torch


class InhabitKernel:
    def __init__(self, vol_min, vol_max, voxel=0.02, trunc_vox=3.0,
                 device="cuda", huber=0.5, robust=True, carve_w=0.5, n_labels=0):
        self.device = device
        self.voxel = float(voxel)
        self.trunc = trunc_vox * self.voxel
        self.huber = huber
        self.robust = robust
        self.carve_w = carve_w
        self.n_labels = n_labels
        vmin = torch.as_tensor(vol_min, dtype=torch.float32)
        vmax = torch.as_tensor(vol_max, dtype=torch.float32)
        self.origin = vmin.clone()
        dims = torch.ceil((vmax - vmin) / self.voxel).long()
        self.dims = tuple(int(x) for x in dims)
        self.N = int(np.prod(self.dims))
        self.wsum = torch.zeros(self.N, device=device)
        self.dsum = torch.zeros(self.N, device=device)
        self.votes = (torch.zeros(self.N * n_labels, device=device)
                      if n_labels > 0 else None)

    def _bounds(self, P, w, nx, ny, nz):
        idx = ((P - self.origin.to(self.device)) / self.voxel).floor().long()
        inb = ((idx[:, 0] >= 0) & (idx[:, 0] < nx) & (idx[:, 1] >= 0) & (idx[:, 1] < ny)
               & (idx[:, 2] >= 0) & (idx[:, 2] < nz) & (w > 0))
        lin = (idx[:, 0].clamp(0, nx - 1) * ny + idx[:, 1].clamp(0, ny - 1)) * nz \
            + idx[:, 2].clamp(0, nz - 1)
        return idx, lin, inb

    @torch.no_grad()
    def integrate(self, depth, pose, K, labels=None, conf=None):
        dev = self.device
        H, W = depth.shape
        fx, fy, cx, cy = K
        R = pose[:3, :3].to(dev); tcam = pose[:3, 3].to(dev)
        depth = depth.to(dev)
        nx, ny, nz = self.dims
        go = self.origin.to(dev)
        mu, voxel = self.trunc, self.voxel

        valid = depth > 0
        if not bool(valid.any()):
            return
        vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                                indexing="ij")
        u = uu[valid].float(); v = vv[valid].float()
        dmeas = depth[valid]
        lab = labels.to(dev)[valid] if (labels is not None and self.votes is not None) else None
        x = (u - cx) / fx; y = (v - cy) / fy
        d_cam = torch.stack([x, y, torch.ones_like(x)], 1)
        d_world = d_cam @ R.T
        zaxis = R[:, 2].to(dev)
        w_conf = (1.0 / (dmeas * dmeas + 1e-3)) if self.robust else torch.ones_like(dmeas)
        if conf is not None:                                    # per-pixel model confidence
            cf = conf.to(dev)[valid]
            w_conf = w_conf * (cf / (cf.median() + 1e-6)).clamp(0.05, 5.0)

        # surface band (TSDF at voxel centre)
        nb = max(int(round(2 * mu / voxel)) + 1, 3)
        offs = torch.linspace(-mu, mu, nb, device=dev)
        z = dmeas[:, None] + offs[None, :]
        Pw = tcam[None, None, :] + z[..., None] * d_world[:, None, :]
        if self.robust:
            r = offs.abs() / mu
            w_rob = torch.where(r <= self.huber, torch.ones_like(r),
                                self.huber / (r + 1e-6))[None, :]
        else:
            w_rob = torch.ones_like(offs)[None, :]
        w = (w_conf[:, None] * w_rob).expand_as(z)
        Pflat = Pw.reshape(-1, 3); wflat = w.reshape(-1)
        idx, lin, inb = self._bounds(Pflat, wflat, nx, ny, nz)
        Cw = (idx.float() + 0.5) * voxel + go
        zc = (Cw - tcam) @ zaxis
        dmeas_s = dmeas[:, None].expand_as(z).reshape(-1)
        sdf = (dmeas_s - zc).clamp(-mu, mu)
        sel = inb
        self.dsum.index_add_(0, lin[sel], (sdf * wflat)[sel])
        self.wsum.index_add_(0, lin[sel], wflat[sel])

        # object votes from near-surface samples
        if lab is not None:
            near = sel & (sdf.abs() < voxel)
            lab_s = lab[:, None].expand_as(z).reshape(-1)[near]
            vlin = lin[near] * self.n_labels + lab_s.clamp(0, self.n_labels - 1)
            self.votes.index_add_(0, vlin, wflat[near])

        # free-space carve
        nf = 4
        znear = 0.2
        far = torch.clamp(dmeas - mu, min=znear)
        fr = torch.linspace(0.0, 1.0, nf, device=dev)[None, :]
        zf = znear + fr * (far[:, None] - znear)
        Pf = tcam[None, None, :] + zf[..., None] * d_world[:, None, :]
        valf = ((far[:, None] - znear) > 0).expand_as(zf).reshape(-1)
        wf = (self.carve_w * w_conf[:, None]).expand_as(zf).reshape(-1) * valf
        Pff = Pf.reshape(-1, 3)
        _, linf, inbf = self._bounds(Pff, wf, nx, ny, nz)
        sf = inbf
        self.dsum.index_add_(0, linf[sf], torch.full_like(wf[sf], mu) * wf[sf])
        self.wsum.index_add_(0, linf[sf], wf[sf])

    def _denoise(self, vol_t, bilateral=True):
        import torch.nn.functional as Fn
        if not bilateral:
            k1 = torch.tensor([0.25, 0.5, 0.25], device=self.device)
            vv = vol_t[None, None]
            for d in range(3):
                kk = k1.view(1, 1, *[3 if i == d else 1 for i in range(3)])
                vv = Fn.conv3d(vv, kk, padding=tuple(1 if i == d else 0 for i in range(3)))
            return vv[0, 0]
        # edge-preserving: weight neighbours by TSDF similarity so the zero-crossing
        # (the surface) is not smeared, but noise on either side is averaged out.
        v = vol_t[None, None]
        pad = Fn.pad(v, (1, 1, 1, 1, 1, 1), mode="replicate")
        sr = 0.5 * self.trunc
        acc = torch.zeros_like(v); wsum = torch.zeros_like(v)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = pad[:, :, 1 + dx:1 + dx + v.shape[2],
                             1 + dy:1 + dy + v.shape[3], 1 + dz:1 + dz + v.shape[4]]
                    w = torch.exp(-((nb - v) ** 2) / (2 * sr * sr))
                    acc += w * nb; wsum += w
        return (acc / wsum.clamp(min=1e-6))[0, 0]

    @torch.no_grad()
    def _surface_nets(self, vol):
        """From-scratch vectorised Surface Nets on GPU. Returns (verts, quads-as-tris)."""
        dev = vol.device; L = 0.0; vx = self.voxel; org = self.origin.to(dev)
        c = torch.stack([vol[:-1, :-1, :-1], vol[1:, :-1, :-1], vol[:-1, 1:, :-1],
                         vol[1:, 1:, :-1], vol[:-1, :-1, 1:], vol[1:, :-1, 1:],
                         vol[:-1, 1:, 1:], vol[1:, 1:, 1:]], -1)          # [cx,cy,cz,8]
        offs = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                             [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
                            device=dev, dtype=torch.float32)
        nin = (c < L).sum(-1)
        active = (nin > 0) & (nin < 8)
        edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6),
                 (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        cx, cy, cz = c.shape[:3]
        pos = torch.zeros((cx, cy, cz, 3), device=dev); cnt = torch.zeros((cx, cy, cz), device=dev)
        for a, b in edges:
            fa, fb = c[..., a], c[..., b]
            strad = (fa < L) != (fb < L)
            t = ((L - fa) / (fb - fa + 1e-9)).clamp(0, 1)
            p = offs[a] + t[..., None] * (offs[b] - offs[a])
            pos += torch.where(strad[..., None], p, torch.zeros_like(p))
            cnt += strad.float()
        local = pos / cnt.clamp(min=1)[..., None]
        I, J, K = torch.meshgrid(torch.arange(cx, device=dev), torch.arange(cy, device=dev),
                                 torch.arange(cz, device=dev), indexing="ij")
        vert = (torch.stack([I, J, K], -1).float() + local) * vx + org
        vid = torch.full((cx, cy, cz), -1, dtype=torch.long, device=dev)
        vid[active] = torch.arange(int(active.sum()), device=dev)
        verts = vert[active]
        ins = vol < L
        tris = []

        def quads(s, a, b, cc, d, front):
            mk = s & (a >= 0) & (b >= 0) & (cc >= 0) & (d >= 0)
            a, b, cc, d, fr = a[mk], b[mk], cc[mk], d[mk], front[mk]
            t1 = torch.stack([a, b, d], 1); t2 = torch.stack([a, d, cc], 1)
            fl = ~fr
            t1[fl] = t1[fl].flip(1); t2[fl] = t2[fl].flip(1)
            tris.append(torch.cat([t1, t2], 0))
        # x-faces
        sx = (ins[:-1] != ins[1:])[:, 1:-1, 1:-1]
        quads(sx, vid[:, :-1, :-1], vid[:, 1:, :-1], vid[:, :-1, 1:], vid[:, 1:, 1:],
              ins[:-1][:, 1:-1, 1:-1])
        # y-faces
        sy = (ins[:, :-1] != ins[:, 1:])[1:-1, :, 1:-1]
        quads(sy, vid[:-1, :, :-1], vid[1:, :, :-1], vid[:-1, :, 1:], vid[1:, :, 1:],
              ins[:, :-1][1:-1, :, 1:-1])
        # z-faces
        sz = (ins[:, :, :-1] != ins[:, :, 1:])[1:-1, 1:-1, :]
        quads(sz, vid[:-1, :-1, :], vid[1:, :-1, :], vid[:-1, 1:, :], vid[1:, 1:, :],
              ins[:, :, :-1][1:-1, 1:-1, :])
        faces = torch.cat(tris, 0) if tris else torch.zeros((0, 3), dtype=torch.long, device=dev)
        return verts, faces, active

    @torch.no_grad()
    def extract_mesh(self, smooth=True, mesher="surfacenets", bilateral=True):
        tsdf = torch.full((self.N,), -self.trunc, device=self.device)
        m = self.wsum > 0
        tsdf[m] = (self.dsum[m] / self.wsum[m])
        vol_t = tsdf.reshape(self.dims)
        if smooth:
            vol_t = self._denoise(vol_t, bilateral=bilateral)
        if mesher == "surfacenets":
            verts, faces, active = self._surface_nets(vol_t)
            labels = None
            if self.votes is not None and len(verts):
                nx, ny, nz = self.dims
                ai = active.nonzero(as_tuple=False)                       # cell ijk per vertex
                lin = (ai[:, 0].clamp(0, nx - 1) * ny + ai[:, 1].clamp(0, ny - 1)) * nz \
                    + ai[:, 2].clamp(0, nz - 1)
                labels = self.votes.view(self.N, self.n_labels)[lin].argmax(1).cpu().numpy()
            return verts.cpu().numpy().astype(np.float32), faces.cpu().numpy().astype(np.int64), labels
        # fallback: skimage marching cubes
        from skimage import measure
        try:
            verts, faces, _, _ = measure.marching_cubes(vol_t.cpu().numpy(), level=0.0,
                                                        spacing=(self.voxel,) * 3)
        except (ValueError, RuntimeError):
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64), None
        verts = (verts + self.origin.numpy()).astype(np.float32)
        labels = None
        if self.votes is not None and len(verts):
            nx, ny, nz = self.dims
            V = torch.from_numpy(verts).to(self.device)
            idx = ((V - self.origin.to(self.device)) / self.voxel).floor().long()
            idx[:, 0].clamp_(0, nx - 1); idx[:, 1].clamp_(0, ny - 1); idx[:, 2].clamp_(0, nz - 1)
            lin = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
            labels = self.votes.view(self.N, self.n_labels)[lin].argmax(1).cpu().numpy()
        return verts, faces.astype(np.int64), labels
