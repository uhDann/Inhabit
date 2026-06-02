"""From-scratch synthetic room: analytic primitives, analytic ray-cast depth.

No external renderer, no mesh library for the camera math. A room (interior of an
axis-aligned box) with two separable objects inside (a sphere and a smaller box).
Cameras live inside the room. Depth is computed by closed-form ray-primitive
intersection, fully vectorized in torch so it runs on GPU.

Conventions (OpenCV camera): +z forward, +x right, +y down. A ray is
  P(s) = c + s * (R @ [x, y, 1]),  x=(u-cx)/f, y=(v-cy)/f
so the scalar s equals the camera-space z-depth of the hit (the quantity TSDF uses).
"""
from __future__ import annotations
import numpy as np
import torch

INF = 1e9


# ---------------------------------------------------------------- scene geometry
class Scene:
    def __init__(self, device="cpu"):
        self.device = device
        # room interior box [0,Lx] x [0,Ly] x [0,Lz]  (metres)
        self.room_min = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.room_max = torch.tensor([4.0, 2.6, 5.0], device=device)
        # object 1: sphere
        self.sph_c = torch.tensor([1.2, 0.5, 2.2], device=device)
        self.sph_r = 0.5
        # object 2: solid box (a table-ish block)
        self.box_min = torch.tensor([2.4, 0.0, 3.0], device=device)
        self.box_max = torch.tensor([3.4, 0.9, 3.8], device=device)

    # ---- ray helpers (all vectorised over rays of shape [...,3]) ----
    def _hit_sphere(self, o, d, c, r):
        oc = o - c
        b = (oc * d).sum(-1)
        cc = (oc * oc).sum(-1) - r * r
        a = (d * d).sum(-1)
        disc = b * b - a * cc
        t = torch.full_like(b, INF)
        m = disc >= 0
        sq = torch.sqrt(torch.clamp(disc, min=0))
        t0 = (-b - sq) / a
        t1 = (-b + sq) / a
        tt = torch.where(t0 > 1e-4, t0, t1)
        t = torch.where(m & (tt > 1e-4), tt, t)
        return t

    def _slab(self, o, d, bmin, bmax):
        # returns (t_near, t_far); valid if t_far > max(t_near,0)
        inv = 1.0 / d
        t1 = (bmin - o) * inv
        t2 = (bmax - o) * inv
        tmin = torch.minimum(t1, t2).max(-1).values
        tmax = torch.maximum(t1, t2).min(-1).values
        return tmin, tmax

    def _hit_box_outside(self, o, d, bmin, bmax):
        tmin, tmax = self._slab(o, d, bmin, bmax)
        hit = (tmax > torch.clamp(tmin, min=0)) & (tmax > 1e-4)
        t = torch.where(hit & (tmin > 1e-4), tmin, torch.full_like(tmin, INF))
        return t

    def _hit_room_inside(self, o, d, bmin, bmax):
        # camera is inside -> the wall hit is t_far
        _, tmax = self._slab(o, d, bmin, bmax)
        return torch.where(tmax > 1e-4, tmax, torch.full_like(tmax, INF))

    def render(self, pose, K, W, H):
        """pose: 4x4 cam->world. K: (fx,fy,cx,cy). Returns depth[H,W], objid[H,W]."""
        dev = self.device
        fx, fy, cx, cy = K
        u = torch.arange(W, device=dev).float()
        v = torch.arange(H, device=dev).float()
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        x = (uu - cx) / fx
        y = (vv - cy) / fy
        d_cam = torch.stack([x, y, torch.ones_like(x)], -1)        # [H,W,3], z=1
        R = pose[:3, :3].to(dev)
        c = pose[:3, 3].to(dev)
        d = d_cam @ R.T                                            # world dirs
        o = c.expand_as(d)
        ts = torch.stack([
            self._hit_room_inside(o, d, self.room_min, self.room_max),
            self._hit_sphere(o, d, self.sph_c, self.sph_r),
            self._hit_box_outside(o, d, self.box_min, self.box_max),
        ], 0)                                                      # [3,H,W]
        t, obj = ts.min(0)                                         # nearest hit
        depth = t                                                 # s == z-depth
        depth[t >= INF] = 0.0
        obj[t >= INF] = -1
        return depth, obj

    # ---- dense GT surface point cloud (for Chamfer) ----
    def gt_points(self, n=400_000, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        pts, lab = [], []
        rmin = self.room_min.cpu(); rmax = self.room_max.cpu()
        L = (rmax - rmin)
        # room: 6 interior faces, area-weighted
        faces = [
            (0, rmin[0]), (0, rmax[0]), (1, rmin[1]), (1, rmax[1]), (2, rmin[2]), (2, rmax[2]),
        ]
        areas = []
        for ax, _ in faces:
            other = [a for a in range(3) if a != ax]
            areas.append(float(L[other[0]] * L[other[1]]))
        areas = np.array(areas); areas /= areas.sum()
        nroom = int(0.6 * n)
        cnt = (areas * nroom).astype(int)
        for (ax, val), k in zip(faces, cnt):
            p = torch.rand(k, 3, generator=g) * L + rmin
            p[:, ax] = val
            pts.append(p); lab.append(torch.zeros(k, dtype=torch.long))
        # sphere
        ns = int(0.22 * n)
        z = torch.randn(ns, 3, generator=g); z = z / z.norm(dim=1, keepdim=True)
        pts.append(z * self.sph_r + self.sph_c.cpu()); lab.append(torch.ones(ns, dtype=torch.long))
        # inner box: 6 faces
        nb = n - sum(x.shape[0] for x in pts)
        bmin = self.box_min.cpu(); bmax = self.box_max.cpu(); Lb = bmax - bmin
        per = max(nb // 6, 1)
        bfaces = [(0, bmin[0]), (0, bmax[0]), (1, bmin[1]), (1, bmax[1]), (2, bmin[2]), (2, bmax[2])]
        for ax, val in bfaces:
            p = torch.rand(per, 3, generator=g) * Lb + bmin
            p[:, ax] = val
            pts.append(p); lab.append(torch.full((per,), 2, dtype=torch.long))
        P = torch.cat(pts, 0); Lb_ = torch.cat(lab, 0)
        return P.numpy().astype(np.float32), Lb_.numpy()


# ---------------------------------------------------------------- camera poses
def look_at(eye, target, up=(0, -1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f = f / np.linalg.norm(f)          # +z forward
    r = np.cross(f, up); r = r / np.linalg.norm(r)        # +x right
    d = np.cross(f, r)                                    # +y down
    R = np.stack([r, d, f], 1)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = eye
    return T.astype(np.float32)


def ring_poses(scene, n=40, radius=1.4, height=1.3):
    cen = ((scene.room_min + scene.room_max) / 2).cpu().numpy()
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = cen + np.array([radius * np.cos(a), height - cen[1], radius * np.sin(a)])
        tgt = cen + np.array([0.6 * np.cos(a + 0.5), 0.0, 0.6 * np.sin(a + 0.5)])
        poses.append(look_at(eye, tgt))
    return poses


def intrinsics(W, H, hfov_deg=70.0):
    f = 0.5 * W / np.tan(0.5 * np.radians(hfov_deg))
    return (f, f, (W - 1) / 2.0, (H - 1) / 2.0)
