"""Photoreal-mesh core engine: cameras, data, UV unwrap, view-dependent appearance,
and a differentiable nvdiffrast renderer.

Geometry is FIXED (our kernel's metric mesh is the asset). We learn only appearance:
a per-texel feature/diffuse atlas + a tiny deferred MLP that decodes view-dependent
color (the "splat look": specular highlights, glossy floors) -- MobileNeRF/BakedSDF
style, rasterizer-native, no per-frame CNN.

Requires (GPU): torch, nvdiffrast, xatlas, numpy, imageio, Pillow.
Conventions: cameras are OpenCV (pose = cam->world, +z fwd, +x right, +y down), the
same as the rest of the kernel. The OpenCV->OpenGL projection below is the standard
recipe; verify orientation once on first run with a known view (see core_selftest()).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------- cameras
def opengl_projection(fx, fy, cx, cy, W, H, near=0.01, far=100.0):
    """OpenGL clip-space projection from OpenCV pinhole intrinsics.

    The y row is negated so the nvdiffrast rasteriser (bottom-left origin) produces
    images in TOP-left-origin order, matching numpy/PIL image arrays. Without this every
    mesh render is vertically flipped vs the GT frame, which silently caps PSNR (~15 dB)."""
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * fx / W
    P[1, 1] = -2 * fy / H                       # flip y -> top-left-origin output
    P[0, 2] = 1.0 - 2 * cx / W
    P[1, 2] = -(2 * cy / H - 1.0)               # = 1 - 2*cy/H
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1.0
    return P


# OpenCV camera looks down +z, +y down; OpenGL looks down -z, +y up. Flip y,z.
_CV2GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def mvp_from_pose(pose_c2w, K, W, H, near=0.01, far=100.0):
    """Return (mvp [4,4], cam_pos [3]) for a cam->world OpenCV pose."""
    fx, fy, cx, cy = K
    view = _CV2GL @ np.linalg.inv(pose_c2w.astype(np.float32))   # world->GL-cam
    proj = opengl_projection(fx, fy, cx, cy, W, H, near, far)
    mvp = (proj @ view).astype(np.float32)
    cam_pos = pose_c2w[:3, 3].astype(np.float32)
    return mvp, cam_pos


# --------------------------------------------------------------------- mesh / UV
def load_mesh(path):
    """Load a triangle mesh (ply/obj). Returns verts[V,3], faces[F,3], normals[V,3]."""
    import trimesh
    m = trimesh.load(path, process=False)
    v = np.asarray(m.vertices, np.float32)
    f = np.asarray(m.faces, np.int32)
    n = np.asarray(m.vertex_normals, np.float32)
    return v, f, n


def uv_unwrap(verts, faces):
    """xatlas UV unwrap. Returns (verts_x[V2,3], faces_x[F,3] int32, uvs[V2,2] in [0,1]).
    xatlas may duplicate vertices at seams, so it returns a remapped vertex set."""
    import xatlas
    vmap, idx, uvs = xatlas.parametrize(verts, faces)
    # vmap: [V2] original-vertex index per new vertex; idx: [F,3] new faces; uvs:[V2,2]
    verts_x = verts[vmap]
    return verts_x.astype(np.float32), idx.astype(np.int32), uvs.astype(np.float32), vmap


# --------------------------------------------------------------------- appearance
def _posenc(x, n=4):
    """Sinusoidal encoding of a [...,D] tensor -> [..., D*2*n]."""
    out = []
    for i in range(n):
        f = 2.0 ** i * np.pi
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, -1)


class DeferredAppearance(nn.Module):
    """Per-texel diffuse + feature atlas, decoded by a tiny view-dependent MLP.

    color(texel, view, normal) = diffuse(texel) + mlp(feature(texel), enc(view), enc(normal))
    The MLP residual carries the view-dependent specular look; diffuse carries the base.
    """
    def __init__(self, atlas=1024, feat_ch=8, hidden=64):
        super().__init__()
        self.diffuse = nn.Parameter(torch.full((1, atlas, atlas, 3), 0.5))
        self.feat = nn.Parameter(torch.zeros(1, atlas, atlas, feat_ch))
        din = feat_ch + 3 * 2 * 4 + 3 * 2 * 4          # feat + posenc(view) + posenc(normal)
        self.mlp = nn.Sequential(
            nn.Linear(din, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 3))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def shade(self, uv, viewdir, normal, dr):
        """uv,viewdir,normal: [H,W,*] from rasterization. dr = nvdiffrast module."""
        diff = dr.texture(self.diffuse, uv[None], filter_mode="linear")[0]
        feat = dr.texture(self.feat, uv[None], filter_mode="linear")[0]
        x = torch.cat([feat, _posenc(viewdir), _posenc(normal)], -1)
        spec = self.mlp(x)
        return (diff + spec).clamp(0, 1)


# --------------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, device="cuda"):
        import nvdiffrast.torch as dr
        self.dr = dr
        self.glctx = dr.RasterizeCudaContext()
        self.device = device

    def render(self, verts, faces, uvs, normals, appearance, mvp, cam_pos, W, H,
               background=1.0):
        """verts[V,3] faces[F,3] uvs[V,2] normals[V,3] torch on device. Returns [H,W,3]."""
        dr = self.dr
        V = verts.shape[0]
        vh = torch.cat([verts, torch.ones(V, 1, device=verts.device)], 1)   # homog
        clip = vh @ torch.as_tensor(mvp.T, device=verts.device)             # [V,4]
        rast, _ = dr.rasterize(self.glctx, clip[None], faces, (H, W))
        uv_i, _ = dr.interpolate(uvs[None], rast, faces)
        nrm_i, _ = dr.interpolate(normals[None], rast, faces)
        pos_i, _ = dr.interpolate(verts[None], rast, faces)
        view = torch.as_tensor(cam_pos, device=verts.device) - pos_i[0]
        view = view / (view.norm(dim=-1, keepdim=True) + 1e-9)
        nrm = nrm_i[0] / (nrm_i[0].norm(dim=-1, keepdim=True) + 1e-9)
        col = appearance.shade(uv_i[0], view, nrm, dr)                      # [H,W,3]
        bg = torch.full_like(col, background)
        vis = (rast[0, ..., 3:4] > 0).float()
        col = col * vis + bg * (1 - vis)
        col = dr.antialias(col[None], rast, clip[None], faces)[0]          # soft edges
        return col


@torch.no_grad()
def bake_diffuse(renderer, verts, faces, uvs, normals, frames, atlas, W, H, device="cuda"):
    """Multi-view texture baking: project each posed GT frame onto the mesh and
    accumulate per-texel colour, view-weighted (front-facing, fronto-parallel preferred).
    Returns a diffuse atlas [atlas,atlas,3]. This is the stable starting point that makes
    the mesh look like the room; learned view-dependent appearance refines on top.
    """
    dr = renderer.dr
    acc = torch.zeros(atlas * atlas, 3, device=device)
    wsum = torch.zeros(atlas * atlas, 1, device=device)
    vh = torch.cat([verts, torch.ones(verts.shape[0], 1, device=device)], 1)
    for rgb, pose, K in frames:
        gt = torch.as_tensor(rgb, device=device)
        mvp, cp = mvp_from_pose(pose, K, W, H)
        clip = vh @ torch.as_tensor(mvp.T, device=device)
        rast, _ = dr.rasterize(renderer.glctx, clip[None], faces, (H, W))
        uv_i, _ = dr.interpolate(uvs[None], rast, faces)
        nrm_i, _ = dr.interpolate(normals[None], rast, faces)
        pos_i, _ = dr.interpolate(verts[None], rast, faces)
        vis = (rast[0, ..., 3] > 0)
        view = torch.as_tensor(cp, device=device) - pos_i[0]
        view = view / (view.norm(dim=-1, keepdim=True) + 1e-9)
        nrm = nrm_i[0] / (nrm_i[0].norm(dim=-1, keepdim=True) + 1e-9)
        # winding-agnostic (surface-nets normals flip): |cos|, fronto-parallel preferred,
        # but NEVER zero-gate a visible pixel (rasteriser already gives the front surface).
        cos = (nrm * view).sum(-1).abs().clamp(min=0.1)
        w = cos * vis.float()
        u = uv_i[0, ..., 0].clamp(0, 1)
        vv = uv_i[0, ..., 1].clamp(0, 1)
        col = (u * (atlas - 1)).long()                                 # u -> column
        row = ((1.0 - vv) * (atlas - 1)).long()                        # nvdiffrast texture origin is bottom-left
        lin = row * atlas + col
        m = w > 1e-4
        lin_m = lin[m]; rgb_m = gt[m]; w_m = w[m][:, None]
        acc.index_add_(0, lin_m, rgb_m * w_m)
        wsum.index_add_(0, lin_m, w_m)
    diffuse = acc / (wsum + 1e-8)
    diffuse[wsum[:, 0] < 1e-8] = 0.5                                   # unseen texels -> neutral
    return diffuse.view(atlas, atlas, 3)


def core_selftest(mesh_path, pose_c2w, K, W, H, out="selftest.png", device="cuda"):
    """Render the bare mesh (gray) from one pose -- use to verify camera orientation."""
    import imageio.v2 as iio
    v, f, n = load_mesh(mesh_path)
    vx, fx, uv, _ = uv_unwrap(v, f)
    nx = n[_uv_vmap_placeholder(v, vx)] if False else np.zeros_like(vx)  # normals recomputed below
    import trimesh
    nx = np.asarray(trimesh.Trimesh(vx, fx, process=False).vertex_normals, np.float32)
    r = Renderer(device)
    app = DeferredAppearance().to(device)
    mvp, cp = mvp_from_pose(pose_c2w, K, W, H)
    img = r.render(torch.tensor(vx, device=device), torch.tensor(fx, device=device),
                   torch.tensor(uv, device=device), torch.tensor(nx, device=device),
                   app, mvp, cp, W, H)
    iio.imwrite(out, (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    print("wrote", out)
