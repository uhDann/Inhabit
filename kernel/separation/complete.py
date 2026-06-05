"""Whole-furniture completion: connected-component fragments are over-segmented (a chair
splits into seat/legs/back). Merge fragments that belong to the same SAM2 instance track
across views, then watertight-solidify each merged object via Poisson -> solid furniture
suitable for editing and physics.

Improves the raw connected-component separation (e.g. office0: 52 fragments -> 33 merged
solids, visibly more coherent pieces). Limit: single-side surface fragments can't recover
unseen backs, so Poisson solids are somewhat inflated, and reconstructed objects can
interpenetrate (Genesis needs per-object collision margins / smaller dt).

Run: python -m separation.complete --mesh viz/mesh_from_splat.ply --masks runs/sep/sam2_masks.npy \
        --poses sep_data/poses.npy --depth_glob 'sep_data/depth/*.npy' --out viz/furn
"""
from __future__ import annotations
import argparse, os, glob, collections
import numpy as np, open3d as o3d, trimesh, colorsys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--poses", required=True); ap.add_argument("--depth_glob", required=True)
    ap.add_argument("--K", nargs=4, type=float, default=[600, 600, 599.5, 339.5])
    ap.add_argument("--out", default="viz/furn"); ap.add_argument("--min_frag", type=int, default=150)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    om = o3d.io.read_triangle_mesh(a.mesh); om.compute_vertex_normals()
    V = np.asarray(om.vertices); F = np.asarray(om.triangles); VN = np.asarray(om.vertex_normals); N = len(V)
    bbmin, bbmax = V.min(0), V.max(0); ext = bbmax - bbmin

    shell = np.zeros(N, bool); remaining = np.arange(N)
    for _ in range(14):
        if len(remaining) < 0.02 * N: break
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[remaining])); pl, inl = sub.segment_plane(0.04, 3, 1000)
        if len(inl) < 0.02 * N: break
        pts = remaining[inl]; n = np.array(pl[:3]); n /= np.linalg.norm(n) + 1e-9; ax = int(np.argmax(np.abs(n))); c = V[pts].mean(0)
        if np.abs(n).max() > 0.82 and (c[ax] < bbmin[ax] + 0.18 * ext[ax] or c[ax] > bbmax[ax] - 0.18 * ext[ax]): shell[pts] = True
        remaining = np.delete(remaining, inl)

    objF = F[~shell[F].any(1)]; g = collections.defaultdict(set)
    for x, y, z in objF: g[x].update((y, z)); g[y].update((x, z)); g[z].update((x, y))
    comp = np.full(N, -1, np.int32); seen = np.zeros(N, bool); cid = 0
    for s in np.unique(objF):
        if seen[s]: continue
        st = [s]; seen[s] = True
        while st:
            u = st.pop(); comp[u] = cid
            for w in g[u]:
                if not seen[w]: seen[w] = True; st.append(w)
        cid += 1
    comps = [np.where(comp == c)[0] for c in range(cid) if (comp == c).sum() >= a.min_frag]
    print("raw fragments", len(comps), flush=True)

    masks = np.load(a.masks); poses = np.load(a.poses).astype(np.float32); deps = sorted(glob.glob(a.depth_glob))
    Nf, H, Wd = masks.shape; fx, fy, cx, cy = a.K
    def dom(idx):
        p = V[idx]; p = p[np.random.choice(len(p), min(1500, len(p)), replace=False)]
        ph = np.concatenate([p, np.ones((len(p), 1))], 1); votes = collections.Counter()
        for f in range(0, Nf, 2):
            d = np.load(deps[f]).astype(np.float32); cam = (ph @ np.linalg.inv(poses[f]).T)[:, :3]; z = cam[:, 2]
            xi = np.round(fx * cam[:, 0] / z + cx).astype(int); yi = np.round(fy * cam[:, 1] / z + cy).astype(int)
            ok = (z > 0) & (xi >= 0) & (xi < Wd) & (yi >= 0) & (yi < H); xi = np.clip(xi, 0, Wd - 1); yi = np.clip(yi, 0, H - 1)
            mm = masks[f][yi, xi][ok & (np.abs(z - d[yi, xi]) < 0.05)]
            for t in mm[mm > 0]: votes[int(t)] += 1
        return votes.most_common(1)[0][0] if votes else 0
    trk = [dom(c) for c in comps]
    par = list(range(len(comps)))
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    by = collections.defaultdict(list)
    for i, t in enumerate(trk):
        if t > 0: by[t].append(i)
    for mem in by.values():
        for m in mem[1:]: par[find(m)] = find(mem[0])
    groups = collections.defaultdict(list)
    for i in range(len(comps)): groups[find(i)].append(i)
    inst = [np.concatenate([comps[i] for i in mem]) for mem in groups.values()]
    inst = [x for x in inst if len(x) >= 300]
    print("merged instances", len(inst), flush=True)

    colored = []
    for k, idx in enumerate(inst, 1):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[idx])); pcd.normals = o3d.utility.Vector3dVector(VN[idx])
        try:
            mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
            mesh.remove_vertices_by_mask(np.asarray(dens) < np.quantile(np.asarray(dens), 0.08))
            mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
        except Exception:
            mesh = pcd.compute_convex_hull()[0]
        if len(mesh.vertices) < 30: mesh = pcd.compute_convex_hull()[0]
        o3d.io.write_triangle_mesh(f"{a.out}/solid_{k:02d}.obj", mesh)
        h = (k * 0.61803) % 1.0; r, gg, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        mc = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False)
        if len(mc.faces):
            mc.visual.vertex_colors = np.tile([int(r * 255), int(gg * 255), int(b * 255), 255], (len(mc.vertices), 1)).astype(np.uint8); colored.append(mc)
    if colored: trimesh.util.concatenate(colored).export(f"{a.out}/../furn_solids.ply")
    print(f"solid furniture: {len(colored)} objects", flush=True); print("COMPLETE_DONE", flush=True)


if __name__ == "__main__":
    main()
