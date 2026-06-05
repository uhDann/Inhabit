"""Object separation that actually isolates whole furniture: remove only the BOUNDARY
structural planes (floor at min-z, ceiling at max-z, walls at x/y extremes) via RANSAC,
keep interior furniture planes, then take mesh CONNECTED COMPONENTS = whole instances.

Quality tracks geometry completeness -- run it on the splat-derived mesh (most complete),
not a thin/decimated reconstruction. Earlier SAM2-vote / pure-affinity methods mislabel the
floor (swallow furniture into the shell, or split the floor into 'objects'); this does not.

Run: python -m separation.instances --mesh viz/mesh_from_splat.ply --out runs/inst
"""
from __future__ import annotations
import argparse, os, collections
import numpy as np, open3d as o3d, trimesh, colorsys


def separate(V, F, boundary_frac=0.18, axis_thr=0.82, min_comp=400):
    N = len(V); bbmin, bbmax = V.min(0), V.max(0); ext = bbmax - bbmin
    shell = np.zeros(N, bool); remaining = np.arange(N)
    for _ in range(14):
        if len(remaining) < 0.02 * N:
            break
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[remaining]))
        plane, inl = sub.segment_plane(0.04, 3, 1000)
        if len(inl) < 0.02 * N:
            break
        pts = remaining[inl]; n = np.array(plane[:3]); n /= np.linalg.norm(n) + 1e-9
        axis = int(np.argmax(np.abs(n))); a = np.abs(n).max(); cen = V[pts].mean(0); structural = False
        if a > axis_thr:                                   # axis-aligned plane
            lo = cen[axis] < bbmin[axis] + boundary_frac * ext[axis]
            hi = cen[axis] > bbmax[axis] - boundary_frac * ext[axis]
            structural = lo or hi                          # only if at the scene boundary
        if structural:
            shell[pts] = True
        remaining = np.delete(remaining, inl)
    # connected components on the non-shell faces -> whole instances
    objF = F[~shell[F].any(1)]; g = collections.defaultdict(set)
    for x, y, z in objF:
        g[x].update((y, z)); g[y].update((x, z)); g[z].update((x, y))
    seen = np.zeros(N, bool); comp = np.full(N, -1, np.int32); cid = 0
    for s in np.unique(objF):
        if seen[s]:
            continue
        st = [s]; seen[s] = True
        while st:
            u = st.pop(); comp[u] = cid
            for w in g[u]:
                if not seen[w]:
                    seen[w] = True; st.append(w)
        cid += 1
    vlab = np.zeros(N, np.int32); oid = 0
    for c in range(cid):
        m = comp == c
        if m.sum() >= min_comp:
            oid += 1; vlab[m] = oid
    return vlab, oid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True); ap.add_argument("--out", default="runs/inst")
    ap.add_argument("--min_comp", type=int, default=400)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    om = o3d.io.read_triangle_mesh(a.mesh); V = np.asarray(om.vertices); F = np.asarray(om.triangles)
    vlab, oid = separate(V, F, min_comp=a.min_comp)
    print(f"instances {oid}  object share {100*(vlab>0).sum()/len(V):.0f}%", flush=True)
    lf = vlab[F]; face_lab = np.where((lf[:, 0] == lf[:, 1]) & (lf[:, 1] == lf[:, 2]), lf[:, 0], 0)
    trimesh.Trimesh(V, F[face_lab == 0], process=False).export(f"{a.out}/shell.ply")
    meshes = []; oids = [o for o in np.unique(face_lab) if o != 0]
    for i, o in enumerate(oids):
        subf = F[face_lab == o]; used = np.unique(subf); rm = {int(x): k for k, x in enumerate(used)}
        mo = trimesh.Trimesh(V[used], np.vectorize(rm.get)(subf), process=False)
        h = (i * 0.61803) % 1.0; r, gg, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        mo.visual.vertex_colors = np.tile([int(r * 255), int(gg * 255), int(b * 255), 255], (len(used), 1)).astype(np.uint8)
        mo.export(f"{a.out}/object_{i:02d}.ply"); meshes.append(mo)
    if meshes:
        trimesh.util.concatenate(meshes).export(f"{a.out}/objects.ply")
    print("INSTANCES_DONE", flush=True)


if __name__ == "__main__":
    main()
