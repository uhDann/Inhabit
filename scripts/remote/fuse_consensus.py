"""Consensus gap-fill fusion (the SAFE kind, per the fusion literature):
trusted backbone + donor methods contribute ONLY where the backbone has a hole.
No blind volumetric averaging -> no doubled walls / smoothed-away detail.

Backbone  = PGSR (textured, planar-clean, hugs the real surface).
Donor 1   = DN-Splatter (also a TSDF mesh that hugs the surface, so it AGREES
            with PGSR geometrically -> safe to fuse; it caught the piano).
            Contributes only its points that are FAR from any PGSR point
            (i.e. fill PGSR's gaps).
Donor 2   = MonoSDF (optional, --use-mono). Its SDF surface balloons ~0.87u
            OUTSIDE the real walls, so it can't be distance-gated cleanly
            (its points are 'far from PGSR' everywhere). Off by default; if on,
            only used where far from BOTH PGSR and DN (true voids).

Prints alignment diagnostics (donor->backbone nearest-neighbour distances) so
we can SEE whether a donor agrees with the backbone before trusting it.

Usage:
    python fuse_consensus.py --pgsr P.ply --dn DN.ply [--mono M.ply --use-mono] \
        --out consensus.ply --tau-dn 0.12 --tau-mono 0.5 --depth 10 --trim 0.02
"""
from __future__ import annotations
import argparse, time
import numpy as np

# inverse of the nerfstudio coolermap dataparser_transforms.json (room run):
#   nerfstudio -> COLMAP  =  [[1,0,0],[0,0,-1],[0,1,0]]
T_DN_TO_COLMAP = np.array([[1, 0, 0, 0],
                           [0, 0, -1, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1]], dtype=np.float64)


def load(path, transform=None, neutral=None):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    if transform is not None:
        m.transform(transform)
        m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    if m.has_vertex_colors() and neutral is None:
        C = np.asarray(m.vertex_colors)
    else:
        C = np.tile(np.array(neutral if neutral else [0.62, 0.60, 0.57]), (len(V), 1))
    return V, N, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgsr", required=True)
    ap.add_argument("--dn", required=True)
    ap.add_argument("--mono", default="")
    ap.add_argument("--use-mono", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-dn", type=float, default=0.12)
    ap.add_argument("--tau-mono", type=float, default=0.5)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--trim", type=float, default=0.02)
    ap.add_argument("--tris", type=int, default=350000)
    ap.add_argument("--clean", action="store_true",
                    help="statistical-outlier-remove the combined cloud before Poisson (kills DN's wispy boundary points that cause lacy walls)")
    ap.add_argument("--crop-z", type=float, default=0.0,
                    help="if >0, clip points beyond this Z half-extent from the centroid (cuts DN's far floaters)")
    a = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    t0 = time.time()
    Vp, Np, Cp = load(a.pgsr)
    Vd, Nd, Cd = load(a.dn, transform=T_DN_TO_COLMAP)
    print(f"PGSR {len(Vp)} pts · DN {len(Vd)} pts ({time.time()-t0:.1f}s)", flush=True)

    treep = cKDTree(Vp)
    # --- alignment diagnostic: does DN agree with PGSR? ---
    dd, _ = treep.query(Vd, k=1, workers=-1)
    print(f"[align] DN->PGSR nn-dist: median={np.median(dd):.3f} p25={np.percentile(dd,25):.3f} "
          f"p75={np.percentile(dd,75):.3f}  (small median => geometrically AGREE)", flush=True)

    # DN contributes only where PGSR has nothing nearby (gap fill)
    dn_gap = dd > a.tau_dn
    print(f"DN gap-fill points (dist>{a.tau_dn}): {dn_gap.sum()} ({100*dn_gap.mean():.1f}% of DN)", flush=True)

    P = [Vp]; N = [Np]; C = [Cp]
    P.append(Vd[dn_gap]); N.append(Nd[dn_gap]); C.append(Cd[dn_gap])

    if a.use_mono and a.mono:
        Vm, Nm, Cm = load(a.mono, neutral=[0.62, 0.60, 0.57])
        dm, _ = treep.query(Vm, k=1, workers=-1)
        print(f"[align] Mono->PGSR nn-dist median={np.median(dm):.3f} (large => ballooned)", flush=True)
        combined = np.concatenate(P)
        treec = cKDTree(combined)
        dmc, _ = treec.query(Vm, k=1, workers=-1)
        mono_void = dmc > a.tau_mono
        print(f"Mono void-fill points (dist>{a.tau_mono}): {mono_void.sum()} "
              f"({100*mono_void.mean():.1f}% of Mono)", flush=True)
        P.append(Vm[mono_void]); N.append(Nm[mono_void]); C.append(Cm[mono_void])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(P))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate(N))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(C), 0, 1))
    print(f"consensus cloud: {len(pcd.points)} pts", flush=True)

    if a.crop_z > 0:
        pts = np.asarray(pcd.points)
        zc = np.median(pts[:, 2])
        keepz = np.abs(pts[:, 2] - zc) < a.crop_z
        pcd = pcd.select_by_index(np.nonzero(keepz)[0])
        print(f"crop-z {a.crop_z}: kept {len(pcd.points)} pts", flush=True)

    if a.clean:
        before = len(pcd.points)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"outlier-removal: {before} -> {len(pcd.points)} pts", flush=True)

    t1 = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=a.depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    print(f"poisson depth={a.depth}: {len(mesh.vertices)} verts ({time.time()-t1:.1f}s)", flush=True)

    if a.trim > 0:
        thr = np.quantile(dens, a.trim)
        mesh.remove_vertices_by_mask(dens < thr)
        print(f"trimmed q={a.trim}: {len(mesh.vertices)} verts", flush=True)

    labels, ccs, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels); areas = np.asarray(areas)
    keep = labels == int(np.argmax(areas))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep])
    mesh.remove_unreferenced_vertices()
    print(f"LCC kept {keep.sum()}/{len(labels)} tris ({len(ccs)} comps)", flush=True)

    if len(mesh.triangles) > a.tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=a.tris)
        print(f"decimated -> {len(mesh.triangles)} tris", flush=True)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, mesh, write_ascii=False, compressed=True)
    bb = mesh.get_axis_aligned_bounding_box()
    print(f"wrote {a.out} bbox={np.round(bb.get_extent(),2)}", flush=True)
    print("CONSENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
