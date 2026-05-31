"""Make ONE complete, textured, watertight room from PGSR by screened-Poisson
reconstruction.

PGSR gives a crisp, textured surface but with open gaps in unobserved corners.
Poisson reconstruction fits a single closed surface to PGSR's oriented, coloured
points: where PGSR is dense it follows it tightly (keeps the texture + detail);
across the gaps it smoothly bridges using the surrounding geometry + normals,
interpolating colour. Density trimming then removes only the wildest
extrapolations so we don't balloon to infinity. The result is a single
watertight mesh -- a complete room -- without MonoSDF's outward-ballooned shell.

Optionally augments PGSR's points with MonoSDF points that fall in genuine PGSR
gaps (nearest PGSR vertex farther than --augment-dist), to give Poisson real
evidence in the corners instead of pure extrapolation. Those augment points are
coloured neutral (honest: no texture was observed there).

Usage:
    python fuse_poisson.py --pgsr pgsr_post.ply --out unified_room.ply \
        --depth 10 --trim-quantile 0.03 --tris 350000 \
        [--mono monosdf_world.ply --augment-dist 0.6]
"""
from __future__ import annotations
import argparse, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgsr", required=True)
    ap.add_argument("--mono", default="", help="optional MonoSDF mesh to augment genuine gaps")
    ap.add_argument("--augment-dist", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--trim-quantile", type=float, default=0.03)
    ap.add_argument("--tris", type=int, default=350000)
    ap.add_argument("--neutral", type=float, nargs=3, default=[0.62, 0.60, 0.57])
    a = ap.parse_args()

    import open3d as o3d

    t0 = time.time()
    pg = o3d.io.read_triangle_mesh(a.pgsr)
    pg.compute_vertex_normals()
    Vp = np.asarray(pg.vertices)
    Cp = np.asarray(pg.vertex_colors)
    Np = np.asarray(pg.vertex_normals)
    print(f"PGSR verts={len(Vp)} colors={pg.has_vertex_colors()}", flush=True)

    pts = [Vp]; cols = [Cp]; nors = [Np]

    if a.mono:
        from scipy.spatial import cKDTree
        mo = o3d.io.read_triangle_mesh(a.mono)
        mo.compute_vertex_normals()
        Vm = np.asarray(mo.vertices)
        Nm = np.asarray(mo.vertex_normals)
        tree = cKDTree(Vp)
        dist, _ = tree.query(Vm, k=1, workers=-1)
        gap = dist > a.augment_dist
        print(f"MonoSDF augment: {gap.sum()}/{len(Vm)} verts in genuine gaps "
              f"(dist>{a.augment_dist})", flush=True)
        neutral = np.array(a.neutral)
        pts.append(Vm[gap]); nors.append(Nm[gap])
        cols.append(np.tile(neutral, (gap.sum(), 1)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.concatenate(cols), 0, 1))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate(nors))
    print(f"combined point cloud: {len(pcd.points)} pts ({time.time()-t0:.1f}s)", flush=True)

    t1 = time.time()
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=a.depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    print(f"poisson depth={a.depth}: verts={len(mesh.vertices)} tris={len(mesh.triangles)} "
          f"({time.time()-t1:.1f}s)", flush=True)

    # trim the lowest-density (most-extrapolated) vertices
    thr = np.quantile(dens, a.trim_quantile)
    rm = dens < thr
    mesh.remove_vertices_by_mask(rm)
    print(f"trimmed {rm.sum()} verts below density q={a.trim_quantile} "
          f"-> verts={len(mesh.vertices)} tris={len(mesh.triangles)}", flush=True)

    # largest component
    labels, ccs, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels); areas = np.asarray(areas)
    keep = labels == int(np.argmax(areas))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep])
    mesh.remove_unreferenced_vertices()
    print(f"LCC kept {keep.sum()}/{len(labels)} tris ({len(ccs)} components)", flush=True)

    if len(mesh.triangles) > a.tris:
        t2 = time.time()
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=a.tris)
        print(f"decimated -> verts={len(mesh.vertices)} tris={len(mesh.triangles)} "
              f"({time.time()-t2:.1f}s)", flush=True)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, mesh, write_ascii=False, compressed=True)
    try:
        import trimesh
        tm = trimesh.load(a.out, process=False)
        wt = tm.is_watertight
    except Exception:
        wt = "?"
    bb = mesh.get_axis_aligned_bounding_box()
    print(f"wrote {a.out}  bbox={np.round(bb.get_extent(),2)}  watertight={wt}", flush=True)
    print("FUSE_POISSON_DONE", flush=True)


if __name__ == "__main__":
    main()
