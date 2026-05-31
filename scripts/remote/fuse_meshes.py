"""Unify two reconstructions into one COMPLETE, TEXTURED room mesh.

Idea (the user's): MonoSDF gives a watertight, *complete* room envelope but no
texture; PGSR gives crisp texture but has gaps in unobserved corners. So use
MonoSDF's geometry as the closed shell and paint PGSR's colour onto it. Where
PGSR saw a surface -> real texture; where it didn't (the corner gaps) ->
MonoSDF's surface is still there and closed, so the room stays complete. The
unobserved patches are tinted toward a neutral wall colour so it's visually
honest about what was actually observed vs. filled.

Both meshes must already be in the same (COLMAP-world) frame -- they are:
MonoSDF via eval.py --world_space (scale_mat), PGSR reads COLMAP directly.

Usage:
    python fuse_meshes.py --geom monosdf_world.ply --color pgsr_post.ply \
        --out unified_room.ply --tris 350000 --max-dist 0.25
"""
from __future__ import annotations
import argparse, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help="watertight geometry source (MonoSDF, world space)")
    ap.add_argument("--color", required=True, help="colour source (PGSR, has vertex colors)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tris", type=int, default=350000)
    ap.add_argument("--max-dist", type=float, default=0.25,
                    help="COLMAP-unit radius beyond which a geom vertex is treated as 'unobserved'")
    ap.add_argument("--neutral", type=float, nargs=3, default=[0.62, 0.60, 0.57],
                    help="wall colour to blend toward in unobserved gaps")
    a = ap.parse_args()

    import open3d as o3d
    from scipy.spatial import cKDTree

    t0 = time.time()
    geom = o3d.io.read_triangle_mesh(a.geom)
    col = o3d.io.read_triangle_mesh(a.color)
    if not col.has_vertex_colors():
        raise SystemExit("colour source has no vertex colors")
    Vg = np.asarray(geom.vertices)
    Vp = np.asarray(col.vertices)
    Cp = np.asarray(col.vertex_colors)
    print(f"geom verts={len(Vg)}  color verts={len(Vp)}  ({time.time()-t0:.1f}s load)", flush=True)

    # nearest PGSR vertex for every MonoSDF vertex
    t1 = time.time()
    tree = cKDTree(Vp)
    dist, idx = tree.query(Vg, k=1, workers=-1)
    print(f"kdtree query {time.time()-t1:.1f}s. dist: "
          f"median={np.median(dist):.3f} p90={np.percentile(dist,90):.3f} max={dist.max():.3f}",
          flush=True)

    colors = Cp[idx].copy()
    # blend unobserved vertices (no nearby PGSR surface) toward the neutral wall
    far = dist > a.max_dist
    neutral = np.array(a.neutral, dtype=float)
    # smooth ramp: fully neutral by 3x max-dist
    w = np.clip((dist - a.max_dist) / (2.0 * a.max_dist), 0.0, 1.0)[:, None]
    colors = colors * (1 - w) + neutral[None, :] * w
    print(f"observed (textured) verts: {(~far).sum()} ({100*(~far).sum()/len(Vg):.1f}%)  "
          f"unobserved (filled): {far.sum()} ({100*far.sum()/len(Vg):.1f}%)", flush=True)

    geom.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    geom.compute_vertex_normals()

    # largest connected component (drop any stray bits), then decimate (colors preserved)
    labels, ccs, areas = geom.cluster_connected_triangles()
    labels = np.asarray(labels); areas = np.asarray(areas)
    keep = labels == int(np.argmax(areas))
    geom.triangles = o3d.utility.Vector3iVector(np.asarray(geom.triangles)[keep])
    geom.remove_unreferenced_vertices()
    print(f"LCC kept {keep.sum()}/{len(labels)} tris ({len(ccs)} components)", flush=True)

    if len(geom.triangles) > a.tris:
        t2 = time.time()
        geom = geom.simplify_quadric_decimation(target_number_of_triangles=a.tris)
        print(f"decimated -> verts={len(geom.vertices)} tris={len(geom.triangles)} "
              f"({time.time()-t2:.1f}s) colors={geom.has_vertex_colors()}", flush=True)

    geom.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, geom, write_ascii=False, compressed=True)
    bb = geom.get_axis_aligned_bounding_box()
    print(f"wrote {a.out}  bbox extent={np.round(bb.get_extent(),2)}", flush=True)
    print("FUSE_DONE", flush=True)


if __name__ == "__main__":
    main()
