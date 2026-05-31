"""Take a raw MonoSDF surface_*.ply (in the unit-sphere training frame) and
turn it into a viewer-ready watertight COLMAP-world mesh:

  1. transform vertices by the converter's `scale_mat` so the mesh sits in
     COLMAP world coordinates (where DN-Splatter and the cameras live);
  2. keep the largest connected component (drops floaters);
  3. quadric-decimate to a target triangle count for the browser;
  4. recompute vertex normals + write a compact binary .ply.

Usage:
    python process_monosdf_tight.py --in surface_100.ply --cameras cameras.npz \
        --out monosdf_decim.ply --tris 200000
"""
from __future__ import annotations
import argparse, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ply_in", required=True)
    ap.add_argument("--cameras", required=True, help="MonoSDF cameras.npz with scale_mat_0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tris", type=int, default=200000)
    a = ap.parse_args()

    import open3d as o3d

    # ---- scale_mat from the converter (training frame -> COLMAP world) ----
    cams = np.load(a.cameras)
    sm_keys = sorted(k for k in cams.files if k.startswith("scale_mat"))
    if not sm_keys:
        raise SystemExit("no scale_mat_* in cameras.npz")
    S = np.asarray(cams[sm_keys[0]], dtype=np.float64)
    print(f"scale_mat[0] diag={[round(float(S[i,i]),3) for i in range(3)]} "
          f"center={[round(float(S[i,3]),3) for i in range(3)]}", flush=True)

    # ---- load ----
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(a.ply_in)
    n_v0, n_t0 = len(m.vertices), len(m.triangles)
    print(f"load {a.ply_in}: verts={n_v0} tris={n_t0} ({time.time()-t0:.1f}s)", flush=True)

    # ---- transform to COLMAP world ----
    V = np.asarray(m.vertices, dtype=np.float64)
    V_world = V @ S[:3, :3].T + S[:3, 3]
    m.vertices = o3d.utility.Vector3dVector(V_world)
    print("applied scale_mat -> COLMAP world", flush=True)

    # ---- largest connected component (drops floaters) ----
    labels, ccs, areas = m.cluster_connected_triangles()
    labels = np.asarray(labels); areas = np.asarray(areas)
    keep = int(np.argmax(areas))
    keep_mask = labels == keep
    print(f"connected components: {len(ccs)}; keep cc#{keep} with {keep_mask.sum()} tris "
          f"({100.0 * keep_mask.sum() / len(labels):.1f}% of total)", flush=True)
    tri_arr = np.asarray(m.triangles)
    m.triangles = o3d.utility.Vector3iVector(tri_arr[keep_mask])
    m.remove_unreferenced_vertices()
    print(f"after LCC: verts={len(m.vertices)} tris={len(m.triangles)}", flush=True)

    # ---- decimate ----
    if len(m.triangles) > a.tris:
        t1 = time.time()
        m = m.simplify_quadric_decimation(target_number_of_triangles=a.tris)
        print(f"decimated -> verts={len(m.vertices)} tris={len(m.triangles)} "
              f"({time.time()-t1:.1f}s)", flush=True)

    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(a.out, m, write_ascii=False, compressed=True)
    print(f"wrote {a.out}", flush=True)
    print("PROCESS_DONE", flush=True)


if __name__ == "__main__":
    main()
