"""CPU-only mesh decimation for browser-loadable .ply files.

Reads each input, runs Open3D quadric decimation to the target triangle count,
preserves vertex colors when present, and writes a compact binary .ply. Designed
to be safe to run while a GPU training job is active (no CUDA / no GPU mem).

Usage:
    python decimate_meshes.py \
        --in /path/room_final.ply       --out /path/dn_decim.ply  --tris 300000 \
        --in /path/monosdf_mesh.ply     --out /path/mono_decim.ply --tris 200000
"""
from __future__ import annotations
import argparse, os, time


def decimate(in_path: str, out_path: str, target_tris: int) -> None:
    import open3d as o3d
    t0 = time.time()
    m = o3d.io.read_triangle_mesh(in_path)
    n_v, n_t = len(m.vertices), len(m.triangles)
    has_col = m.has_vertex_colors()
    print(f"[{in_path}] verts={n_v} tris={n_t} colors={has_col}", flush=True)
    if n_t <= target_tris:
        print("  already at/under target; skipping decimation", flush=True)
    else:
        # Open3D's quadric decimation preserves vertex colors / normals when present.
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
        print(f"  decimated -> tris={len(m.triangles)} verts={len(m.vertices)} "
              f"({time.time()-t0:.1f}s)", flush=True)
    # Recompute normals so the browser viewer gets smooth shading
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, m, write_ascii=False, compressed=True)
    sz = os.path.getsize(out_path)
    print(f"  wrote {out_path} ({sz/1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", action="append", required=True)
    ap.add_argument("--out", dest="outs", action="append", required=True)
    ap.add_argument("--tris", dest="trises", action="append", type=int, required=True)
    a = ap.parse_args()
    assert len(a.ins) == len(a.outs) == len(a.trises), "in/out/tris must be paired"
    for ip, op, tt in zip(a.ins, a.outs, a.trises):
        decimate(ip, op, tt)
    print("DECIMATE_DONE", flush=True)


if __name__ == "__main__":
    main()
