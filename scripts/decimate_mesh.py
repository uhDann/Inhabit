"""Decimate a mesh (Recast can fail on very dense meshes)."""
import argparse
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--target", type=int, default=60000)
a = ap.parse_args()

m = o3d.io.read_triangle_mesh(a.inp)
m = m.simplify_quadric_decimation(a.target)
m.remove_unreferenced_vertices()
m.remove_degenerate_triangles()
m.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, m)
o3d.io.write_triangle_mesh(a.out.rsplit(".", 1)[0] + ".obj", m)
print(f"decimated -> verts={len(m.vertices)} tris={len(m.triangles)} -> {a.out}", flush=True)
