"""Export a GLB pre-rotated by +90deg about X, to cancel Habitat's GLB-import
rotation (Open3D GLBs come in such that Habitat applies Rx(-90)). After this, the
mesh appears Y-up inside Habitat (floor horizontal -> navmesh builds), and the
agent ends up operating in our aligned frame (so M_s2h mapping is unchanged)."""
import argparse
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)   # aligned OBJ (Y-up)
ap.add_argument("--out", required=True)              # GLB for Habitat
a = ap.parse_args()

m = o3d.io.read_triangle_mesh(a.inp)
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # +90 about X: (x,y,z)->(x,-z,y)
m.rotate(Rx, center=(0, 0, 0))
m.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out, m)
print(f"wrote {a.out}", flush=True)
