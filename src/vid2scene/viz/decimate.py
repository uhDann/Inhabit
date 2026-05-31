"""Quadric-decimate a mesh to a browser-friendly triangle budget, preserving
vertex colours. CPU-only (Open3D).

CLI:
    python -m vid2scene viz --in mesh.ply --out mesh_web.ply --tris 300000
"""
from __future__ import annotations

import os


def decimate(in_ply: str, out_ply: str, target_tris: int = 300_000,
             largest_component: bool = True) -> dict:
    import open3d as o3d
    import numpy as np

    m = o3d.io.read_triangle_mesh(in_ply)
    n0 = len(m.triangles)
    if largest_component and n0 > 0:
        labels, _, areas = m.cluster_connected_triangles()
        labels, areas = np.asarray(labels), np.asarray(areas)
        keep = labels == int(np.argmax(areas))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(m.triangles)[keep])
        m.remove_unreferenced_vertices()
    if target_tris and len(m.triangles) > target_tris:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_tris)
    m.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, m, write_ascii=False, compressed=True)
    return {"in_tris": int(n0), "out_tris": int(len(m.triangles)),
            "colors": bool(m.has_vertex_colors()),
            "size_mb": round(os.path.getsize(out_ply) / 1e6, 1), "out_ply": out_ply}
