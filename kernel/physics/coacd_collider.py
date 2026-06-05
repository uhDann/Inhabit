"""CoACD convex decomposition of a (concave) room-shell mesh into convex parts.

Genesis builds an SDF for a fixed mesh collider, which OOMs on a full room mesh (450^3
grid cap). Decomposing into convex parts gives cheap convex colliders (no SDF) -> no OOM.

Run (genesis env, has coacd): python physics/coacd_collider.py shell.ply out_dir/
"""
import sys, os
import trimesh, coacd


def decompose(mesh_path, out_dir, target_faces=80000, threshold=0.06, max_hulls=40):
    os.makedirs(out_dir, exist_ok=True)
    m = trimesh.load(mesh_path, process=False)
    if len(m.faces) > target_faces:
        m = m.simplify_quadric_decimation(face_count=target_faces)
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold, max_convex_hull=max_hulls)
    for i, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f, process=False).export(f"{out_dir}/part_{i:02d}.obj")
    print(f"coacd: {len(parts)} convex parts -> {out_dir}")
    return len(parts)


if __name__ == "__main__":
    decompose(sys.argv[1], sys.argv[2])
