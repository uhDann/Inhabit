"""Export the kernel's reconstruction as simulator-ready assets:
a static room-shell collider + one watertight collision body per movable object.
Run in the splat env (needs skimage/trimesh/open3d). Writes OBJs + assets.json.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import trimesh

import scene as S
from kernel import InhabitKernel

OUT = "/tmp/ikernel/assets"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    W, H, ncam, voxel = 256, 192, 48, 0.013
    sc = S.Scene(device=dev)
    K = S.intrinsics(W, H)
    poses = S.ring_poses(sc, n=ncam)

    depths, objids = [], []
    for p in poses:
        d, o = sc.render(torch.from_numpy(p).to(dev), K, W, H)
        depths.append(d.cpu()); objids.append(o.cpu())

    pad = 0.15
    ker = InhabitKernel([-pad, -pad, -pad], [4.0 + pad, 2.6 + pad, 5.0 + pad],
                        voxel=voxel, trunc_vox=3.0, device=dev, robust=True, n_labels=4)
    for d, p, o in zip(depths, poses, objids):
        ker.integrate(d, torch.from_numpy(p), K, labels=o)
    verts, faces, vlab = ker.extract_mesh()

    lf = vlab[faces]
    a, b, c = lf[:, 0], lf[:, 1], lf[:, 2]
    face_lab = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, a)))

    # gravity-align: scene is y-down (floor at y=room_max[1]). Map to Genesis z-up
    # with the floor at z=0:  (x, y, z)_scene -> (x, z, floor_y - y).
    floor_y = float(sc.room_max[1].item())

    def to_zup(V):
        V = np.asarray(V, np.float64)
        return np.stack([V[:, 0], V[:, 2], floor_y - V[:, 1]], 1)

    assets = {"room": None, "objects": [], "floor_z": 0.0}
    names = {0: "room_shell", 1: "sphere", 2: "inner_box"}
    for oid, name in names.items():
        sub_f = faces[face_lab == oid]
        if len(sub_f) == 0:
            continue
        used = np.unique(sub_f)
        remap = {int(o): i for i, o in enumerate(used)}
        nf = np.vectorize(remap.get)(sub_f)
        sm = trimesh.Trimesh(vertices=to_zup(verts[used]), faces=nf, process=False)
        if oid == 0:
            # decimate the room shell to a light collision mesh (Genesis builds an SDF
            # collider per mesh; the full ~1.8M-vert shell OOMs, ~25k tris is fine)
            import open3d as o3d
            om = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(sm.vertices.astype(np.float64)),
                o3d.utility.Vector3iVector(sm.faces.astype(np.int32)))
            om = om.simplify_quadric_decimation(25000)
            om.remove_degenerate_triangles(); om.remove_unreferenced_vertices()
            path = f"{OUT}/room_shell.obj"
            o3d.io.write_triangle_mesh(path, om)
            assets["room"] = path; assets["room_tris"] = len(om.triangles)
            # CoACD convex decomposition -> cheap convex colliders (no big mesh SDF,
            # which OOMs Genesis on a large room bbox). This is the physics-ready room.
            import coacd
            rv = np.asarray(om.vertices); rf = np.asarray(om.triangles)
            parts = coacd.run_coacd(coacd.Mesh(rv, rf), threshold=0.06, max_convex_hull=24)
            ppaths = []
            for i, (pv, pf) in enumerate(parts):
                pp = f"{OUT}/room_part_{i:02d}.obj"
                trimesh.Trimesh(pv, pf, process=False).export(pp); ppaths.append(pp)
            assets["room_parts"] = ppaths
        else:
            hull = sm.convex_hull                     # watertight collision proxy
            ctr = hull.vertices.mean(0)
            hull.vertices = hull.vertices - ctr       # recentre so body origin = centroid
            path = f"{OUT}/obj_{name}.obj"
            hull.export(path)
            lo = hull.vertices.min(0); hi = hull.vertices.max(0)
            assets["objects"].append(
                {"name": name, "path": path, "centroid": ctr.tolist(),
                 "extent": (hi - lo).tolist(),
                 "bottom_offset": float(-lo[2])})       # centroid height above lowest point
    rmin = to_zup(sc.room_min.cpu().numpy()[None])[0].tolist()
    rmax = to_zup(sc.room_max.cpu().numpy()[None])[0].tolist()
    assets["room_xy"] = [[min(rmin[0], rmax[0]), min(rmin[1], rmax[1])],
                         [max(rmin[0], rmax[0]), max(rmin[1], rmax[1])]]
    json.dump(assets, open(f"{OUT}/assets.json", "w"), indent=2)
    print("exported:", json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
