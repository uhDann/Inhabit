"""Prepare a reconstructed room for an embodied simulator.

Genesis (and Habitat) want a gravity-aligned, Y-up, metric collider mesh. Our
meshes come out in the reconstruction's world frame (COLMAP / metric). This
applies the source->simulator transform and exports a GLB the simulator can
load as a static collider, alongside the splat path used for photoreal camera
rendering.

The heavy simulator integration (physics step, robot URDF, RL) is documented in
../../docs/PHASE2_GENESIS.md and driven by ../../scripts/ (Habitat backend is
implemented end-to-end today; Genesis is the proposed upgrade). This function is
the geometry hand-off both backends share.

CLI:
    python -m vid2scene embodied --mesh room.ply --out room_sim.glb --up y
"""
from __future__ import annotations

import json
import numpy as np

# Habitat/Genesis import GLBs with a frame rotation; this Rx(+90) cancels it so
# the room stands upright (matches scripts/export_habitat_glb.py).
RX_PLUS_90 = np.array([[1, 0, 0],
                       [0, 0, -1],
                       [0, 1, 0]], dtype=np.float64)


def export_for_genesis(mesh_ply: str, out_glb: str, splat_ply: str | None = None,
                       up: str = "y", scene_json: str | None = None) -> dict:
    """Gravity-align + export the reconstruction as a sim-ready GLB collider.

    `up` selects the simulator's up-axis convention ('y' for Habitat/Genesis-GL,
    'z' for MuJoCo-style). Writes `out_glb` and, if `scene_json` is given, a
    manifest recording the transform + the splat to render the robot's camera
    from (so geometry and appearance stay in the same frame).
    """
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(mesh_ply)
    m.compute_vertex_normals()
    R = RX_PLUS_90 if up == "y" else np.eye(3)
    m.rotate(R, center=(0, 0, 0))
    o3d.io.write_triangle_mesh(out_glb, m)

    manifest = {
        "mesh_glb": out_glb,
        "splat_ply": splat_ply,
        "up": up,
        "source_to_sim_R": R.tolist(),
        "note": "static collider for Habitat/Genesis; render robot camera from splat_ply. "
                "See docs/PHASE2_GENESIS.md for the physics + RL integration.",
    }
    if scene_json:
        json.dump(manifest, open(scene_json, "w"), indent=2)
    return manifest
