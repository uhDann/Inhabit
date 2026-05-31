"""Embodied stage — turn the reconstructed room into a robot-explorable world.

This is vid2scene's headline differentiator: we don't stop at a pretty mesh, we
drop a robot into the *metric* reconstruction and let it act. Two backends:

- `habitat` (implemented): the reconstructed mesh -> Recast navmesh -> a greedy
  navigation agent whose first-person view is re-rendered photorealistically
  through the Gaussian splat. Drives the `explore_photoreal.mp4` demo.
  (Driver scripts: ../../scripts/habitat_record_path.py, gsplat_render_path.py.)

- `genesis` (proposed, ../../docs/PHASE2_GENESIS.md): the Genesis-Embodied-AI
  stack — genesis-world physics + genesis-nyx photoreal 3DGS camera sensor +
  quadrants compiler. A real robot body (Go2 quadruped) with contact physics
  and in-sim camera observations rendered live from our splat, trained with RL.
  A strict upgrade over the Habitat + offline-render pipeline.

`genesis_world.export_for_genesis` prepares the metric mesh (gravity-aligned,
Y-up GLB) + splat for ingestion; see PHASE2_GENESIS.md for the migration plan.
"""

from .genesis_world import export_for_genesis

__all__ = ["export_for_genesis"]
