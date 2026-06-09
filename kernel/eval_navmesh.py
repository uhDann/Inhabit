"""Task-transfer evaluation: is the reconstructed twin's NAVIGABILITY faithful to reality?

Measures a reconstruction by ROBOT TASK UTILITY (can an agent navigate it the same way it
would navigate the real room) rather than image metrics -- the axis the field admits it does
not benchmark. Recompute a robot navmesh (Habitat/Recast) on the GT scene mesh and on our
reconstructed twin, then count how many GT shortest paths reproduce in the twin and compare
their lengths.

office0 result (twin = PGSR mesh): 81/100 GT paths reproducible, 0% median path-length error,
navigable-area ratio 1.58 (the twin is slightly too open -- thin/missing furniture obstacles,
which is why it is 81% not 100%).

Two steps (two conda envs): rotate Z-up->Y-up (any env with trimesh), then run the navmesh
comparison in the habitat env.

Run (habitat env): python eval_navmesh.py gt_yup.ply twin_yup.ply
"""
import sys
import numpy as np
import habitat_sim


def make_sim(mesh, agent_radius=0.2, agent_height=1.0):
    cfg = habitat_sim.SimulatorConfiguration(); cfg.scene_id = mesh; cfg.create_renderer = False
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [habitat_sim.agent.AgentConfiguration()]))
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = agent_radius; ns.agent_height = agent_height
    sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder; area = getattr(pf, "navigable_area", -1.0)
    return sim, float(area), pf.is_loaded


def transfer(gt_mesh, twin_mesh, n=100, seed=0):
    sim_gt, ag, _ = make_sim(gt_mesh); sim_tw, at, _ = make_sim(twin_mesh)
    print(f"GT navmesh area {ag:.2f}  | twin {at:.2f}  (ratio {at/max(ag,1e-6):.2f})", flush=True)
    np.random.seed(seed); succ = 0; lerr = []
    for _ in range(n):
        a = sim_gt.pathfinder.get_random_navigable_point(); b = sim_gt.pathfinder.get_random_navigable_point()
        sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
        if not sim_gt.pathfinder.find_path(sp) or not np.isfinite(sp.geodesic_distance) or sp.geodesic_distance < 0.5:
            continue
        gL = sp.geodesic_distance
        sp2 = habitat_sim.ShortestPath(); sp2.requested_start = sim_tw.pathfinder.snap_point(a); sp2.requested_end = sim_tw.pathfinder.snap_point(b)
        if sim_tw.pathfinder.find_path(sp2) and np.isfinite(sp2.geodesic_distance):
            succ += 1; lerr.append(abs(sp2.geodesic_distance - gL) / gL)
    print(f"TASK-TRANSFER: {succ}/{n} GT paths reproducible in twin | median path-length error {np.median(lerr)*100:.0f}%", flush=True)
    return succ, n, float(np.median(lerr)) if lerr else None


if __name__ == "__main__":
    transfer(sys.argv[1], sys.argv[2])
