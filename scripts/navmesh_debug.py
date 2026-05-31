"""Diagnose why Habitat Recast fails to build a navmesh: print the scene bounding
box Habitat actually sees (catches import re-orientation/scale), then sweep
NavMeshSettings to find a combination that builds."""

import argparse
import numpy as np
import habitat_sim


def cfg(glb):
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = glb
    sc.enable_physics = False
    return habitat_sim.Configuration(sc, [habitat_sim.agent.AgentConfiguration()])


def try_nav(sim, **kw):
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    for k, v in kw.items():
        setattr(ns, k, v)
    try:
        ok = sim.recompute_navmesh(sim.pathfinder, ns)
        return f"ok={ok} area={sim.pathfinder.navigable_area:.3f} loaded={sim.pathfinder.is_loaded}"
    except Exception as e:
        return f"EXC {repr(e)[:160]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glbs", nargs="+", required=True)
    args = ap.parse_args()
    sweeps = [
        dict(),
        dict(cell_size=0.1),
        dict(cell_size=0.2),
        dict(cell_size=0.05, cell_height=0.05),
        dict(agent_radius=0.1, agent_height=0.7, agent_max_climb=0.3),
        dict(cell_size=0.15, agent_radius=0.1, agent_height=0.7, agent_max_climb=0.3, cell_height=0.1),
        dict(cell_size=0.1, agent_radius=0.05, agent_height=0.4, agent_max_climb=0.5),
    ]
    for glb in args.glbs:
        print(f"\n===== {glb} =====", flush=True)
        sim = habitat_sim.Simulator(cfg(glb))
        try:
            bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
            print(f"habitat scene bb: min={list(bb.min)} max={list(bb.max)}", flush=True)
        except Exception as e:
            print(f"bb error {repr(e)[:120]}", flush=True)
        for kw in sweeps:
            print(f"  {kw} -> {try_nav(sim, **kw)}", flush=True)
        sim.close()


if __name__ == "__main__":
    main()
