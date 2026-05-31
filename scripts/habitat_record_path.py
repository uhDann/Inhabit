"""Habitat navigation that RECORDS the agent's camera path (no splat here).
Loads the aligned room mesh, builds a navmesh, drives a greedy follower between
random goals, and saves the per-step camera->world transforms (Habitat OpenGL
frame) + intrinsics to a JSON. A separate splat-render step turns these into a
photorealistic first-person video.

Run in the `habitat` env:
    python scripts/habitat_record_path.py --glb room_aligned.glb --out path.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


def make_cfg(glb, width, height, hfov, eye, fwd, turn, gpu=0):
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color_sensor"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, eye, 0.0]
    rgb.hfov = hfov
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    A, S = habitat_sim.agent.ActionSpec, habitat_sim.agent.ActuationSpec
    agent_cfg.action_space = {
        "move_forward": A("move_forward", S(amount=fwd)),     # smaller -> slower, smoother
        "turn_left": A("turn_left", S(amount=turn)),
        "turn_right": A("turn_right", S(amount=turn)),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--hfov", type=float, default=70.0)
    ap.add_argument("--eye", type=float, default=0.4)
    ap.add_argument("--goals", type=int, default=8)
    ap.add_argument("--max-steps-per-goal", type=int, default=300)
    ap.add_argument("--agent-radius", type=float, default=0.08)
    ap.add_argument("--agent-height", type=float, default=0.6)
    ap.add_argument("--agent-max-climb", type=float, default=0.4)
    ap.add_argument("--goal-radius", type=float, default=0.3)
    ap.add_argument("--cell-size", type=float, default=0.0, help="navmesh cell size (0=default)")
    ap.add_argument("--forward-step", type=float, default=0.08, help="meters per forward step (smaller=smoother)")
    ap.add_argument("--turn-angle", type=float, default=4.0, help="degrees per turn (smaller=smoother)")
    ap.add_argument("--central-frac", type=float, default=0.5, help="keep agent within this frac of room half-extent (on-manifold)")
    args = ap.parse_args()

    import habitat_sim
    sim = habitat_sim.Simulator(make_cfg(args.glb, args.width, args.height, args.hfov,
                                         args.eye, args.forward_step, args.turn_angle))
    print("sim created", flush=True)

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = args.agent_radius
    ns.agent_height = args.agent_height
    ns.agent_max_climb = args.agent_max_climb
    if args.cell_size > 0:
        ns.cell_size = args.cell_size
    print("recomputing navmesh...", flush=True)
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    print("recompute returned", ok, flush=True)
    pf = sim.pathfinder
    area = 0.0
    try:
        if pf.is_loaded:
            area = pf.navigable_area
    except Exception as e:
        print("area-access error", repr(e)[:80], flush=True)
    print(f"navmesh_ok={ok} navigable_area={area:.3f}", flush=True)
    if not pf.is_loaded or area <= 0:
        print("NO_NAVMESH", flush=True); return

    lo, hi = pf.get_bounds()
    cxz = (0.5 * (lo[0] + hi[0]), 0.5 * (lo[2] + hi[2]))
    max_r = args.central_frac * 0.5 * min(hi[0] - lo[0], hi[2] - lo[2])

    def central_point():
        for _ in range(300):
            p = pf.get_random_navigable_point()
            if (p[0] - cxz[0]) ** 2 + (p[2] - cxz[1]) ** 2 < max_r ** 2:
                return p
        return pf.get_random_navigable_point()

    follower = sim.make_greedy_follower(0, goal_radius=args.goal_radius)
    agent = sim.get_agent(0)
    st = agent.get_state(); st.position = central_point(); agent.set_state(st)

    def cam_to_world():
        sensor = sim.get_agent(0)._sensors["color_sensor"]
        return np.array(sensor.node.absolute_transformation()).tolist()

    poses = []
    reached = 0
    for g in range(args.goals):
        goal = central_point()
        try:
            follower.reset()
        except Exception:
            pass
        for _ in range(args.max_steps_per_goal):
            try:
                action = follower.next_action_along(goal)
            except Exception:
                break
            if action is None:
                reached += 1
                break
            sim.step(action)
            poses.append(cam_to_world())
    print(f"goals_reached={reached}/{args.goals} poses={len(poses)}", flush=True)
    if poses:
        print(f"first cam pos {np.round(np.array(poses[0])[:3,3],3)}", flush=True)
    json.dump({"poses": poses, "hfov": args.hfov, "width": args.width, "height": args.height}, open(args.out, "w"))
    print(f"wrote {args.out}", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
