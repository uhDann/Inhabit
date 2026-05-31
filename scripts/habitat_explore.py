"""Load a reconstructed room mesh into Habitat-Sim, build a navmesh, and drive a
navigation agent (greedy geodesic follower) between random reachable goals,
rendering first-person RGB to an MP4. The "phone scan -> explorable RL env" demo.

Run in the `habitat` env (habitat-sim 0.3.3, headless):
    python scripts/habitat_explore.py --glb room_mesh.glb --out explore.mp4
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")


def make_cfg(glb, width, height, eye, gpu=0):
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = gpu
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, eye, 0.0]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--goals", type=int, default=6)
    ap.add_argument("--max-steps-per-goal", type=int, default=400)
    ap.add_argument("--agent-radius", type=float, default=0.1)
    ap.add_argument("--agent-height", type=float, default=0.7)
    ap.add_argument("--agent-max-climb", type=float, default=0.3)
    ap.add_argument("--eye", type=float, default=0.5)
    ap.add_argument("--goal-radius", type=float, default=0.4)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import habitat_sim
    import imageio.v2 as imageio

    sim = habitat_sim.Simulator(make_cfg(args.glb, args.width, args.height, args.eye))

    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = args.agent_radius
    ns.agent_height = args.agent_height
    ns.agent_max_climb = args.agent_max_climb
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    pf = sim.pathfinder
    lo, hi = pf.get_bounds()
    print(f"navmesh_ok={ok} loaded={pf.is_loaded} navigable_area={pf.navigable_area:.3f}", flush=True)
    print(f"scene bounds lo={np.round(lo,2)} hi={np.round(hi,2)}", flush=True)
    if not pf.is_loaded or pf.navigable_area <= 0:
        print("NO_NAVMESH -> mesh likely not Y-up or agent params off; aborting", flush=True)
        return

    follower = sim.make_greedy_follower(0, goal_radius=args.goal_radius)
    agent = sim.get_agent(0)
    st = agent.get_state()
    st.position = pf.get_random_navigable_point()
    agent.set_state(st)

    frames = []
    reached = 0
    for g in range(args.goals):
        goal = pf.get_random_navigable_point()
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
            obs = sim.step(action)
            frames.append(np.asarray(obs["rgb"])[:, :, :3].copy())
    print(f"goals_reached={reached}/{args.goals}  frames={len(frames)}", flush=True)

    if frames:
        imageio.mimwrite(args.out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"wrote {args.out} ({len(frames)} frames)", flush=True)
    sim.close()


if __name__ == "__main__":
    main()
