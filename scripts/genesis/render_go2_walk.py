#!/usr/bin/env python3
"""
Fixed-camera clip of the Go2 walking a path across the reconstructed room.

The camera is stationary; the robot moves. Locomotion here is kinematic: the base
is driven along a planned path over the open floor and the legs cycle a trot gait
so the motion reads as walking. This is a visualization of an agent traversing the
scanned environment, not a physics-trained controller (that needs a learned policy,
see docs/PHASE2_GENESIS.md). The point it makes is that the metric mesh is a usable
embodied environment at real scale.

Because the traversal is kinematic (no collision response), the path must lie in
free floor or the robot would clip furniture. The default --wp below is a circle
in the most open part of the room0 floor: its centre and radius come from a
clearance map (distance transform of the floor footprint with furniture in a
0.05-0.9 m height band masked out); the chosen radius keeps the robot's footprint
clear of every obstacle. Re-derive it for a new scene before changing the path.

GPU rasterizer (EGL), light VRAM.
"""
import argparse
import os
import math
import numpy as np
import genesis as gs

GO2_URDF = ("/cs/student/projects3/2023/dkozlov/conda-envs/genesis/lib/python3.10/"
            "site-packages/genesis/assets/urdf/go2/urdf/go2.urdf")
# 12 leg dofs: [hip x4, thigh x4, calf x4] in (FL, FR, RL, RR) order.
NOMINAL = np.array([0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)
# Diagonal trot: FL+RR swing together, FR+RL a half-cycle later.
LEG_PHASE = np.array([0.0, math.pi, math.pi, 0.0])     # FL, FR, RL, RR


def to_np(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.asarray(x)


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], dtype=np.float32)


def gait_pose(t, stride_thigh=0.30, stride_calf=0.35, freq=2.0):
    """Trot leg angles at phase time t (seconds)."""
    q = NOMINAL.copy()
    for leg in range(4):
        ph = 2 * math.pi * freq * t + LEG_PHASE[leg]
        # thigh swings fore/aft; calf lifts the foot during the swing half.
        q[4 + leg] = NOMINAL[4 + leg] + stride_thigh * math.sin(ph)
        lift = max(0.0, math.sin(ph))            # only lift in swing phase
        q[8 + leg] = NOMINAL[8 + leg] + stride_calf * lift
    return q


def catmull(path_pts, n):
    """Smooth a few waypoints into n positions (centripetal-ish via numpy interp)."""
    path_pts = np.asarray(path_pts, dtype=np.float64)
    seg = np.linspace(0, 1, len(path_pts))
    tt = np.linspace(0, 1, n)
    return np.stack([np.interp(tt, seg, path_pts[:, k]) for k in range(path_pts.shape[1])], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned_decim.obj")
    ap.add_argument("--outdir", default="/cs/student/projects3/2023/dkozlov/genesis-work/walk")
    ap.add_argument("--res_w", type=int, default=960)
    ap.add_argument("--res_h", type=int, default=720)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--base_z", type=float, default=0.30)   # standing base height
    ap.add_argument("--fov", type=float, default=46)
    # fixed camera (3/4 elevated view of the open floor)
    ap.add_argument("--cam", type=float, nargs=3, default=[-2.6, -2.6, 1.7])
    ap.add_argument("--look", type=float, nargs=3, default=[0.2, 0.2, 0.30])
    # path waypoints across the open floor (x, y); clearance-verified circle by
    # default (centre 0.58, 1.40; r 0.5) in the most open part of the room0 floor.
    ap.add_argument("--wp", type=float, nargs="+",
                    default=[1.080, 1.400, 1.042, 1.591, 0.934, 1.754, 0.771, 1.862,
                             0.580, 1.900, 0.389, 1.862, 0.226, 1.754, 0.118, 1.591,
                             0.080, 1.400, 0.118, 1.209, 0.226, 1.046, 0.389, 0.938,
                             0.580, 0.900, 0.771, 0.938, 0.934, 1.046, 1.042, 1.209,
                             1.080, 1.400])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gs.init(backend=gs.gpu)
    sc = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, 0)),   # kinematic
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    sc.add_entity(gs.morphs.Mesh(file=args.mesh, fixed=True, collision=False, convexify=False))
    wp = np.asarray(args.wp, dtype=np.float64).reshape(-1, 2)
    robot = sc.add_entity(gs.morphs.URDF(file=GO2_URDF, pos=(wp[0, 0], wp[0, 1], args.base_z)))

    try:
        sc.add_light(pos=(0.6, 0.6, 2.6), dir=(0, 0, -1), color=(1.0, 0.98, 0.95),
                     intensity=7.0, directional=False, castshadow=True, cutoff=75.0)
    except Exception as e:
        print(f"[walk] add_light skipped: {e}")

    cam = sc.add_camera(res=(args.res_w, args.res_h), pos=tuple(args.cam),
                        lookat=tuple(args.look), fov=args.fov, GUI=False)
    sc.build()

    leg = list(range(robot.n_dofs - 12, robot.n_dofs))
    robot.set_dofs_position(NOMINAL, leg)

    pos_path = catmull(wp, args.frames)
    from PIL import Image
    frames = []
    dt = 0.02                                    # gait clock per frame
    for i in range(args.frames):
        p = pos_path[i]
        # heading = direction of travel (look a few frames ahead)
        j = min(i + 3, args.frames - 1)
        d = pos_path[j] - p
        yaw = math.atan2(d[1], d[0]) if np.linalg.norm(d) > 1e-6 else 0.0
        robot.set_pos(np.array([p[0], p[1], args.base_z], dtype=np.float32))
        robot.set_quat(yaw_quat(yaw))
        robot.set_dofs_position(gait_pose(i * dt), leg)
        sc.step()                                # refresh forward kinematics / render state
        out = cam.render()
        rgb = out[0] if isinstance(out, (tuple, list)) else out
        frames.append(to_np(rgb).astype("uint8"))
    print(f"[walk] captured {len(frames)} frames")

    mp4 = os.path.join(args.outdir, "go2_walk.mp4")
    try:
        import imageio.v2 as imageio
        imageio.mimsave(mp4, frames, fps=30, quality=8, macro_block_size=8)
        print(f"[walk] wrote {mp4}")
    except Exception as e:
        print(f"[walk] mp4 failed ({e}); dumping PNGs")
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(args.outdir, f"frame_{i:03d}.png"))


if __name__ == "__main__":
    main()
