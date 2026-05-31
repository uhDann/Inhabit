#!/usr/bin/env python3
"""
Clean hero render + turntable of the Go2 standing in the reconstructed room.

Stands the Go2 (kp=30, kv=1.5, symmetric pose) at a clean flat floor patch, then:
  - saves a single well-framed 3/4-angle hero PNG (960x720), and
  - orbits the camera around the standing robot for N frames and encodes an MP4.

GPU rasterizer (EGL), light VRAM. Run after the env is set up.
"""
import argparse
import os
import math
import numpy as np
import genesis as gs

GO2_URDF = ("/cs/student/projects3/2023/dkozlov/conda-envs/genesis/lib/python3.10/"
            "site-packages/genesis/assets/urdf/go2/urdf/go2.urdf")
NOMINAL = np.array([0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8, -1.5, -1.5, -1.5, -1.5], dtype=np.float32)


def to_np(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned_decim.obj")
    ap.add_argument("--outdir", default="/cs/student/projects3/2023/dkozlov/genesis-work/hero")
    ap.add_argument("--x", type=float, default=5.0)
    ap.add_argument("--y", type=float, default=0.5)
    ap.add_argument("--res_w", type=int, default=960)
    ap.add_argument("--res_h", type=int, default=720)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--radius", type=float, default=2.0)   # camera distance from robot
    ap.add_argument("--eye_h", type=float, default=1.0)    # camera eye height
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gs.init(backend=gs.gpu)
    sc = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )
    sc.add_entity(gs.morphs.Mesh(file=args.mesh, fixed=True, collision=True, convexify=False))
    robot = sc.add_entity(gs.morphs.URDF(file=GO2_URDF, pos=(args.x, args.y, 0.42)))

    # Aim the look-at at the robot body (~0.25 m up). The recognizable furniture
    # (sofa/chairs/window) is toward lower x / higher y, so the default 3/4 hero
    # azimuth looks back across the robot toward that side of the room.
    look = (args.x, args.y, 0.27)
    az0 = math.radians(135)  # 3/4 view, camera behind-right looking toward furniture
    cam = sc.add_camera(
        res=(args.res_w, args.res_h),
        pos=(args.x + args.radius * math.cos(az0),
             args.y + args.radius * math.sin(az0),
             args.eye_h),
        lookat=look, fov=45, GUI=False,
    )
    sc.build()

    leg = list(range(robot.n_dofs - 12, robot.n_dofs))
    robot.set_dofs_kp(np.full(12, 30.0, dtype=np.float32), leg)
    robot.set_dofs_kv(np.full(12, 1.5, dtype=np.float32), leg)
    robot.set_dofs_position(NOMINAL, leg)

    # Let it settle to a stable stand.
    for _ in range(200):
        robot.control_dofs_position(NOMINAL, leg)
        sc.step()
    z = float(to_np(robot.get_pos()).reshape(-1)[2])
    print(f"[hero] Go2 settled base z = {z:.3f} m")

    from PIL import Image

    # --- Hero shot ---
    out = cam.render()
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    rgb = to_np(rgb).astype("uint8")
    hero_path = os.path.join(args.outdir, "go2_hero.png")
    Image.fromarray(rgb).save(hero_path)
    print(f"[hero] wrote {hero_path}  shape={rgb.shape}")

    # --- Turntable: orbit camera around the standing robot ---
    frames = []
    for i in range(args.frames):
        az = az0 + 2 * math.pi * i / args.frames
        cam.set_pose(
            pos=(args.x + args.radius * math.cos(az),
                 args.y + args.radius * math.sin(az),
                 args.eye_h),
            lookat=look,
        )
        # keep the robot actively holding its stand while orbiting
        robot.control_dofs_position(NOMINAL, leg)
        sc.step()
        out = cam.render()
        rgb = out[0] if isinstance(out, (tuple, list)) else out
        frames.append(to_np(rgb).astype("uint8"))
    print(f"[hero] captured {len(frames)} turntable frames")

    # Encode MP4 via imageio (ffmpeg).
    mp4_path = os.path.join(args.outdir, "go2_room.mp4")
    try:
        import imageio.v2 as imageio
        imageio.mimsave(mp4_path, frames, fps=20, quality=8, macro_block_size=8)
        print(f"[hero] wrote {mp4_path}")
    except Exception as e:
        print(f"[hero] mp4 encode failed ({e}); dumping frames as PNGs")
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(args.outdir, f"frame_{i:03d}.png"))


if __name__ == "__main__":
    main()
