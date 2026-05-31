#!/usr/bin/env python3
"""
Genesis Step 1 — the credibility milestone.

Load the gravity-aligned reconstructed room mesh as a STATIC collider, enable
gravity, drop a rigid SPHERE above the floor, step ~2 s, and confirm the sphere
comes to rest ON the floor (settles to ~ floor + radius; no tunnelling, no
falling to infinity).

Outputs:
  - height-vs-time CSV + matplotlib PNG
  - optional Genesis camera frames (PNG) if a renderer is available
  - a PASS/FAIL verdict printed to stdout

Usage:
  python step1_sphere_drop.py --backend gpu   # or cpu
"""
import argparse
import os
import numpy as np

import genesis as gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--mesh", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned_decim.obj")
    ap.add_argument("--outdir", default="/cs/student/projects3/2023/dkozlov/genesis-work/step1_out")
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--drop_x", type=float, default=3.0)
    ap.add_argument("--drop_y", type=float, default=1.16)
    ap.add_argument("--drop_z", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--render", action="store_true", help="try to render camera frames")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    backend = gs.gpu if args.backend == "gpu" else gs.cpu
    gs.init(backend=backend)
    print(f"[step1] gs.init OK on backend={args.backend}")

    dt = 1.0 / 100.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)),
        show_viewer=False,
    )

    # Static room mesh collider (fixed=True so it never moves).
    room = scene.add_entity(
        gs.morphs.Mesh(
            file=args.mesh,
            fixed=True,
            collision=True,
            convexify=False,   # keep the concave room interior for real collision
        ),
    )
    print(f"[step1] added room collider from {args.mesh}")

    # Rigid sphere dropped above the floor.
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=args.radius,
            pos=(args.drop_x, args.drop_y, args.drop_z),
        ),
    )
    print(f"[step1] added sphere r={args.radius} at "
          f"({args.drop_x},{args.drop_y},{args.drop_z})")

    cam = None
    if args.render:
        try:
            cam = scene.add_camera(
                res=(640, 480),
                pos=(args.drop_x + 2.5, args.drop_y - 2.5, 1.6),
                lookat=(args.drop_x, args.drop_y, 0.3),
                fov=50,
                GUI=False,
            )
            print("[step1] camera added")
        except Exception as e:
            print(f"[step1] camera unavailable, continuing headless: {e}")
            cam = None

    scene.build()
    print("[step1] scene.build() OK")

    n_steps = int(args.seconds / dt)
    times, heights = [], []
    frame_idx = 0
    for i in range(n_steps):
        scene.step()
        pos = sphere.get_pos()
        try:
            pos = pos.cpu().numpy()
        except Exception:
            pos = np.asarray(pos)
        z = float(np.asarray(pos).reshape(-1)[2])
        times.append(i * dt)
        heights.append(z)
        if cam is not None and i % 25 == 0:
            try:
                rgb = cam.render()
                arr = rgb[0] if isinstance(rgb, (tuple, list)) else rgb
                try:
                    arr = arr.cpu().numpy()
                except Exception:
                    arr = np.asarray(arr)
                from PIL import Image
                Image.fromarray(arr.astype(np.uint8)).save(
                    os.path.join(args.outdir, f"frame_{frame_idx:03d}.png"))
                frame_idx += 1
            except Exception as e:
                if i == 0:
                    print(f"[step1] render failed, dropping frames: {e}")
                cam = None

    times = np.array(times); heights = np.array(heights)
    csv = os.path.join(args.outdir, "height_vs_time.csv")
    np.savetxt(csv, np.column_stack([times, heights]),
               delimiter=",", header="t_sec,sphere_z", comments="")
    print(f"[step1] wrote {csv}")

    final_z = float(heights[-1])
    settle_z = float(np.median(heights[-25:]))  # last 0.25 s
    expected_rest = args.radius  # floor at z=0, center rests at ~radius
    moved_last = float(abs(heights[-1] - heights[-10]))
    print("\n========== STEP 1 RESULT ==========")
    print(f"start z       = {heights[0]:.3f} m")
    print(f"min z reached = {heights.min():.3f} m")
    print(f"final z       = {final_z:.3f} m")
    print(f"settled z     = {settle_z:.3f} m (median of last 0.25 s)")
    print(f"expected rest = {expected_rest:.3f} m (floor 0 + radius)")
    print(f"motion in last 0.1 s = {moved_last:.4f} m")

    rested_on_floor = (abs(settle_z - expected_rest) < 0.06) and (moved_last < 0.01)
    tunneled = settle_z < -0.1
    fell_to_infinity = settle_z < -1.0
    if fell_to_infinity:
        verdict = "FAIL: sphere fell through to infinity (no collision)"
    elif tunneled:
        verdict = "FAIL: sphere tunnelled below floor"
    elif rested_on_floor:
        verdict = "PASS: sphere rested ON the floor at ~ radius height"
    else:
        verdict = (f"PARTIAL: settled at {settle_z:.3f} (not yet at radius "
                   f"{expected_rest:.3f} or still moving)")
    print(f"VERDICT: {verdict}")
    print("===================================")

    # Matplotlib plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4.5))
        plt.plot(times, heights, lw=2, label="sphere center z")
        plt.axhline(expected_rest, color="g", ls="--", label=f"floor+radius={expected_rest:.2f}")
        plt.axhline(0, color="k", ls=":", label="floor (z=0)")
        plt.xlabel("time (s)"); plt.ylabel("sphere height z (m)")
        plt.title("Genesis Step 1: sphere drop onto reconstructed room floor")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        png = os.path.join(args.outdir, "height_vs_time.png")
        plt.savefig(png, dpi=110)
        print(f"[step1] wrote plot {png}")
    except Exception as e:
        print(f"[step1] plot failed: {e}")


if __name__ == "__main__":
    main()
