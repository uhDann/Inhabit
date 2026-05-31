#!/usr/bin/env python3
"""
Genesis Step 2 (stretch) — Go2 quadruped standing on the reconstructed room floor.

Genesis 1.0 Go2 URDF has 18 DOFs: a 6-DOF free base (root_joint) followed by 12
actuated leg joints (FL/FR/RL/RR x hip/thigh/calf). We hold the standard Genesis
Go2 standing pose with PD position control (kp=20, kv=0.5), let it settle, then
apply a small cyclic thigh/calf drive to nudge it forward. We log base height z
and forward x to confirm it stays on the floor and translates.

This is a hand-tuned stand + nudge demo (not a trained RL gait). Run after Step 1.
Usage: python step2_go2.py --backend gpu
"""
import argparse
import os
import numpy as np
import genesis as gs

GO2_URDF = ("/cs/student/projects3/2023/dkozlov/conda-envs/genesis/lib/python3.10/"
            "site-packages/genesis/assets/urdf/go2/urdf/go2.urdf")

# Standard Genesis Go2 nominal standing joint angles, ordered as the 12 leg DOFs
# appear after the free base (hips, then thighs, then calves — matches joint list:
# FL/FR/RL/RR hip, FL/FR/RL/RR thigh, FL/FR/RL/RR calf).
NOMINAL = np.array([
    0.0, 0.0, 0.0, 0.0,        # hips
    0.8, 0.8, 0.8, 0.8,        # thighs (symmetric — verified to stand on flat ground)
    -1.5, -1.5, -1.5, -1.5,    # calves
], dtype=np.float32)
KP, KV = 30.0, 1.5             # PD gains verified to hold a clean stand (base z~0.28)


def to_np(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--mesh", default="/cs/student/projects3/2023/dkozlov/genesis-work/room0_aligned_decim.obj")
    ap.add_argument("--urdf", default=GO2_URDF)
    ap.add_argument("--outdir", default="/cs/student/projects3/2023/dkozlov/genesis-work/step2_out")
    ap.add_argument("--start_x", type=float, default=5.0)
    ap.add_argument("--start_y", type=float, default=1.0)
    ap.add_argument("--start_z", type=float, default=0.42)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--walk", action="store_true", help="apply forward gait after settling")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    gs.init(backend=gs.gpu if args.backend == "gpu" else gs.cpu)
    dt = 1.0 / 100.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0, 0, -9.81)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Mesh(file=args.mesh, fixed=True, collision=True, convexify=False))
    robot = scene.add_entity(
        gs.morphs.URDF(file=args.urdf, pos=(args.start_x, args.start_y, args.start_z))
    )
    scene.build()
    print("[step2] scene.build() OK")

    jnames = [j.name for j in robot.joints]
    print(f"[step2] n_dofs={robot.n_dofs}  joints={jnames}")
    # 12 actuated leg DOFs are the last 12 (free base = first 6).
    leg_idx = list(range(robot.n_dofs - 12, robot.n_dofs))
    robot.set_dofs_kp(np.full(12, KP, dtype=np.float32), leg_idx)
    robot.set_dofs_kv(np.full(12, KV, dtype=np.float32), leg_idx)
    robot.set_dofs_position(NOMINAL, leg_idx)  # start at the standing pose

    times, base_z, base_x = [], [], []
    n_steps = int(args.seconds / dt)
    settle = int(1.5 / dt)
    for i in range(n_steps):
        target = NOMINAL.copy()
        if args.walk and i > settle:
            t = (i - settle) * dt
            phase = np.sin(2 * np.pi * 1.5 * t)
            # diagonal trot: FL+RR vs FR+RL thighs out of phase
            target[4] += 0.20 * phase     # FL thigh
            target[7] += 0.20 * phase     # RR thigh
            target[5] -= 0.20 * phase     # FR thigh
            target[6] -= 0.20 * phase     # RL thigh
        robot.control_dofs_position(target, leg_idx)
        scene.step()
        p = to_np(robot.get_pos()).reshape(-1)
        times.append(i * dt); base_z.append(float(p[2])); base_x.append(float(p[0]))

    times = np.array(times); base_z = np.array(base_z); base_x = np.array(base_x)
    np.savetxt(os.path.join(args.outdir, "go2_traj.csv"),
               np.column_stack([times, base_x, base_z]),
               delimiter=",", header="t,base_x,base_z", comments="")

    settle_z = float(np.median(base_z[-50:]))
    dx = float(base_x[-1] - base_x[settle])
    print("\n========== STEP 2 RESULT ==========")
    print(f"start base z={base_z[0]:.3f}  settled base z={settle_z:.3f}")
    print(f"forward dx after settle = {dx:+.3f} m")
    stood = settle_z > 0.20
    print("VERDICT:", "STAND OK (base held ~0.3m above floor)" if stood
          else f"collapsed (z={settle_z:.2f})",
          "| moved forward" if (args.walk and abs(dx) > 0.03) else "")
    print("===================================")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(times, base_z); ax[0].axhline(0, color="k", ls=":")
        ax[0].set_title("Go2 base height z"); ax[0].set_xlabel("t (s)"); ax[0].set_ylabel("z (m)"); ax[0].grid(alpha=0.3)
        ax[1].plot(times, base_x); ax[1].set_title("Go2 base x (forward)")
        ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("x (m)"); ax[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "go2_traj.png"), dpi=110)
        print("[step2] wrote go2_traj.png")
    except Exception as e:
        print(f"[step2] plot failed: {e}")


if __name__ == "__main__":
    main()
