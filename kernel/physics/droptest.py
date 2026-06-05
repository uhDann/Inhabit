"""Genesis drop test: load the reconstructed room as convex colliders (CoACD parts) +
each separated furniture instance as a rigid body, simulate under gravity, report settling.

The room->sim loop: segmented instances become rigid bodies; the room shell becomes the
static environment collider. Run physics/coacd_collider.py on the shell first.

Run (genesis env): python physics/droptest.py --shellparts viz/phys/shellparts --objects viz/phys
"""
import os, glob, argparse
import numpy as np
import genesis as gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shellparts", required=True, help="dir of CoACD convex part_*.obj")
    ap.add_argument("--objects", required=True, help="dir of object_*.obj (floor at z=0)")
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    try:
        gs.init(backend=gs.gpu, logging_level="warning")
    except Exception as e:
        print("gpu init failed -> cpu:", e); gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)))
    scene.add_entity(gs.morphs.Plane())                       # solid ground at z=0 (no fall-through)
    nparts = 0
    for p in sorted(glob.glob(f"{a.shellparts}/part_*.obj")):
        scene.add_entity(gs.morphs.Mesh(file=p, fixed=True, collision=True, convexify=True)); nparts += 1
    objs = []
    for p in sorted(glob.glob(f"{a.objects}/object_*.obj")):
        try:
            e = scene.add_entity(gs.morphs.Mesh(file=p, collision=True, convexify=True, pos=(0, 0, 0.03)))
            objs.append((os.path.basename(p), e))
        except Exception as ex:
            print("skip", os.path.basename(p), ex)
    print(f"building: {nparts} convex shell parts + {len(objs)} rigid objects", flush=True)
    scene.build()

    def pos(e):
        v = e.get_pos()
        try: return np.asarray(v.cpu().numpy())
        except Exception: return np.asarray(v)
    p0 = [pos(e) for _, e in objs]
    for _ in range(a.steps):
        scene.step()
    settled = 0
    for (n, _), q0, q1 in zip(objs, p0, [pos(e) for _, e in objs]):
        d = float(np.linalg.norm(q1 - q0))
        settled += d < 0.05
        print(f"{n}: moved {d:.3f} m, final z={float(q1[2]):.2f}", flush=True)
    print(f"=== {settled}/{len(objs)} settled (<5cm) ===", flush=True)


if __name__ == "__main__":
    main()
