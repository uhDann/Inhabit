"""Phase 3 (#21): domain-randomized physics eval inside the RECONSTRUCTED ROOM.

Loads the kernel's decimated room-shell mesh as the static collider and drops the
per-object collision proxies into it, with randomized pose / mass / friction. All
trials are batched into ONE scene build (one SDF), which is fast and avoids the OOM
from rebuilding the room SDF per trial. Reports a physical-plausibility score.

Run in the genesis env.
"""
from __future__ import annotations
import json, math
import numpy as np
import genesis as gs

ASSETS = "/tmp/ikernel/assets"


def to_np(x):
    try:
        return x.cpu().numpy()
    except Exception:
        return np.asarray(x)


def main():
    rng = np.random.default_rng(0)
    A = json.load(open(f"{ASSETS}/assets.json"))
    (x0, y0), (x1, y1) = A["room_xy"]
    gs.init(backend=gs.gpu)
    sc = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01, gravity=(0, 0, -9.81)),
                  show_viewer=False, renderer=gs.renderers.Rasterizer())
    # the reconstructed room as CoACD convex parts -> cheap convex colliders, no SDF
    rooms = A.get("room_parts") or [A["room"]]
    for rp in rooms:
        sc.add_entity(gs.morphs.Mesh(file=rp, fixed=True, collision=True, convexify=True))
    print(f"=== Phase 3 (#21): physics eval INSIDE the reconstructed room "
          f"({len(rooms)} CoACD convex parts) ===")

    N = 4
    items = [(o, k) for o in A["objects"] for k in range(N)]
    gx = np.linspace(x0 + 0.9, x1 - 0.9, 4)
    gy = np.linspace(y0 + 0.9, y1 - 0.9, max(2, (len(items) + 3) // 4))
    cells = [(x, y) for y in gy for x in gx][:len(items)]
    entries = []
    for (obj, k), (x, y) in zip(items, cells):
        dens = float(rng.uniform(200, 800)); fric = float(rng.uniform(0.3, 1.2))
        yaw = float(rng.uniform(0, 2 * math.pi))
        try:
            mat = gs.materials.Rigid(rho=dens, friction=fric)
        except Exception:
            mat = gs.materials.Rigid(rho=dens) if hasattr(gs.materials, "Rigid") else None
        morph = gs.morphs.Mesh(file=obj["path"], fixed=False, collision=True, convexify=True,
                               pos=(float(x), float(y), 0.5 + obj["bottom_offset"]),
                               euler=(0, 0, math.degrees(yaw)))
        body = sc.add_entity(morph, material=mat) if mat is not None else sc.add_entity(morph)
        entries.append((obj, body))
    sc.build()

    steps = 350
    last = None
    for i in range(steps):
        sc.step()
        if i == steps - 11:
            last = np.array([to_np(b.get_pos()).reshape(-1)[:3] for _, b in entries])
    final = np.array([to_np(b.get_pos()).reshape(-1)[:3] for _, b in entries])
    speed = np.linalg.norm(final - last, axis=1) / (10 * 0.01)

    from collections import defaultdict
    agg = defaultdict(lambda: dict(n=0, rest=0, notun=0, pen=[]))
    for (obj, _), fz, sp in zip(entries, final[:, 2], speed):
        bo = obj["bottom_offset"]; lowest = fz - bo
        a = agg[obj["name"]]; a["n"] += 1
        a["rest"] += int(sp < 0.06 and lowest > -0.05 and fz > -0.2)
        a["notun"] += int(lowest > -0.05)
        a["pen"].append(max(0.0, -lowest))
    hdr = ["object", "trials", "rest_ok", "no_tunnel", "stable%", "pen_cm"]
    w = [12, 7, 8, 10, 9, 8]
    print("".join(c.ljust(wi) for c, wi in zip(hdr, w)))
    print("-" * sum(w))
    tot = tr = 0
    for name, a in agg.items():
        vals = [name, a["n"], f"{a['rest']}/{a['n']}", f"{a['notun']}/{a['n']}",
                f"{100*a['rest']//a['n']}", f"{100*np.mean(a['pen']):.1f}"]
        print("".join(str(v).ljust(wi) for v, wi in zip(vals, w)))
        tot += a["n"]; tr += a["rest"]
    nt = sum(a["notun"] for a in agg.values())
    print(f"\noverall: {tr}/{tot} stable, {nt}/{tot} no-tunnel inside the reconstructed room")


if __name__ == "__main__":
    main()
