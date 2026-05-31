"""Top-down view of the reconstructed room with the Habitat agent's recorded
navigation path drawn on it. A clean, honest depiction of the embodied stage:
the metric mesh becomes a navmesh the agent walks.

Usage:
    python topdown_path.py MESH.ply path.json scene.json OUT.png
"""
from __future__ import annotations
import argparse, json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("path_json"); ap.add_argument("scene_json")
    ap.add_argument("out")
    a = ap.parse_args()

    import open3d as o3d
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = o3d.io.read_triangle_mesh(a.mesh)
    V = np.asarray(m.vertices)
    C = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None

    # agent path: Habitat camera centres -> source/mesh frame
    poses = np.asarray(json.load(open(a.path_json))["poses"])
    Minv = np.linalg.inv(np.asarray(json.load(open(a.scene_json))["M_s2h"]))
    P = np.array([(Minv @ p)[:3, 3] for p in poses])

    # top-down plane = the two largest-spread axes of the room
    ext = V.max(0) - V.min(0)
    up = int(np.argmin(ext)); hx, hy = [i for i in range(3) if i != up]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(V), size=min(150000, len(V)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.scatter(V[idx, hx], V[idx, hy], s=1.2,
               c=(C[idx] if C is not None else "0.6"), alpha=0.5, linewidths=0)
    ax.plot(P[:, hx], P[:, hy], "-", color="#ff3b30", lw=2.4, zorder=5, label="agent path")
    ax.scatter(P[0, hx], P[0, hy], c="#ff3b30", s=60, zorder=6, edgecolor="white")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Agent navigation over the reconstructed room (top-down)", color="#222")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(a.out, facecolor="white", bbox_inches="tight")
    print(f"wrote {a.out}", flush=True)
    print("TOPDOWN_DONE", flush=True)


if __name__ == "__main__":
    main()
