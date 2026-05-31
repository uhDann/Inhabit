"""Top-down plot of the COLMAP capture: where the cameras were and which way
they looked, over the room footprint. Explains which walls got observed.

Usage:
    python cam_coverage.py --room <colmap_dir> --gsplat-examples <dir> \
        --mesh <pgsr.ply> --out coverage.png
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--gsplat-examples", required=True)
    ap.add_argument("--mesh", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--factor", type=int, default=4)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, a.gsplat_examples)
    from datasets.colmap import Parser
    p = Parser(data_dir=a.room, factor=a.factor, normalize=False, test_every=10**9)
    c2w = np.asarray(p.camtoworlds)             # (N,4,4)
    pos = c2w[:, :3, 3]                          # camera centres
    fwd = c2w[:, :3, 2]                          # OpenCV camera looks +Z
    print(f"{len(pos)} cameras", flush=True)

    # top-down = X (horizontal) vs Z (depth). Use mesh verts for the footprint.
    fig, ax = plt.subplots(figsize=(9, 9), dpi=120)
    if a.mesh:
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(a.mesh)
        V = np.asarray(m.vertices)
        C = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
        # subsample for speed
        idx = np.random.default_rng(0).choice(len(V), size=min(120000, len(V)), replace=False)
        ax.scatter(V[idx, 0], V[idx, 2], s=1,
                   c=(C[idx] if C is not None else "0.6"), alpha=0.35, linewidths=0)

    # camera centres + view directions
    ax.scatter(pos[:, 0], pos[:, 2], s=18, c="red", zorder=5, label="camera positions")
    L = 0.25 * (pos[:, 0].max() - pos[:, 0].min())
    ax.quiver(pos[:, 0], pos[:, 2], fwd[:, 0], fwd[:, 2],
              color="red", alpha=0.35, scale=18, width=0.003, zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("X (m, COLMAP)"); ax.set_ylabel("Z (m, COLMAP)")
    ax.set_title(f"Capture coverage (top-down): {len(pos)} cameras\n"
                 "red = camera positions + look directions; dots = reconstructed room")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, facecolor="white")
    print(f"wrote {a.out}", flush=True)
    print("COVERAGE_DONE", flush=True)


if __name__ == "__main__":
    main()
