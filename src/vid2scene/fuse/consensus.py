"""Consensus gap-fill fusion of two reconstruction meshes.

The safe primitive (per the multi-reconstruction-fusion literature): take a
trusted *backbone* mesh and let a *donor* mesh contribute geometry ONLY where
the backbone has a hole (nearest-neighbour distance gate). Then fit a single
screened-Poisson surface to the gated, oriented, coloured point union. This
gap-fills via consensus without the doubled walls / smoothed-away detail that
blind volumetric averaging produces when methods disagree (e.g. MonoSDF's SDF
shell sits ~0.87 u outside PGSR's surface — averaging them is exactly wrong).

Runs on CPU (Open3D + SciPy); no GPU needed. This is the laptop-runnable heart
of the pipeline — point it at any two coloured meshes in the same metric frame.

CLI:
    python -m vid2scene fuse --backbone pgsr.ply --donor dn.ply --out consensus.ply
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ConsensusConfig:
    tau: float = 0.05           # donor contributes where >tau (metres) from the backbone surface
    poisson_depth: int = 11     # screened-Poisson octree depth (detail vs smoothing)
    trim_quantile: float = 0.02 # drop the lowest-density (most-extrapolated) Poisson verts
    target_tris: int = 350_000  # decimate to this many triangles (0 = keep full)
    # 4x4 rigid transform applied to the DONOR before fusing, to bring it into
    # the backbone's frame. Identity when both are already in the same world
    # frame (e.g. both from COLMAP). For a nerfstudio-frame donor, pass the
    # inverse dataparser transform.
    donor_transform: np.ndarray | None = None


def _load(path: str, transform: np.ndarray | None, neutral=(0.62, 0.60, 0.57)):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(path)
    m.compute_vertex_normals()
    if transform is not None:
        m.transform(transform)
        m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    C = np.asarray(m.vertex_colors) if m.has_vertex_colors() else np.tile(neutral, (len(V), 1))
    return V, N, C


def fuse_consensus(backbone_ply: str, donor_ply: str, out_ply: str,
                   cfg: ConsensusConfig | None = None) -> dict:
    """Fuse `donor` into `backbone` (gap-fill) -> screened-Poisson -> `out_ply`.

    Returns a dict of diagnostics (donor->backbone nn-distance stats, point
    counts, final mesh size) — the same numbers the benchmark uses to explain
    *why* a fusion helped or didn't.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    cfg = cfg or ConsensusConfig()
    Vb, Nb, Cb = _load(backbone_ply, None)
    Vd, Nd, Cd = _load(donor_ply, cfg.donor_transform)

    # alignment diagnostic: does the donor agree with the backbone where they overlap?
    tree = cKDTree(Vb)
    dist, _ = tree.query(Vd, k=1, workers=-1)
    diag = {
        "backbone_pts": int(len(Vb)), "donor_pts": int(len(Vd)),
        "donor_nn_median_m": float(np.median(dist)),
        "donor_nn_p25_m": float(np.percentile(dist, 25)),
        "donor_nn_p75_m": float(np.percentile(dist, 75)),
    }

    # donor contributes only its points far from the backbone surface (the gaps)
    gap = dist > cfg.tau
    diag["donor_gapfill_pts"] = int(gap.sum())
    diag["donor_gapfill_frac"] = float(gap.mean())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate([Vb, Vd[gap]]))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate([Nb, Nd[gap]]))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(np.concatenate([Cb, Cd[gap]]), 0, 1))

    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, scale=1.1, linear_fit=False)
    dens = np.asarray(dens)
    if cfg.trim_quantile > 0:
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, cfg.trim_quantile))

    # keep the largest connected component (drop stray Poisson bubbles)
    labels, _, areas = mesh.cluster_connected_triangles()
    labels, areas = np.asarray(labels), np.asarray(areas)
    keep = labels == int(np.argmax(areas))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[keep])
    mesh.remove_unreferenced_vertices()

    if cfg.target_tris and len(mesh.triangles) > cfg.target_tris:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=cfg.target_tris)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_ply, mesh, write_ascii=False, compressed=True)
    diag["out_tris"] = int(len(mesh.triangles))
    diag["out_ply"] = out_ply
    return diag


# --- nerfstudio -> COLMAP convenience (the donor-frame fix from the mip360 run) ---
NERFSTUDIO_TO_COLMAP = np.array([[1, 0, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]], dtype=np.float64)
