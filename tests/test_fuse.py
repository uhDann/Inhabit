"""Tests for the consensus fusion. The geometry path needs Open3D + SciPy; it is
skipped when those are unavailable so the pure-Python suite still runs."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vid2scene.fuse.consensus import ConsensusConfig, NERFSTUDIO_TO_COLMAP  # noqa: E402


def test_config_defaults():
    cfg = ConsensusConfig()
    assert cfg.tau > 0 and cfg.poisson_depth >= 8 and cfg.target_tris > 0


def test_nerfstudio_transform_is_rigid():
    import numpy as np
    R = NERFSTUDIO_TO_COLMAP[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)        # orthonormal
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-9           # no scaling


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("open3d") is None
    or __import__("importlib").util.find_spec("scipy") is None,
    reason="open3d/scipy not installed",
)
def test_fuse_two_boxes_runs(tmp_path):
    """Fuse two overlapping coloured boxes; the result should be a non-empty mesh."""
    import numpy as np
    import open3d as o3d
    from vid2scene.fuse.consensus import fuse_consensus

    def box(path, shift):
        m = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
        m.translate(np.asarray(shift, float))
        m.compute_vertex_normals()
        m.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.io.write_triangle_mesh(path, m)

    a, b, out = (str(tmp_path / f"{n}.ply") for n in ("a", "b", "out"))
    box(a, [0, 0, 0]); box(b, [0.5, 0, 0])
    diag = fuse_consensus(a, b, out, ConsensusConfig(target_tris=0, poisson_depth=6))
    assert os.path.exists(out) and diag["out_tris"] > 0
    assert 0.0 <= diag["donor_gapfill_frac"] <= 1.0
