"""Visualization helpers: decimate meshes for the browser, prep viewer payloads.

The interactive viewers live in ../../viewer/*.html (three.js + PLYLoader, and
a GaussianSplats3D splat viewer). This module produces the lightweight,
vertex-coloured .ply payloads they load. CPU-only (Open3D).
"""

from .decimate import decimate

__all__ = ["decimate"]
