"""vid2scene — video to geometrically coherent 3D scene.

Pipeline stages (each a subpackage):
  1. ingest    — frame selection + camera-systems validation  [CPU, implemented]
  2. geometry  — feed-forward reconstruction (MapAnything/VGGT) [GPU, planned]
  3. radiance  — Gaussian Splatting / NeRF baseline             [GPU, planned]
  4. semantics — geometry-coherent 3D labels                    [GPU, planned]
  5. eval      — metrics + sanity checks                        [planned]
"""

__version__ = "0.1.0"
