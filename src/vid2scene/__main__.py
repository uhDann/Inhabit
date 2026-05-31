"""vid2scene — phone video -> metric 3D reconstruction -> robot-explorable world.

A single CLI over the pipeline stages. The CPU stages (ingest, fuse, benchmark,
viz, embodied export) run on a laptop; the GPU reconstruction stage (PGSR /
DN-Splatter / MonoSDF training) is driven by scripts/remote/ on a CUDA box —
see docs/ARCHITECTURE.md.

Stages:
  ingest     select + validate keyframes from a video            [CPU]
  reconstruct  (info) how to run the multi-method GPU stage      [GPU]
  fuse       consensus gap-fill of two reconstruction meshes     [CPU]
  benchmark  collate GT-mesh metrics into a comparison table     [CPU]
  viz        decimate a mesh for the web viewer                  [CPU]
  embodied   export a sim-ready GLB (capture -> Genesis/Habitat) [CPU]
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vid2scene", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="[CPU] select + validate keyframes")
    pi.add_argument("video"); pi.add_argument("--out", required=True)
    pi.add_argument("--target-frames", type=int, default=60)
    pi.add_argument("--stride", type=int, default=1)
    pi.add_argument("--blur-threshold", type=float, default=0.6)

    sub.add_parser("reconstruct", help="[GPU] info: run PGSR/DN-Splatter/MonoSDF (see scripts/remote)")

    pf = sub.add_parser("fuse", help="[CPU] consensus gap-fill of two meshes")
    pf.add_argument("--backbone", required=True, help="trusted mesh (e.g. PGSR)")
    pf.add_argument("--donor", required=True, help="gap-fill mesh (e.g. DN-Splatter)")
    pf.add_argument("--out", required=True)
    pf.add_argument("--tau", type=float, default=0.05)
    pf.add_argument("--depth", type=int, default=11)
    pf.add_argument("--donor-from-nerfstudio", action="store_true",
                    help="apply inverse nerfstudio dataparser transform to the donor")

    pb = sub.add_parser("benchmark", help="[CPU] collate GT-mesh metrics -> table")
    pb.add_argument("--eval-dir", required=True)
    pb.add_argument("--no-markdown", action="store_true")

    pv = sub.add_parser("viz", help="[CPU] decimate a mesh for the web viewer")
    pv.add_argument("--in", dest="inp", required=True); pv.add_argument("--out", required=True)
    pv.add_argument("--tris", type=int, default=300_000)

    pe = sub.add_parser("embodied", help="[CPU] export a sim-ready GLB (Genesis/Habitat)")
    pe.add_argument("--mesh", required=True); pe.add_argument("--out", required=True)
    pe.add_argument("--splat", default=None); pe.add_argument("--up", default="y", choices=["y", "z"])
    pe.add_argument("--scene-json", default=None)

    a = p.parse_args(argv)

    if a.cmd == "ingest":
        from .ingest import IngestConfig, run_ingest
        cfg = IngestConfig(target_frames=a.target_frames, sample_stride=a.stride,
                           blur_rel_threshold=a.blur_threshold)
        s = run_ingest(a.video, a.out, cfg)
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))
        print(f"\nwrote keyframes + report to: {a.out}/report.html")
        return 0

    if a.cmd == "reconstruct":
        print("The GPU reconstruction stage runs three methods on a CUDA box.\n"
              "Drivers (set up envs, train, extract TSDF/SDF meshes):\n"
              "  scripts/remote/setup_pgsr.sh + launch_pgsr.sh\n"
              "  scripts/remote/setup_dnsplat.sh\n"
              "  scripts/remote/setup_monosdf.sh + launch_monosdf_tight.sh\n"
              "Each emits a coloured mesh; fuse them with `vid2scene fuse`.\n"
              "See docs/ARCHITECTURE.md and docs/BENCHMARK.md.")
        return 0

    if a.cmd == "fuse":
        from .fuse import ConsensusConfig, fuse_consensus
        from .fuse.consensus import NERFSTUDIO_TO_COLMAP
        cfg = ConsensusConfig(tau=a.tau, poisson_depth=a.depth,
                              donor_transform=NERFSTUDIO_TO_COLMAP if a.donor_from_nerfstudio else None)
        print(json.dumps(fuse_consensus(a.backbone, a.donor, a.out, cfg), indent=2))
        return 0

    if a.cmd == "benchmark":
        from .benchmark import collate
        collate(a.eval_dir, markdown=not a.no_markdown)
        return 0

    if a.cmd == "viz":
        from .viz import decimate
        print(json.dumps(decimate(a.inp, a.out, a.tris), indent=2))
        return 0

    if a.cmd == "embodied":
        from .embodied import export_for_genesis
        print(json.dumps(export_for_genesis(a.mesh, a.out, a.splat, a.up, a.scene_json), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
