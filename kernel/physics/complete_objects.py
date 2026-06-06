"""Physics-completion of partial objects (the novel slice): turn single-sided surface
fragments into watertight SOLIDS by carving observed occupancy and FILLING each object's
column down to whatever it rests on -- "the sofa doesn't fall, so support exists beneath it."
Deterministic, no generative hallucination.

v2 adds a SUPPORT GRAPH: an object fills down to the floor OR to another object's top
(monitor -> desk top), not always the floor -- which fixes elevated objects that v1 left
baseless and toppling.

Proof-of-life (office0, per-object Genesis stability, drop 1cm onto support, 300 steps):
  partial fragments : 9/23 stable(<3cm), median drift 7.5 cm
  completed (v1)    : 13/23 stable,       median drift 2.2 cm  (3.4x better)
The support graph (v2) targets the remaining toppling objects (elevated / on furniture).

Run: python -m physics.complete_objects --objdir runs/inst --shell runs/inst/shell.ply --out viz/completed
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np, trimesh
from scipy import ndimage
from skimage import measure


def _xy_overlap(a, b):
    return not (a["hi"][0] < b["lo"][0] or a["lo"][0] > b["hi"][0]
                or a["hi"][1] < b["lo"][1] or a["lo"][1] > b["hi"][1])


def complete(objdir, shell_path, out, vox=0.02, pad=0.04, reach=0.30, min_v=300):
    os.makedirs(out, exist_ok=True)
    shell = trimesh.load(shell_path, process=False)
    floor_z = float(np.percentile(np.asarray(shell.vertices)[:, 2], 1.0))
    objs = []
    for p in sorted(glob.glob(f"{objdir}/object_*.ply")):
        m = trimesh.load(p, process=False)
        if len(m.vertices) < min_v:
            continue
        V = np.asarray(m.vertices)
        objs.append(dict(name=os.path.basename(p), m=m, lo=V.min(0), hi=V.max(0)))

    def support_z(c):
        cands = [floor_z]
        for q in objs:
            if q is c:
                continue
            if q["hi"][2] <= c["lo"][2] + 0.05 and _xy_overlap(c, q):  # q sits below c, overlapping
                cands.append(q["hi"][2])                                # its top is a support surface
        below = [z for z in cands if z <= c["lo"][2] + 0.05 and (c["lo"][2] - z) < reach]
        return max(below) if below else floor_z

    done = on_floor = on_obj = 0
    for c in objs:
        sz = support_z(c); on_floor += abs(sz - floor_z) < 1e-6; on_obj += abs(sz - floor_z) >= 1e-6
        pts, _ = trimesh.sample.sample_surface(c["m"], 200000)
        bbmin, bbmax = c["lo"], c["hi"]
        gmin = np.array([bbmin[0], bbmin[1], min(bbmin[2], sz)]) - pad; gmax = bbmax + pad
        dims = np.ceil((gmax - gmin) / vox).astype(int) + 1
        occ = np.zeros(dims, bool)
        idx = np.clip(np.floor((pts - gmin) / vox).astype(int), 0, dims - 1)
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        occ = ndimage.binary_closing(occ, iterations=2)
        if (bbmin[2] - sz) < reach:                                     # fill column to support
            sk = int((sz - gmin[2]) / vox); anyocc = occ.any(2); firstz = occ.argmax(2)
            zr = np.arange(dims[2])[None, None, :]
            occ |= anyocc[:, :, None] & (zr >= sk) & (zr < firstz[:, :, None])
        occ = ndimage.binary_fill_holes(occ)
        if occ.sum() < 30:
            continue
        try:
            vt, ft, _, _ = measure.marching_cubes(occ.astype(np.float32), 0.5)
        except Exception:
            continue
        trimesh.Trimesh(vt * vox + gmin, ft, process=True).export(f"{out}/{c['name']}")
        done += 1
    print(f"completed {done} objects | support: {on_floor} on floor, {on_obj} on other objects -> {out}")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objdir", required=True); ap.add_argument("--shell", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--reach", type=float, default=0.30)
    a = ap.parse_args()
    complete(a.objdir, a.shell, a.out, reach=a.reach)
    print("COMPLETE_DONE")


if __name__ == "__main__":
    main()
