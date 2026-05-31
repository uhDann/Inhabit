#!/bin/bash
# Render clean interior fly-throughs of the Replica reconstructions (DN-Splatter
# meshes) for several scenes, and encode each to mp4. Self-contained (defines its
# own vars) so it is safe to launch inside screen.
ROOT=/cs/student/projects3/2023/dkozlov
W=$ROOT/work
PY=$ROOT/conda-envs/pgsr/bin/python
RD=$ROOT/vid2scene/scripts/remote
mkdir -p $W/fly
for s in room0 room1 room2; do
  DN=$W/out/dn_${s}/mesh/Open3dTSDFfusion_mesh.ply
  SC=$ROOT/datasets/replica/Replica/${s}
  $PY $RD/render_mesh_traj.py "$DN" "$SC" "$W/fly/${s}" --stride 16 --scale 0.55 > $W/fly_${s}.log 2>&1
  ffmpeg -y -loglevel error -framerate 14 -i "$W/fly/${s}_%04d.png" -c:v mpeg4 -q:v 4 "$W/fly_${s}.mp4"
  echo "done ${s}: $(ls $W/fly/${s}_*.png 2>/dev/null | wc -l) frames"
done
echo FLY_DONE2 > $W/fly2.done
