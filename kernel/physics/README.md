# Room -> physics (Genesis)

Turns a reconstructed + segmented room into a Genesis scene: the room shell becomes the
static environment collider, each separated furniture instance becomes a rigid body.

```
python physics/coacd_collider.py viz/shell.ply viz/phys/shellparts   # decompose shell -> convex parts
python physics/droptest.py --shellparts viz/phys/shellparts --objects viz/phys
```

## Lessons
- **Genesis OOMs** loading a full room mesh as one fixed collider — it builds an SDF
  (capped ~450^3 cells). Fix: CoACD-decompose the shell into convex parts (cheap convex
  colliders, no SDF). 40 parts is plenty for a room.
- Drop objects from a small lift (`pos=(0,0,0.03)`) and add a solid `gs.morphs.Plane()`
  ground so nothing falls through gaps between convex parts.

## Honest status
The sim is valid (no OOM, gravity/contacts work, objects come to rest on the floor at
sensible heights), but it is a proof-of-concept: the separated "furniture" are **surface
fragments** (e.g. a desk top without legs), not solid closed bodies, so they slide/topple
rather than behave like real furniture. Only ~3/23 stay within 5 cm of their start.

The real unblock is upstream: per-object **watertight completion** (merge over-segmented
fragments by SAM2 instance track, then solidify each via Poisson) so each piece is a solid
object. That makes both the separation and the physics meaningful — see the furniture
completion experiment.

## Physics-completion of the unseen (`complete_objects.py`) — the novel slice

Instead of hallucinating the occluded geometry generatively, INFER it from physics: a
resting object must have support beneath it, so carve the observed occupancy and fill each
object's column down to whatever it rests on (floor, or another object's top via a support
graph). Deterministic, no generative prior. Closest prior work uses physics only as a
stability penalty (PhyRecon) or to *pick* a generative guess (HoloScene); using physics as
the *completion oracle* for unseen scene geometry from passive video is the defensible
novel slice (cf. Vysics, which does this only for a single object with active robot contact).

Proof-of-life (office0, per-object Genesis stability — drop 1 cm onto support, 300 steps):

| input | watertight | stable (<3cm drift) | median drift |
|---|---|---|---|
| partial fragments | no | 9/23 | 7.5 cm |
| **physics-completed** | **yes (23/23)** | **13/23** | **2.2 cm** (3.4x better) |

Honest status: real, measured improvement (median drift 3.4x lower; all objects watertight)
but partial — the remaining unstable objects are elevated items whose support graph / fill
needs refinement. This is the kernel of a "video -> physics-true simulatable world" thesis.
