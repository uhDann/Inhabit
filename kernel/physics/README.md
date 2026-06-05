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
