"""Geometric superpoints over a mesh: normal-coherent region growing on the vertex
graph. Pooling instance votes per-superpoint (instead of per-voxel) is the key
robustness win -- a superpoint will not straddle a chair-leg/floor normal discontinuity,
so contact patches separate cleanly and a single noisy 2D mask can't carve an object in
half. (SAI3D-style, simplified.)
"""
from __future__ import annotations
import numpy as np


def vertex_adjacency(faces, n_verts):
    from collections import defaultdict
    adj = defaultdict(set)
    for a, b, c in faces:
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def superpoints(verts, faces, normals, normal_thresh=0.92, max_size=4000):
    """Region-grow vertices with similar normals into superpoints. Returns sp[V] int."""
    n = len(verts)
    adj = vertex_adjacency(faces, n)
    sp = np.full(n, -1, np.int32)
    cur = 0
    order = np.arange(n)
    for s in order:
        if sp[s] >= 0:
            continue
        stack = [s]; sp[s] = cur; size = 0; seed_n = normals[s]
        while stack and size < max_size:
            v = stack.pop(); size += 1
            for w in adj[v]:
                if sp[w] < 0 and float(normals[w] @ seed_n) > normal_thresh:
                    sp[w] = cur; stack.append(w)
        cur += 1
    return sp, cur


def pool_votes(sp, n_sp, vert_votes, n_labels, margin=1.3):
    """vert_votes[V,n_labels] -> per-vertex object label after pooling per superpoint.
    A superpoint commits to an object only if its top vote beats the runner-up by
    `margin` and beats the shell (label 0); else -> shell (0).

    Vectorised: aggregate all votes per superpoint with one scatter-add (O(V+n_sp)),
    not a Python loop over superpoints (which is O(n_sp*V) and unusable at 1.5M verts)."""
    agg = np.zeros((n_sp, n_labels), np.float64)
    np.add.at(agg, sp, vert_votes)                       # sum votes per superpoint
    top = agg.argmax(1)                                  # [n_sp]
    srt = np.sort(agg, 1)[:, ::-1]
    conf = srt[:, 0] > margin * (srt[:, 1] + 1e-6)
    sp_label = np.where(conf & (top != 0), top, 0).astype(np.int32)  # 0 = room shell
    return sp_label[sp]                                  # broadcast back to vertices
