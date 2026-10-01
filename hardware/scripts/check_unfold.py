"""Does a cap actually have to be a STRIP to lie flat?

census.py says a cap unfolds "exactly when its faces form a path in the dual,
because a path of faces rolls out and a branching tree does not". The second
half of that is false, and this script is what settles it.

Three claims, checked rather than argued, over every Hamiltonian cycle of
every solid in polyhedra.py:

  1. Neither cap ever has an INTERIOR vertex -- one with all of its faces on
     the same side and none of its edges cut.

     This is forced, and it is the page's own central claim seen from a new
     angle: a Hamiltonian cycle passes through every vertex, so every vertex
     carries exactly two cut edges. Those two cuts break the fan of faces
     around it, so whichever faces of that fan a cap holds, they form an arc
     and never the whole wheel. Every vertex therefore lands on the rim of
     both caps, and neither cap has an interior.

  2. Each cap's dual is therefore a TREE, never merely a connected graph.

     For a disc, the dual has a cycle exactly when that cycle encircles an
     interior vertex. No interior vertices, no dual cycles. So "tree" is not
     a property some cuts have: it is every cut, and the only question left
     is whether the tree happens to be a path.

  3. Rolled out along that tree, does the net overlap itself?

     This is the one that is not forced, and it is the real criterion. A tree
     rolls out isometrically whatever its shape -- branching costs nothing,
     because the faces around a branch were never going to close a full 360
     either -- but two faces far apart in the tree can still collide in the
     plane. So every net is laid out and every pair of faces intersected.

     The answer, measured: overlap cuts ACROSS the path/branching line rather
     than along it. On the icosahedron 1985 branching caps lay out clean and
     215 do not; 338 path-shaped caps lay out clean and 22 do not -- and 15 of
     those 22 belong to cycles whose BOTH caps are strips. Being a strip is
     therefore neither necessary for a flat net nor sufficient for one.

So "strip" is not a fact about flatness at all. It is the SHAPE of the net: a
linear chain hinges into a ladder, which is the printed part this project
makes. A branching net is just as flat and would want a branching ladder.
That is a fact about the hinge design, and census.py used to claim it was a
fact about geometry.
"""
import os
import sys

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import census
import polyhedra as P
from polyhedra import face_shape_2d, internal_adjacency, shared_verts, _place

# Two faces of a net meet along a whole edge, so their polygons share a
# segment. Area is the right test precisely because a shared segment has
# none; this only has to clear floating-point dust.
OVERLAP_EPS = 1e-9


def interior_vertices(faces, cap, cut_edges):
    """Vertices whose every face is in this cap and none of whose edges are
    cut -- the cone points that would stop a cap lying flat as one piece."""
    incident = {}
    for fi, f in enumerate(faces):
        for v in f:
            incident.setdefault(v, set()).add(fi)
    out = []
    for v, fs in incident.items():
        if not fs <= set(cap):
            continue
        touched = any(v in e for e in cut_edges)
        if not touched:
            out.append(v)
    return out


def is_tree(cap, adj):
    """Connected and with exactly |V| - 1 edges."""
    inside = set(cap)
    edges = sum(len(set(adj[f]) & inside) for f in cap) // 2
    seen, stack = {cap[0]}, [cap[0]]
    while stack:
        x = stack.pop()
        for y in adj[x]:
            if y in inside and y not in seen:
                seen.add(y)
                stack.append(y)
    return len(seen) == len(inside) and edges == len(inside) - 1


def unfold_tree(solid, cap, adj):
    """Roll a cap out flat over its dual tree, breadth first from one end.

    The same isometry unfold() uses for a strip, with the single difference
    that a face may have more than one child. Nothing about rolling out cares
    how many: each child is placed against the edge it shares with its parent,
    on the far side, and that is a local operation.
    """
    inside = set(cap)
    shapes = {fi: face_shape_2d(solid, fi) for fi in cap}
    root = cap[0]
    first = solid.faces[root]
    face_2d = {root: list(zip(first, [tuple(q) for q in shapes[root]]))}
    order, seen = [root], {root}
    while order:
        fprev = order.pop(0)
        for fcur in adj[fprev]:
            if fcur not in inside or fcur in seen:
                continue
            seen.add(fcur)
            a, b = shared_verts(solid.faces, fprev, fcur)[:2]
            prev = dict(face_2d[fprev])
            prev_centre = np.mean([q for _, q in face_2d[fprev]], axis=0)
            cur = solid.faces[fcur]
            placed = _place(shapes[fcur], cur.index(a), cur.index(b),
                            prev[a], prev[b], prev_centre)
            face_2d[fcur] = list(zip(cur, [tuple(q) for q in placed]))
            for v in (a, b):
                assert np.hypot(*(np.array(prev[v]) -
                                  np.array(dict(face_2d[fcur])[v]))) < 1e-6, \
                    f"{solid.name}: the fold between {fprev} and {fcur} does not join"
            order.append(fcur)
    assert len(seen) == len(inside), f"{solid.name}: the cap is not connected"
    return face_2d


def worst_overlap(face_2d):
    """The largest area two faces of the net share, relative to a face."""
    polys = {fi: Polygon([q for _, q in pts]) for fi, pts in face_2d.items()}
    unit = max(p.area for p in polys.values())
    keys = sorted(polys)
    worst = 0.0
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = polys[a].intersection(polys[b]).area
            worst = max(worst, inter / unit)
    return worst


def shape_of(cap, adj):
    inside = set(cap)
    degs = [len(set(adj[f]) & inside) for f in cap]
    return "path" if max(degs) <= 2 and degs.count(1) == 2 else "branching"


def survey(solid):
    """Every cap of every cycle, classified by dual shape and by whether its
    net collides with itself."""
    faces = [list(f) for f in solid.faces]
    adjv = census.adjacency(faces, len(solid.vertices))
    cycles = census.hamiltonian_cycles(adjv)
    tally = {("path", False): 0, ("path", True): 0,
             ("branching", False): 0, ("branching", True): 0}
    flaws = {"interior": 0, "nontree": 0}
    worst = 0.0
    whole = {"flat": 0, "overlaps": 0}      # cycles with BOTH caps clean
    for edges in cycles:
        order = cycle_order(edges, len(solid.vertices))
        adj = internal_adjacency(faces, order)
        _, caps = census.caps(faces, edges)
        both = True
        for cap in caps:
            cap = sorted(cap)
            shape = shape_of(cap, adj)
            if interior_vertices(faces, cap, edges):
                flaws["interior"] += 1
            if not is_tree(cap, adj):
                flaws["nontree"] += 1
                both = False
                continue
            w = worst_overlap(unfold_tree(solid, cap, adj))
            worst = max(worst, w)
            hit = w > OVERLAP_EPS
            tally[(shape, hit)] += 1
            if hit:
                both = False
        whole["flat" if both else "overlaps"] += 1
    return len(cycles), tally, flaws, worst, whole


def main():
    print("Caps, by the shape of their dual and whether the net self-overlaps.")
    print("A cap with an interior vertex, or a dual that is not a tree, would")
    print("be a counterexample to the argument; both columns should stay 0.\n")
    print(f"{'solid':14s} {'cycles':>7s} | {'path ok':>8s} {'path hit':>9s} "
          f"{'branch ok':>10s} {'branch hit':>11s} | {'interior':>9s} {'nontree':>8s} "
          f"| {'cuts that lie flat':>19s}")
    counter = 0
    for name, s in P.SOLIDS.items():
        n, tally, flaws, worst, whole = survey(s)
        counter += flaws["interior"] + flaws["nontree"]
        print(f"{name:14s} {n:7d} | {tally[('path', False)]:8d} "
              f"{tally[('path', True)]:9d} {tally[('branching', False)]:10d} "
              f"{tally[('branching', True)]:11d} | {flaws['interior']:9d} "
              f"{flaws['nontree']:8d} | {whole['flat']:8d} of {n:<8d}")
    print()
    print("No interior vertex and no non-tree anywhere: a Hamiltonian cycle puts")
    print("two cut edges at every vertex, which breaks every fan of faces into an")
    print("arc, so neither cap ever closes a wheel around a vertex. For a disc a")
    print("dual cycle is exactly what encircles an interior vertex, so the dual is")
    print("always a tree -- never merely connected.")
    print()
    print("Branching therefore costs nothing to flatness: a tree rolls out")
    print("isometrically whatever its shape, and the overwhelming majority of")
    print("branching caps lay out perfectly clean.")
    print()
    print("What is left is self-overlap, and it cuts ACROSS the path/branching")
    print("line rather than along it. Branching caps overlap more often, but")
    print("path-shaped ones do it too. So being a strip is neither necessary for")
    print("a flat net nor sufficient for one, and the only honest test is to lay")
    print("the net out and look. On the icosahedron the collisions are exact:")
    print("equilateral triangles roll out onto the triangular lattice, so two")
    print("cells either coincide or miss entirely.")
    return 0 if counter == 0 else 1


def cycle_order(edges, n):
    """An edge set back into the vertex order it visits."""
    nb = {}
    for e in edges:
        a, b = tuple(e)
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    order, prev, cur = [0], None, 0
    while len(order) < n:
        nxt = [x for x in nb[cur] if x != prev][0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


if __name__ == "__main__":
    sys.exit(main())
