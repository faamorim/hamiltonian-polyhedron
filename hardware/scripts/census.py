"""How many essentially different ways is there to cut each solid?

Four numbers per solid, all computed here rather than written down:

  cycles       every Hamiltonian cycle, as an undirected edge set
  forms        those, counted up to the solid's own symmetry -- which for a
               3-connected planar graph is exactly its graph automorphism
               group, by Whitney, so rotations AND reflections
  strips       the cycles whose two caps each have PATH-shaped face adjacency,
               which is what the printed part is: a linear ladder of faces on
               living hinges. NOT a condition for lying flat -- see below
  stripForms   those, again up to symmetry

and whether the cut the project ships has congruent halves, which is what
lets one STL be printed twice.

A correction worth keeping, because this file used to assert the opposite:
a path of faces rolls out flat and "a branching tree does not" is FALSE.
Every cap of every Hamiltonian cycle rolls out flat, branching or not. A
cycle puts two cut edges at every vertex, which breaks the fan of faces
around it into an arc, so no cap ever closes a wheel and no cap has an
interior vertex -- and for a disc, a dual cycle is exactly what encircles
one, so the dual is always a tree rather than merely connected. A tree
unfolds isometrically whatever its shape.

What can still go wrong is the net colliding with itself somewhere far from
where it branched, and that cuts across the path/branching line rather than
along it: on the icosahedron, 1985 branching caps lay out clean against 215
that do not, and 338 path caps against 22 that do not. Being a strip is
neither necessary for a flat net nor sufficient for one. hardware/scripts/
check_unfold.py lays every net out and measures it; this file only counts
shapes.

The counts agree with the literature where it has an opinion: 30 Hamiltonian
cycles on the dodecahedron (Hamilton's own icosian game) and 2560 directed,
which is 1280 undirected, on the icosahedron.

Seconds to run for all six. The icosahedron dominates.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polyhedra as P


def adjacency(faces, n):
    adj = [set() for _ in range(n)]
    for f in faces:
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            adj[a].add(b)
            adj[b].add(a)
    return [sorted(x) for x in adj]


def automorphisms(adj):
    """Every vertex permutation preserving adjacency, by backtracking.

    Degree and partial-adjacency pruning make this fast at these sizes; the
    dodecahedron's 20 vertices of degree 3 are the worst case and finish in
    well under a second.
    """
    n = len(adj)
    deg = [len(a) for a in adj]
    adjset = [set(a) for a in adj]
    out, mapping = [], [-1] * n

    def place(v):
        if v == n:
            out.append(tuple(mapping))
            return
        used = set(mapping[:v])
        for c in range(n):
            if c in used or deg[c] != deg[v]:
                continue
            if all((u in adjset[v]) == (mapping[u] in adjset[c]) for u in range(v)):
                mapping[v] = c
                place(v + 1)
                mapping[v] = -1

    place(0)
    return out


def hamiltonian_cycles(adj):
    """Every Hamiltonian cycle as a frozenset of undirected edges.

    Vertex 0 lies on every Hamiltonian cycle, so the search starts there; and
    only neighbours above the start are extended, which drops each cycle's
    mirror image rather than finding it twice.
    """
    n = len(adj)
    seen, out = set(), []

    def walk(path, used):
        v = path[-1]
        if len(path) == n:
            if path[0] in adj[v]:
                edges = frozenset(frozenset((path[i], path[(i + 1) % n]))
                                  for i in range(n))
                if edges not in seen:
                    seen.add(edges)
                    out.append(edges)
            return
        for w in adj[v]:
            if w not in used and w > path[0]:
                used.add(w)
                path.append(w)
                walk(path, used)
                path.pop()
                used.remove(w)

    walk([0], {0})
    return out


def caps(faces, edges):
    """The two halves, as sets of face indices, with their face adjacency."""
    fadj = {i: set() for i in range(len(faces))}
    seen = {}
    for fi, f in enumerate(faces):
        for k in range(len(f)):
            e = frozenset((f[k], f[(k + 1) % len(f)]))
            if e in seen:
                if e not in edges:        # not cut: the two faces stay joined
                    fadj[fi].add(seen[e])
                    fadj[seen[e]].add(fi)
            else:
                seen[e] = fi
    unseen, parts = set(fadj), []
    while unseen:
        x = unseen.pop()
        comp, stack = {x}, [x]
        while stack:
            y = stack.pop()
            for z in fadj[y]:
                if z not in comp:
                    comp.add(z)
                    unseen.discard(z)
                    stack.append(z)
        parts.append(comp)
    return fadj, parts


def is_strip(faces, edges):
    fadj, parts = caps(faces, edges)
    if len(parts) != 2:
        return False
    for comp in parts:
        degs = [len(fadj[x] & comp) for x in comp]
        if max(degs) > 2 or degs.count(1) != 2:
            return False
    return True


def congruent_halves(faces, edges, aut):
    """Is there a symmetry of the solid carrying one cap onto the other?"""
    _, parts = caps(faces, edges)
    if len(parts) != 2 or len(parts[0]) != len(parts[1]):
        return False
    by_verts = {frozenset(f): i for i, f in enumerate(faces)}
    a, b = parts
    for g in aut:
        img, ok = set(), True
        for fi in a:
            fj = by_verts.get(frozenset(g[v] for v in faces[fi]))
            if fj is None:
                ok = False
                break
            img.add(fj)
        if ok and img == set(b):
            return True
    return False


def count_forms(items, aut):
    """How many orbits the edge sets fall into under the symmetry group."""
    index = {e: i for i, e in enumerate(items)}
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, edges in enumerate(items):
        for g in aut:
            j = index.get(frozenset(frozenset((g[a], g[b]))
                                    for a, b in (tuple(e) for e in edges)))
            if j is not None:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(len(items))})


def census(solid):
    faces = [list(f) for f in solid.faces]
    adj = adjacency(faces, len(solid.vertices))
    aut = automorphisms(adj)
    cycles = hamiltonian_cycles(adj)
    strips = [e for e in cycles if is_strip(faces, e)]
    ours = frozenset(frozenset((solid.cycle[i],
                                solid.cycle[(i + 1) % len(solid.cycle)]))
                     for i in range(len(solid.cycle)))
    assert ours in cycles, f"{solid.name}: the shipped cycle is not a cycle"
    assert ours in strips, f"{solid.name}: the shipped cut does not fold as strips"
    return {
        "symmetries": len(aut),
        "cycles": len(cycles),
        "forms": count_forms(cycles, aut),
        "strips": len(strips),
        "stripForms": count_forms(strips, aut),
        "congruent": congruent_halves(faces, ours, aut),
    }


if __name__ == "__main__":
    print(f"{'solid':14s} {'sym':>5s} {'cycles':>7s} {'forms':>6s} "
          f"{'strips':>7s} {'forms':>6s}  halves")
    for name, s in P.SOLIDS.items():
        c = census(s)
        print(f"{name:14s} {c['symmetries']:5d} {c['cycles']:7d} {c['forms']:6d} "
              f"{c['strips']:7d} {c['stripForms']:6d}  "
              f"{'congruent' if c['congruent'] else 'not congruent'}")
