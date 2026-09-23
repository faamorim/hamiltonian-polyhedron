"""Shared polyhedron data and net machinery for the flat-strip templates.

A Hamiltonian cycle on a solid's vertex graph cuts its surface into two caps.
Because EVERY vertex lies on the cut, no vertex is left in a cap's interior,
so each cap's face-adjacency graph is a tree -- and for both solids here it
is a simple path, i.e. a strip that unfolds flat edge to edge.

Everything below works for any regular-faced solid, triangles or pentagons
alike; only the two SOLID definitions at the bottom are specific.

The icosahedron's data is the same as hardware/scripts/gen_flat_strip.py, the
3D print generator, which keeps its own copy because its frustum/miter
geometry is icosahedron-specific -- this module is the 2D/graph half only.
The dodecahedron is not typed in by hand at all: it is built as the DUAL of
that icosahedron (one vertex per icosahedron face, one pentagon per
icosahedron vertex), so it inherits data that is already verified rather
than adding twenty fresh coordinates to get wrong.
"""

import math
import numpy as np


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def face_edges(face):
    return [edge_key(face[k], face[(k + 1) % len(face)]) for k in range(len(face))]


def internal_adjacency(faces, cycle):
    """Face-to-face adjacency WITHIN a cap: shared edges that the Hamiltonian
    cycle does not cut. Faces in different caps are never adjacent here."""
    cycle_edges = {edge_key(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
    owners = {}
    for fi, f in enumerate(faces):
        for key in face_edges(f):
            owners.setdefault(key, []).append(fi)
    adj = {i: [] for i in range(len(faces))}
    for key, lst in owners.items():
        if len(lst) == 2 and key not in cycle_edges:
            adj[lst[0]].append(lst[1])
            adj[lst[1]].append(lst[0])
    return adj


def split_into_caps(faces, adj):
    comp_of = [-1] * len(faces)
    comp = 0
    for s in range(len(faces)):
        if comp_of[s] != -1:
            continue
        stack, comp_of[s] = [s], comp
        while stack:
            f = stack.pop()
            for nb in adj[f]:
                if comp_of[nb] == -1:
                    comp_of[nb] = comp
                    stack.append(nb)
        comp += 1
    return [[i for i in range(len(faces)) if comp_of[i] == c] for c in range(comp)]


def walk_path(cap_faces, adj):
    """Walk a cap end to end. Asserts the cap really is a strip rather than a
    branched tree -- which is what makes it unfoldable into one flat band."""
    degrees = sorted(len(adj[f]) for f in cap_faces)
    assert degrees == [1, 1] + [2] * (len(cap_faces) - 2), \
        f"cap is not a simple strip; face degrees were {degrees}"
    start = next(f for f in cap_faces if len(adj[f]) == 1)
    path, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            return path
        prev, cur = cur, nxt[0]
        path.append(cur)


def shared_verts(faces, fa, fb):
    return [v for v in faces[fa] if v in faces[fb]]


def unfold(faces, path, edge_len):
    """Roll the strip out flat, face by face. Each face is a regular polygon
    laid on the far side of the edge it shares with its predecessor, which is
    the isometric development of the real surface -- lengths and angles within
    a face are exact, only the fold between faces is opened out.

    Returns {face_index: [(global_vertex, (x, y)), ...]} in the face's own
    cyclic vertex order, so the points can be used as a polygon directly.
    """
    n = len(faces[path[0]])
    assert all(len(faces[f]) == n for f in path), "mixed face sizes in one strip"
    circumradius = edge_len / (2 * math.sin(math.pi / n))
    apothem = edge_len / (2 * math.tan(math.pi / n))

    def ngon_from_edge(face, a, b, pa, pb, away_from):
        """Place `face`'s regular polygon so vertex a sits at pa and b at pb,
        on the side of that edge away from the point `away_from`."""
        # Neighbouring faces of a closed surface run their shared edge in
        # OPPOSITE directions, so a->b may be backwards in this face's cyclic
        # order. Anchor on whichever end makes it forwards.
        if face[(face.index(a) + 1) % n] != b:
            a, b, pa, pb = b, a, pb, pa
        assert face[(face.index(a) + 1) % n] == b, f"{a}-{b} is not an edge of {face}"
        pa, pb = np.array(pa), np.array(pb)
        mid = (pa + pb) / 2
        d = (pb - pa) / np.linalg.norm(pb - pa)
        normal = np.array([-d[1], d[0]])
        if np.dot(normal, mid - np.array(away_from)) < 0:
            normal = -normal
        centre = mid + normal * apothem
        v0 = pa - centre
        # the turn that carries vertex a to vertex b tells us which way round
        # this face's cyclic order runs in the plane
        step = math.atan2(float(np.cross(v0, pb - centre)), float(np.dot(v0, pb - centre)))
        assert abs(abs(step) - 2 * math.pi / n) < 1e-9, \
            f"corner turn is {math.degrees(step):.3f} deg, not {360 / n:.3f}"
        j = face.index(a)
        placed = []
        for k in range(n):
            ang = k * step
            c, s = math.cos(ang), math.sin(ang)
            p = centre + np.array([c * v0[0] - s * v0[1], s * v0[0] + c * v0[1]])
            placed.append((face[(j + k) % n], (float(p[0]), float(p[1]))))
        assert abs(np.linalg.norm(v0) - circumradius) < 1e-9, "edge length mismatch"
        return placed

    f0 = faces[path[0]]
    seed_away = (0.0, -1.0)   # puts the first face on the +y side of its base edge
    face_2d = {path[0]: ngon_from_edge(f0, f0[0], f0[1], (0.0, 0.0), (edge_len, 0.0), seed_away)}

    for i in range(1, len(path)):
        fprev, fcur = path[i - 1], path[i]
        a, b = shared_verts(faces, fprev, fcur)
        prev_pts = dict(face_2d[fprev])
        prev_centre = np.mean([p for _, p in face_2d[fprev]], axis=0)
        face_2d[fcur] = ngon_from_edge(faces[fcur], a, b, prev_pts[a], prev_pts[b], prev_centre)
    return face_2d


def compute_fold_edges(faces, path):
    return {edge_key(*shared_verts(faces, path[i], path[i + 1])) for i in range(len(path) - 1)}


class Solid:
    def __init__(self, name, vertices, faces, cycle):
        self.name, self.vertices, self.faces, self.cycle = name, vertices, faces, cycle
        self.verify()

    def verify(self):
        edges = {e for f in self.faces for e in face_edges(f)}
        v, e, f = len(self.vertices), len(edges), len(self.faces)
        assert v - e + f == 2, f"{self.name}: Euler characteristic is {v - e + f}, not 2"
        assert sorted(self.cycle) == list(range(v)), \
            f"{self.name}: the cycle does not visit every vertex exactly once"
        for i in range(v):
            a, b = self.cycle[i], self.cycle[(i + 1) % v]
            assert edge_key(a, b) in edges, f"{self.name}: {a}-{b} is not an edge"
        lengths = {round(float(np.linalg.norm(np.array(self.vertices[a]) - self.vertices[b])), 6)
                   for a, b in edges}
        assert len(lengths) == 1, f"{self.name}: faces are not regular, edge lengths {lengths}"

    def strips(self):
        """Both caps, each walked end to end as a strip of faces."""
        adj = internal_adjacency(self.faces, self.cycle)
        caps = split_into_caps(self.faces, adj)
        assert len(caps) == 2, f"{self.name}: expected 2 caps, got {len(caps)}"
        assert len(caps[0]) == len(caps[1]), \
            f"{self.name}: caps are uneven ({len(caps[0])}/{len(caps[1])})"
        return [walk_path(cap, adj) for cap in caps]


PHI = (1 + 5 ** 0.5) / 2

_ICO_VERTS = [
    [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
    [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
    [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
]
_ICO_FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]


def _dual_of(vertices, faces):
    """The dual solid: a vertex at each face's centre, and a face around each
    original vertex made of the faces meeting there, walked in cyclic order."""
    dual_verts = [list(np.mean([vertices[i] for i in f], axis=0)) for f in faces]
    scale = 1.0 / np.linalg.norm(dual_verts[0])
    dual_verts = [[c * scale for c in p] for p in dual_verts]

    dual_faces = []
    for v in range(len(vertices)):
        ring = [i for i, f in enumerate(faces) if v in f]
        order = [ring.pop(0)]
        while ring:
            cur = set(faces[order[-1]])
            nxt = next(i for i in ring if len(cur & set(faces[i])) == 2)
            ring.remove(nxt)
            order.append(nxt)
        pts = np.array([dual_verts[i] for i in order])
        if np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), pts.mean(axis=0)) < 0:
            order.reverse()
        dual_faces.append(order)
    return dual_verts, dual_faces


_DOD_VERTS, _DOD_FACES = _dual_of(_ICO_VERTS, _ICO_FACES)

ICOSAHEDRON = Solid(
    "Icosahedron", _ICO_VERTS, _ICO_FACES,
    [0, 11, 5, 1, 7, 6, 3, 8, 9, 4, 2, 10],
)

# Chosen from the 30 Hamiltonian cycles of the dodecahedron; every one of them
# happens to split it into two 6-pentagon strips, so this is picked for how
# well the two nets nest on the page rather than for its structure.
DODECAHEDRON = Solid(
    "Dodecahedron", _DOD_VERTS, _DOD_FACES,
    [0, 6, 15, 5, 1, 2, 3, 8, 17, 12, 13, 18, 9, 19, 14, 10, 11, 16, 7, 4],
)

SOLIDS = {"icosahedron": ICOSAHEDRON, "dodecahedron": DODECAHEDRON}
