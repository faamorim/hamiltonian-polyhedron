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


def face_shape_2d(solid, fi):
    """One face laid flat in its own plane, keeping every length and angle.
    Works for any planar face -- the d10's kites as much as a regular polygon
    -- because it reads the face's real corners rather than reconstructing an
    ideal n-gon."""
    pts = np.array([solid.vertices[i] for i in solid.faces[fi]], float)
    origin = pts[0]
    e1 = pts[1] - origin
    e1 = e1 / np.linalg.norm(e1)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n = n / np.linalg.norm(n)
    e2 = np.cross(n, e1)
    flat = np.array([[(q - origin) @ e1, (q - origin) @ e2] for q in pts])
    return flat if flat[:, 1].sum() >= 0 else flat * np.array([1.0, -1.0])


def _place(shape, ia, ib, pa, pb, away_from):
    """Drop a face's flat shape onto the page so its shared edge lands on an
    edge already placed, on the far side from the face it came off. Two rigid
    placements satisfy the edge; the centroid picks the right one."""
    la, lb = shape[ia], shape[ib]
    u, v = shape[ib] - la, np.asarray(pb) - np.asarray(pa)
    ang = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    c, s_ = math.cos(ang), math.sin(ang)
    rot = np.array([[c, -s_], [s_, c]])
    uh = u / np.linalg.norm(u)
    mirror = 2 * np.outer(uh, uh) - np.eye(2)
    best = None
    for m in (np.eye(2), mirror):
        out = (shape - la) @ m.T @ rot.T + np.asarray(pa)
        d = np.linalg.norm(out.mean(0) - np.asarray(away_from))
        if best is None or d > best[0]:
            best = (d, out)
    return best[1]


def unfold(solid, path, edge_len):
    """Roll the strip out flat, face by face -- the isometric development of
    the real surface. Lengths and angles inside a face are exact; only the
    fold between faces is opened out.

    `edge_len` is the length the solid's reference edge (the first edge of its
    first face) comes out at, so it scales the whole net.

    Returns {face: [(global vertex, (x, y)), ...]} in the face's own cyclic
    order, ready to use as a polygon.
    """
    scale = edge_len / solid.reference_edge()
    shapes = {fi: face_shape_2d(solid, fi) * scale for fi in path}

    first = solid.faces[path[0]]
    face_2d = {path[0]: list(zip(first, [tuple(q) for q in shapes[path[0]]]))}

    for i in range(1, len(path)):
        fprev, fcur = path[i - 1], path[i]
        a, b = shared_verts(solid.faces, fprev, fcur)
        prev = dict(face_2d[fprev])
        prev_centre = np.mean([q for _, q in face_2d[fprev]], axis=0)
        cur = solid.faces[fcur]
        placed = _place(shapes[fcur], cur.index(a), cur.index(b),
                        prev[a], prev[b], prev_centre)
        face_2d[fcur] = list(zip(cur, [tuple(q) for q in placed]))
        for v in (a, b):
            assert math.dist(prev[v], dict(face_2d[fcur])[v]) < 1e-6, \
                f"the seam between faces {fprev} and {fcur} does not join"
    return face_2d


def net_corners(face_2d, path):
    return np.array(sorted({(round(q[0], 6), round(q[1], 6))
                            for fi in path for _, q in face_2d[fi]}))


def net_alignment_deg(face_2d_a, path_a, face_2d_b, path_b):
    """The turn that lands cap B's net on cap A's, found on the flat nets
    (cheap and exact) rather than guessed from a symmetry of the solid. It has
    to be searched for: on a solid with irregular faces the angle is not a
    multiple of anything obvious -- the d10's is 307.885530 degrees."""
    A, B = net_corners(face_2d_a, path_a), net_corners(face_2d_b, path_b)
    if len(A) != len(B):
        return None, np.inf
    A0, B0 = A - A.mean(0), B - B.mean(0)

    def err(deg):
        t = math.radians(deg)
        c, s_ = math.cos(t), math.sin(t)
        r = B0 @ np.array([[c, s_], [-s_, c]])
        return max(np.min(np.linalg.norm(A0 - q, axis=1)) for q in r)

    best = min((err(d), d) for d in np.arange(0, 360, 0.5))
    step = 0.5
    for _ in range(8):
        step /= 8
        best = min((err(d), d) for d in np.arange(best[1] - 4 * step, best[1] + 4 * step, step))
    return best[1], best[0]


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
        for fi, f in enumerate(self.faces):
            pts = np.array([self.vertices[i] for i in f], float)
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            n /= np.linalg.norm(n)
            off = float(np.abs((pts - pts[0]) @ n).max())
            assert off < 1e-9, f"{self.name}: face {fi} is not planar (off by {off:.2e})"

    def reference_edge(self):
        f = self.faces[0]
        return float(np.linalg.norm(np.array(self.vertices[f[0]]) - self.vertices[f[1]]))

    def face_normals(self):
        out = []
        for f in self.faces:
            pts = np.array([self.vertices[i] for i in f], float)
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            n /= np.linalg.norm(n)
            out.append(n if np.dot(n, pts.mean(0) - self.centre()) > 0 else -n)
        return np.array(out)

    def centre(self):
        return np.array(self.vertices, float).mean(0)

    def fold_angles(self):
        """How far each edge has to close, keyed by edge: the angle between the
        two faces' outward normals, which is 180 minus the dihedral angle.
        Read per edge rather than taken as one constant, because a solid need
        not have only one dihedral -- the d10 has two."""
        normals = self.face_normals()
        owners = {}
        for fi, f in enumerate(self.faces):
            for e in face_edges(f):
                owners.setdefault(e, []).append(fi)
        return {e: math.degrees(math.acos(np.clip(normals[o[0]] @ normals[o[1]], -1, 1)))
                for e, o in owners.items() if len(o) == 2}

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

def _orient(vertices, faces):
    """Wind every face counter-clockwise seen from outside, so a face's vertex
    order and its normal agree no matter how the face list was typed in."""
    centre = np.array(vertices, float).mean(0)
    out = []
    for f in faces:
        pts = np.array([vertices[i] for i in f], float)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        out.append(list(f) if np.dot(n, pts.mean(0) - centre) > 0 else list(f)[::-1])
    return out


_TET_VERTS = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
_TET_FACES = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]

_CUBE_VERTS = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
_CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
               [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5]]

_OCT_VERTS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
_OCT_FACES = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
              [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]


def _trapezohedron():
    """The d10. Not Platonic and not regular: its ten faces are kites, with
    two different edge lengths and two different dihedral angles. It is here
    because everything downstream reads each face's real shape and each edge's
    real fold angle rather than assuming one of each."""
    a, radius = 0.15, 1.0
    apex = 9.472135954999583 * a      # makes each kite exactly planar
    def ring(k, z):
        ang = math.radians(k * 72 + (36 if z < 0 else 0))
        return [radius * math.cos(ang), radius * math.sin(ang), z]
    verts = [[0, 0, apex], [0, 0, -apex]]
    verts += [ring(k, a) for k in range(5)] + [ring(k, -a) for k in range(5)]
    upper, lower = (lambda k: 2 + k % 5), (lambda k: 7 + k % 5)
    faces = [[0, upper(k), lower(k), upper(k + 1)] for k in range(5)]
    faces += [[1, lower(k), upper(k + 1), lower(k + 1)] for k in range(5)]
    return verts, faces


_TRAP_VERTS, _TRAP_FACES = _trapezohedron()

TETRAHEDRON = Solid("Tetrahedron", _TET_VERTS, _orient(_TET_VERTS, _TET_FACES),
                    [0, 1, 2, 3])
CUBE = Solid("Cube", _CUBE_VERTS, _orient(_CUBE_VERTS, _CUBE_FACES),
             [0, 1, 2, 3, 7, 6, 5, 4])
OCTAHEDRON = Solid("Octahedron", _OCT_VERTS, _orient(_OCT_VERTS, _OCT_FACES),
                   [0, 2, 1, 4, 3, 5])
TRAPEZOHEDRON = Solid("Pentagonal Trapezohedron", _TRAP_VERTS,
                      _orient(_TRAP_VERTS, _TRAP_FACES),
                      [0, 2, 7, 3, 8, 4, 9, 5, 10, 1, 11, 6])

# ordered by face count, the way the web visualiser lists them
SOLIDS = {
    "tetrahedron": TETRAHEDRON,
    "cube": CUBE,
    "octahedron": OCTAHEDRON,
    "trapezohedron": TRAPEZOHEDRON,
    "dodecahedron": DODECAHEDRON,
    "icosahedron": ICOSAHEDRON,
}
