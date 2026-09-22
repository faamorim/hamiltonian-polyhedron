"""Generates each cap's 10-face strip as ONE flat, foldable print: a thin
continuous base (the "paper," and the living hinge at every fold line) with
a raised, open-topped, hollow frustum on top of each triangle. Neighboring
frustums are inset from the shared edge and don't touch each other, so only
the thin base has to flex when the strip is folded by hand into the
icosahedron's real 3D shape afterward.

Why flat instead of pre-folded: printing flat is about as reliable as FDM
printing gets (zero overhangs anywhere), and the whole strip stays perfectly
aligned since it's one connected print -- no per-edge connectors needed at
all. The tradeoff is the fold itself: rigid filaments like PLA can crack
when bent, so this is explicitly a prototype -- if a fold doesn't survive
being bent to the correct angle, glue/reinforce it there instead.

The two caps are NOT congruent (verified: no icosahedral symmetry -- of all
120, including reflections -- maps cap 0's face set onto cap 1's), so each
gets its own file, not a shared/mirrored one.

Requires: pip install manifold3d numpy

Run from the repo root:
    python3 hardware/scripts/gen_flat_strip.py
"""

import math
import manifold3d as m3d

PHI = (1 + 5 ** 0.5) / 2

VERTICES = [
    [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
    [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
    [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
]
FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
CYCLE = [0, 11, 5, 1, 7, 6, 3, 8, 9, 4, 2, 10]

TARGET_EDGE = 22.0      # mm, prototype face edge length
BASE_THICKNESS = 0.8    # mm, thin continuous "paper" layer -- the living hinge
WALL_HEIGHT = 3.0       # mm, raised rim height above the base
CLEARANCE = 0.4         # mm, minimum gap kept between neighboring walls at every
                        # height, even at the target fold angle -- keeps them
                        # mechanically independent (only the thin base connects
                        # them) and leaves room for print tolerance
FOLD_ANGLE_DEG = 41.81031489577859  # exact: 180 - icosahedron's dihedral angle
TOP_MARGIN = 1.0        # mm, extra inset on FREE (non-fold) edges at the top,
                        # purely cosmetic -- tapers the rim like the fold edges do
OVERLAP = 0.3           # mm, the frustum's base is embedded this far INTO the base
                        # layer (not just touching it) -- touching-but-not-overlapping
                        # solids are a known ambiguous case for mesh boolean union

# A CONSTANT inset margin (independent of height) leaves a wedge-shaped gap
# that does NOT close when folded: at height h above the base, two opposing
# points only meet if each is inset by h*tan(fold_angle/2) -- exactly a
# mitered edge, the same trick as a picture frame's mitered corner. Using a
# constant margin instead (this file's first attempt) left a persistent
# ~1-3mm gap at the target fold angle even at the top of a 3mm wall.
FOLD_MITER = math.tan(math.radians(FOLD_ANGLE_DEG) / 2)


def edge_key(a, b):
    return (a, b) if a < b else (b, a)


def internal_adjacency(faces, cycle):
    cycle_edges = {edge_key(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
    face_edge_map = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            key = edge_key(f[k], f[(k + 1) % 3])
            face_edge_map.setdefault(key, []).append(fi)
    adj = {i: [] for i in range(len(faces))}
    for key, lst in face_edge_map.items():
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
        stack = [s]
        comp_of[s] = comp
        while stack:
            f = stack.pop()
            for nb in adj[f]:
                if comp_of[nb] == -1:
                    comp_of[nb] = comp
                    stack.append(nb)
        comp += 1
    return [[i for i in range(len(faces)) if comp_of[i] == c] for c in range(comp)]


def walk_path(cap_faces, adj):
    """A cap's internal adjacency is always a simple path (verified: 9 edges
    on 10 faces, both extremities degree 1, no branching) -- walk it end to end."""
    ends = [f for f in cap_faces if len(adj[f]) == 1]
    start = ends[0]
    path = [start]
    prev, cur = None, start
    while True:
        nxt = [n for n in adj[cur] if n != prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        path.append(cur)
    return path


def shared_verts(fa, fb):
    return [v for v in FACES[fa] if v in FACES[fb]]


def unfold(path, edge_len):
    """Lay the path of equilateral triangles out flat, edge to edge, each new
    triangle placed on the side away from its predecessor's own third vertex
    (standard net-unfolding)."""
    h = edge_len * math.sqrt(3) / 2
    face_2d = {}
    f0 = FACES[path[0]]
    face_2d[path[0]] = list(zip(f0, [(0.0, 0.0), (edge_len, 0.0), (edge_len / 2, h)]))

    def pt_of(fi, gv):
        for g, p in face_2d[fi]:
            if g == gv:
                return p

    for i in range(1, len(path)):
        fprev, fcur = path[i - 1], path[i]
        a, b = shared_verts(fprev, fcur)
        pa, pb = pt_of(fprev, a), pt_of(fprev, b)
        third = [v for v in FACES[fcur] if v not in (a, b)][0]
        prev_third = [v for v in FACES[fprev] if v not in (a, b)][0]
        prev_third_pt = pt_of(fprev, prev_third)

        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        d = math.hypot(dx, dy)
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        hgt = math.sqrt(max(edge_len * edge_len - (d / 2) ** 2, 0))
        ux, uy = -dy / d, dx / d
        c1 = (mx + ux * hgt, my + uy * hgt)
        c2 = (mx - ux * hgt, my - uy * hgt)
        d1 = math.hypot(c1[0] - prev_third_pt[0], c1[1] - prev_third_pt[1])
        d2 = math.hypot(c2[0] - prev_third_pt[0], c2[1] - prev_third_pt[1])
        new_pt = c1 if d1 > d2 else c2

        face_2d[fcur] = [(a, pa), (b, pb), (third, new_pt)]
    return face_2d


def compute_fold_edges(path):
    """Global vertex-pair keys for every internal (fold) edge in this path."""
    fold_edges = set()
    for i in range(len(path) - 1):
        fa, fb = path[i], path[i + 1]
        a, b = shared_verts(fa, fb)
        fold_edges.add(edge_key(a, b))
    return fold_edges


def offset_triangle_per_edge(pts2d, margins):
    """Move each of the triangle's 3 edges inward (toward the centroid side)
    by its own margin, then re-intersect consecutive edges for the new
    vertices. Unlike a uniform centroid-scale, this allows a DIFFERENT inset
    per edge -- needed because fold edges must taper with height (a miter)
    while free edges don't."""
    cx = sum(p[0] for p in pts2d) / 3
    cy = sum(p[1] for p in pts2d) / 3
    lines = []
    for k in range(3):
        p1, p2 = pts2d[k], pts2d[(k + 1) % 3]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        midx, midy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if nx * (cx - midx) + ny * (cy - midy) < 0:
            nx, ny = -nx, -ny
        offset_p1 = (p1[0] + nx * margins[k], p1[1] + ny * margins[k])
        lines.append((offset_p1, (dx, dy)))

    def line_intersect(p1, d1, p2, d2):
        a11, a12 = d1[0], -d2[0]
        a21, a22 = d1[1], -d2[1]
        b1, b2 = p2[0] - p1[0], p2[1] - p1[1]
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-9:
            return p1
        t = (b1 * a22 - a12 * b2) / det
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])

    return [line_intersect(*lines[(k - 1) % 3], *lines[k]) for k in range(3)]


def raised_frustum(pts2d, global_verts, fold_edges):
    """A raised, solid, tapered rim for one face. Fold-facing sides are
    mitered at half the icosahedron's fold angle, so when the strip is bent
    to the true dihedral angle, neighboring walls close up to just
    CLEARANCE apart at every height -- not just at the base."""
    is_fold = [edge_key(global_verts[k], global_verts[(k + 1) % 3]) in fold_edges for k in range(3)]

    bottom_margins = [CLEARANCE] * 3  # at h=0 the miter term vanishes regardless
    top_margins = [
        CLEARANCE + (WALL_HEIGHT * FOLD_MITER if is_fold[k] else TOP_MARGIN)
        for k in range(3)
    ]

    base_ring = offset_triangle_per_edge(pts2d, bottom_margins)
    top_ring = offset_triangle_per_edge(pts2d, top_margins)
    verts = ([(x, y, BASE_THICKNESS - OVERLAP) for x, y in base_ring]
             + [(x, y, BASE_THICKNESS + WALL_HEIGHT) for x, y in top_ring])
    return m3d.Manifold.hull_points(verts)


def ccw(pts):
    s = sum(pts[i][0] * pts[(i + 1) % 3][1] - pts[(i + 1) % 3][0] * pts[i][1] for i in range(3))
    return pts if s > 0 else list(reversed(pts))


def build_strip(path, edge_len=TARGET_EDGE):
    face_2d = unfold(path, edge_len)
    fold_edges = compute_fold_edges(path)

    # CrossSection needs consistent CCW winding per contour (a CW contour is
    # treated as a hole) -- the zigzag unfolding naturally alternates
    # triangle orientation, so normalize each one before unioning.
    contours = [ccw([p for _, p in face_2d[fi]]) for fi in path]
    base = m3d.Manifold.extrude(m3d.CrossSection(contours), BASE_THICKNESS)

    frustums = [
        raised_frustum([p for _, p in face_2d[fi]], [gv for gv, _ in face_2d[fi]], fold_edges)
        for fi in path
    ]
    solid = m3d.Manifold.batch_boolean([base] + frustums, m3d.OpType.Add)
    return solid, face_2d


def export_stl(manifold_obj, path_out):
    mesh = manifold_obj.to_mesh()
    verts = mesh.vert_properties
    tris = mesh.tri_verts
    with open(path_out, "w") as f:
        f.write("solid part\n")
        for tri in tris:
            v0, v1, v2 = (verts[i][:3] for i in tri)
            ux, uy, uz = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            vx, vy, vz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            norm = (nx ** 2 + ny ** 2 + nz ** 2) ** 0.5 or 1.0
            f.write(f"  facet normal {nx/norm:.6f} {ny/norm:.6f} {nz/norm:.6f}\n")
            f.write("    outer loop\n")
            for v in (v0, v1, v2):
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid part\n")


if __name__ == "__main__":
    adj = internal_adjacency(FACES, CYCLE)
    caps = split_into_caps(FACES, adj)
    paths = [walk_path(cap, adj) for cap in caps]

    for i, path in enumerate(paths):
        solid, _ = build_strip(path)
        n_parts = len(solid.decompose())
        assert n_parts == 1, f"cap {i} strip is split into {n_parts} disconnected shells"
        print(f"cap{i}: path={path}  volume={solid.volume():.1f}mm3  bbox={solid.bounding_box()}")
        export_stl(solid, f"hardware/tests/flat_strip_cap{i}.stl")
        print(f"  wrote hardware/tests/flat_strip_cap{i}.stl")
