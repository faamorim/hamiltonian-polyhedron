"""Generates a cap's 10-face strip as ONE flat, foldable print: a continuous
base (the "paper", and the living hinge at every fold line) carrying a
raised, open-topped frustum on each triangle. The strip is folded by hand
afterwards into the icosahedron's real 3D shape.

Why flat: printing flat is about as reliable as FDM gets (no overhangs
anywhere) and the whole strip stays aligned because it is one connected
print. The tradeoff is the fold itself -- rigid filaments like PLA can crack
when bent, so this is explicitly a prototype; if a fold does not survive
being bent to the correct angle, glue or reinforce it there.

ONE FILE, PRINTED TWICE. The two caps are not related by any symmetry of the
icosahedron, but their flat NETS are congruent -- a 120 degree rotation maps
one onto the other exactly, because unfolding keeps only the local turn
pattern and forgets which global vertices are involved. Since this part is
printed flat, that makes the two caps the same physical object, and the
script checks it rather than assuming it.

TWO DECOUPLINGS. Earlier versions used one number for two unrelated jobs
each, and were necessarily bad at both:

  * CLEARANCE set both the bare strip of base left free to flex at a fold
    AND where the wall's contact face sat. Pulling the wall back far enough
    to bend also parked it a constant distance short of its neighbour, so
    the walls never met at the true fold angle -- the strip only stopped
    when the wall TOPS jammed, about 13 degrees past the target, leaving the
    gap at the base that showed up on the first print. Now HINGE_GAP sets
    the bare strip and CONTACT_CLEARANCE sets the contact face, and the
    pull-back ramps away over RELIEF_HEIGHT so the wall returns to the true
    miter plane for the rest of its height.

  * BASE_THICKNESS set both the flexibility of the hinge AND the stiffness
    of the face skin. Thin enough to fold meant floppy faces, and the
    floppiness got worse the bigger the model. Now FACE_THICKNESS carries
    the faces and the base is milled down to HINGE_THICKNESS only in the
    bare strip along each fold.

The miter is measured from the real pivot -- the middle of the THINNED
hinge, not the top of the base -- since that is the line the sheet actually
bends about.

SCALE. TARGET_EDGE is the only dimension that scales the model; everything
else is absolute millimetres, because the hinge is a local feature whose
behaviour depends on the printer and filament, not on how big the
icosahedron is. So the fold behaves identically at any size, and the script
asserts the chosen size still leaves a sane frustum top and fits the bed.

NOT YET IMPLEMENTED, deliberately -- both wanted, neither urgent:
  * recesses in the frustum floors for magnets at the seam;
  * a shallow inset on the end triangle of each strip to take a contrasting
    insert, so a black strip carries a white triangle and vice versa (the
    yin-yang reading of the two interlocking caps).

Requires: pip install manifold3d numpy

Run from the repo root:
    python3 hardware/scripts/gen_flat_strip.py
"""

import math

import manifold3d as m3d
import numpy as np

from polyhedra import ICOSAHEDRON, unfold, compute_fold_edges, shared_verts, edge_key

SOLID = ICOSAHEDRON

# --- scale ----------------------------------------------------------------
TARGET_EDGE = 22.0          # mm; the ONLY dimension that scales with the model
BED_MM = 180.0              # printer bed, checked against the finished strip

# --- base: stiffness and flexibility, separately --------------------------
FACE_THICKNESS = 1.2        # mm, skin under a face
HINGE_THICKNESS = 0.6       # mm, skin in the bare strip along a fold
HINGE_GAP = 0.8             # mm, width of that bare strip (wall foot to wall foot)

# --- walls: bending relief and contact face, separately -------------------
WALL_HEIGHT = 3.0           # mm, rim height above the face
RELIEF_HEIGHT = 1.2         # mm, height over which the foot's pull-back ramps away
CONTACT_CLEARANCE = 0.08    # mm, how far the contact face sits off the true miter.
                            # Not zero: a wall printing proud would stop the fold
                            # BEFORE the target angle, which is a hard stop you
                            # cannot push past -- worse than a small gap.

FREE_EDGE_MARGIN = 0.4      # mm, inset of the wall on silhouette (non-fold) edges
TOP_MARGIN = 1.0            # mm, extra inset there at the top -- cosmetic taper
OVERLAP = 0.3               # mm, frustum foot embedded INTO the base (touching-but-
                            # not-overlapping solids are ambiguous for mesh booleans)

FOLD_ANGLE_DEG = 41.81031489577859   # exact: 180 - the icosahedron's dihedral angle
FOLD_MITER = math.tan(math.radians(FOLD_ANGLE_DEG) / 2)

PIVOT_Z = HINGE_THICKNESS / 2        # the sheet bends about the middle of the thinned band
FOOT_Z = FACE_THICKNESS - OVERLAP
TOP_Z = FACE_THICKNESS + WALL_HEIGHT
RELIEF_Z = FACE_THICKNESS + RELIEF_HEIGHT

assert 0 < HINGE_THICKNESS <= FACE_THICKNESS, "hinge cannot be thicker than the face"
assert 0 < RELIEF_HEIGHT < WALL_HEIGHT, "the relief must end below the top of the wall"



def fold_margin(z):
    """How far the wall face sits from its fold line, at height z above the
    outer skin. Above the relief it is the true miter plane through the pivot,
    offset by CONTACT_CLEARANCE; below, it is pulled back by enough to leave
    HINGE_GAP of bare base, ramping away linearly over RELIEF_HEIGHT."""
    miter = max(0.0, z - PIVOT_Z) * FOLD_MITER
    pullback = HINGE_GAP / 2 - CONTACT_CLEARANCE - max(0.0, FOOT_Z - PIVOT_Z) * FOLD_MITER
    assert pullback >= 0, (
        f"HINGE_GAP of {HINGE_GAP}mm is narrower than the miter already demands at the "
        f"wall foot; raise HINGE_GAP or lower FACE_THICKNESS")
    ramp = max(0.0, 1.0 - (z - FOOT_Z) / RELIEF_HEIGHT)
    return CONTACT_CLEARANCE + miter + pullback * ramp


# The wall's foot must not overhang the thinned band, or it would be left
# standing on air where the trench cuts under it. By construction it lands
# exactly on the trench wall; check it rather than trust it.
assert abs(fold_margin(FOOT_Z) - HINGE_GAP / 2) < 1e-9, (
    f"the wall foot sits at {fold_margin(FOOT_Z):.4f}mm but the thinned band "
    f"reaches {HINGE_GAP / 2:.4f}mm -- the foot would overhang the trench")


def free_margin(z):
    frac = (z - FOOT_Z) / (TOP_Z - FOOT_Z)
    return FREE_EDGE_MARGIN + frac * TOP_MARGIN


def margins_at(z, is_fold):
    return [fold_margin(z) if f else free_margin(z) for f in is_fold]


def offset_polygon_per_edge(pts2d, margins):
    """Move each edge inward by its OWN margin and re-intersect consecutive
    edges. A uniform centroid-scale cannot do this, and the whole design needs
    it: fold edges taper with height (the miter) while free edges do not."""
    n = len(pts2d)
    cx = sum(p[0] for p in pts2d) / n
    cy = sum(p[1] for p in pts2d) / n
    lines = []
    for k in range(n):
        p1, p2 = pts2d[k], pts2d[(k + 1) % n]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        midx, midy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        if nx * (cx - midx) + ny * (cy - midy) < 0:
            nx, ny = -nx, -ny
        lines.append(((p1[0] + nx * margins[k], p1[1] + ny * margins[k]), (dx, dy)))

    def intersect(p1, d1, p2, d2):
        det = d1[0] * -d2[1] - -d2[0] * d1[1]
        if abs(det) < 1e-9:
            return p1
        b1, b2 = p2[0] - p1[0], p2[1] - p1[1]
        t = (b1 * -d2[1] - -d2[0] * b2) / det
        return (p1[0] + t * d1[0], p1[1] + t * d1[1])

    return [intersect(*lines[(k - 1) % n], *lines[k]) for k in range(n)]


def inradius(pts2d):
    a = np.array(pts2d)
    n = len(a)
    area = 0.5 * abs(sum(a[k][0] * a[(k + 1) % n][1] - a[(k + 1) % n][0] * a[k][1]
                         for k in range(n)))
    per = sum(np.linalg.norm(a[k] - a[(k + 1) % n]) for k in range(n))
    return 2 * area / per


def raised_frustum(pts2d, global_verts, fold_edges):
    """One face's rim. Three rings, not two: the wall's foot is pulled back to
    clear the hinge, then returns to the miter plane, and that needs a profile
    with a knee in it. The margin is convex in z (the slope only increases),
    so the convex hull of the three rings is exactly the intended solid."""
    n = len(pts2d)
    is_fold = [edge_key(global_verts[k], global_verts[(k + 1) % n]) in fold_edges
               for k in range(n)]
    verts = []
    for z in (FOOT_Z, RELIEF_Z, TOP_Z):
        ring = offset_polygon_per_edge(pts2d, margins_at(z, is_fold))
        assert inradius(ring) > 0.2, (
            f"the wall insets have eaten this face at TARGET_EDGE={TARGET_EDGE}mm "
            f"(top inradius {inradius(ring):.2f}mm) -- the model is too small")
        verts += [(x, y, z) for x, y in ring]
    return m3d.Manifold.hull_points(verts)


def ccw(pts):
    n = len(pts)
    s = sum(pts[k][0] * pts[(k + 1) % n][1] - pts[(k + 1) % n][0] * pts[k][1] for k in range(n))
    return pts if s > 0 else list(reversed(pts))


def hinge_trench(pa, pb):
    """The cut that thins the base along one fold line. It spans exactly the
    bare strip between the two wall feet, so it never undercuts a wall."""
    ax, ay = pa
    bx, by = pb
    length = math.hypot(bx - ax, by - ay)
    depth = FACE_THICKNESS - HINGE_THICKNESS
    if depth <= 0:
        return None
    box = m3d.Manifold.cube([length, HINGE_GAP, depth + 1.0], True)
    box = box.rotate([0, 0, math.degrees(math.atan2(by - ay, bx - ax))])
    return box.translate([(ax + bx) / 2, (ay + by) / 2,
                          HINGE_THICKNESS + (depth + 1.0) / 2])


def build_strip(path, edge_len=TARGET_EDGE):
    face_2d = unfold(SOLID.faces, path, edge_len)
    fold_edges = compute_fold_edges(SOLID.faces, path)

    # CrossSection needs consistent CCW winding per contour (a CW contour reads
    # as a hole) -- the zigzag unfolding alternates orientation face to face.
    contours = [ccw([p for _, p in face_2d[fi]]) for fi in path]
    base = m3d.Manifold.extrude(m3d.CrossSection(contours), FACE_THICKNESS)

    trenches = []
    for i in range(len(path) - 1):
        a, b = shared_verts(SOLID.faces, path[i], path[i + 1])
        pts = dict(face_2d[path[i]])
        t = hinge_trench(pts[a], pts[b])
        if t is not None:
            trenches.append(t)
    if trenches:
        base -= m3d.Manifold.batch_boolean(trenches, m3d.OpType.Add)

    frustums = [
        raised_frustum([p for _, p in face_2d[fi]], [gv for gv, _ in face_2d[fi]], fold_edges)
        for fi in path
    ]
    return m3d.Manifold.batch_boolean([base] + frustums, m3d.OpType.Add), face_2d


def export_stl(manifold_obj, path_out):
    mesh = manifold_obj.to_mesh()
    verts, tris = mesh.vert_properties, mesh.tri_verts
    with open(path_out, "w") as f:
        f.write("solid part\n")
        for tri in tris:
            v0, v1, v2 = (verts[i][:3] for i in tri)
            u = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            w = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx, ny, nz = (u[1]*w[2] - u[2]*w[1], u[2]*w[0] - u[0]*w[2], u[0]*w[1] - u[1]*w[0])
            norm = (nx*nx + ny*ny + nz*nz) ** 0.5 or 1.0
            f.write(f"  facet normal {nx/norm:.6f} {ny/norm:.6f} {nz/norm:.6f}\n")
            f.write("    outer loop\n")
            for v in (v0, v1, v2):
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid part\n")


def stop_angle_deg():
    """The fold angle at which the two contact faces actually meet. The wall
    top is the leading edge once the fold passes the target, so contact is
    where it reaches the mirror plane."""
    lever = TOP_Z - PIVOT_Z
    return 2 * math.degrees(math.atan((CONTACT_CLEARANCE + lever * FOLD_MITER) / lever))


def net_turn(face_2d, path):
    """Which way this net's first step points -- the reference for lining one
    cap's net up with the other's."""
    centre = lambda fi: np.mean([p for _, p in face_2d[fi]], axis=0)
    d = centre(path[1]) - centre(path[0])
    return math.degrees(math.atan2(d[1], d[0]))


def congruence_error(a, b, turn_deg):
    """Volume of b that fails to coincide with a after the best turn about the
    build plate, trying the net's turn and the icosahedron's 3-fold multiples
    of it. Zero means the two caps are literally the same object."""
    mid = lambda m: ((m.bounding_box()[0] + m.bounding_box()[3]) / 2,
                     (m.bounding_box()[1] + m.bounding_box()[4]) / 2)
    ax, ay = mid(a)
    bx, by = mid(b)
    best = None
    for deg in (turn_deg, turn_deg + 120, turn_deg + 240):
        r = b.translate([-bx, -by, 0]).rotate([0, 0, deg])
        rx, ry = mid(r)
        r = r.translate([ax - rx, ay - ry, 0])
        err = (a - r).volume() + (r - a).volume()
        if best is None or err < best[0]:
            best = (err, deg % 360)
    return best


def ray_hits_segment(origin, direction, a, b):
    """Where a ray leaving `origin` along `direction` first crosses segment ab,
    as a distance along the ray, or None. Measuring by ray rather than by
    polygon vertex matters here: the wall face runs parallel to its fold line,
    so its only vertices are at the far corners of the triangle, nowhere near
    the point being measured."""
    e = np.asarray(b, float) - np.asarray(a, float)
    det = direction[0] * -e[1] - -e[0] * direction[1]
    if abs(det) < 1e-12:
        return None
    r = np.asarray(a, float) - origin
    t = (r[0] * -e[1] - -e[0] * r[1]) / det
    sseg = (direction[0] * r[1] - direction[1] * r[0]) / det
    return t if t > 1e-9 and -1e-9 <= sseg <= 1 + 1e-9 else None


def measure_wall_profile(solid, face_2d, path, samples=25):
    """Read the wall face back off the FINISHED mesh, so the CSG is checked
    against the design rather than trusted. Slices the solid at a series of
    heights and measures how far the material starts from the fold line --
    sampling the surface itself, not just the few heights that happen to
    carry mesh vertices."""
    i = len(path) // 2
    va, vb = shared_verts(SOLID.faces, path[i], path[i + 1])
    pts = dict(face_2d[path[i]])
    pa, pb = np.array(pts[va]), np.array(pts[vb])
    mid, span = (pa + pb) / 2, np.linalg.norm(pb - pa)
    u = (pb - pa) / span
    nrm = np.array([-u[1], u[0]])
    # which side of the fold line this face lies on
    face_centre = np.mean([q for _, q in face_2d[path[i]]], axis=0)
    if np.dot(nrm, face_centre - mid) < 0:
        nrm = -nrm

    out = []
    for z in np.linspace(FACE_THICKNESS + 0.05, TOP_Z - 0.05, samples):
        hits = []
        for poly in solid.slice(float(z)).to_polygons():
            q = np.asarray(poly)
            for k in range(len(q)):
                t = ray_hits_segment(mid, nrm, q[k], q[(k + 1) % len(q)])
                if t is not None:
                    hits.append(t)
        if hits:
            out.append((z, min(hits)))
    return out


if __name__ == "__main__":
    paths = SOLID.strips()
    print(f"fold angle wanted {FOLD_ANGLE_DEG:.2f} deg; the contact faces meet at "
          f"{stop_angle_deg():.2f} deg ({stop_angle_deg() - FOLD_ANGLE_DEG:+.2f})")
    print(f"bare base strip at a fold: {HINGE_GAP:.2f}mm wide, {HINGE_THICKNESS:.2f}mm "
          f"thick, under {FACE_THICKNESS:.2f}mm faces")

    solids = []
    for i, path in enumerate(paths):
        solid, face_2d = build_strip(path)
        n = len(solid.decompose())
        assert n == 1, f"cap {i} is split into {n} disconnected shells"
        bb = solid.bounding_box()
        w, h = bb[3] - bb[0], bb[4] - bb[1]
        assert max(w, h) <= BED_MM, (
            f"strip is {w:.0f}x{h:.0f}mm and will not fit a {BED_MM:.0f}mm bed "
            f"at TARGET_EDGE={TARGET_EDGE}mm")
        print(f"cap{i}: volume {solid.volume():8.1f}mm3   footprint {w:.1f} x {h:.1f} x "
              f"{bb[5] - bb[2]:.1f}mm")
        solids.append((solid, face_2d, path))

    # The design is only worth one file if the two caps really are the same
    # part. Comparing vertex lists does not answer that -- the two meshes are
    # tessellated differently (205 vs 201 vertices) even though the solids are
    # identical. Ask the boolean kernel instead: turn one onto the other and
    # measure the volume that fails to overlap.
    print(f"\nboth caps: volume {solids[0][0].volume():.6f} / "
          f"{solids[1][0].volume():.6f}, area {solids[0][0].surface_area():.6f} / "
          f"{solids[1][0].surface_area():.6f}")
    diff, turn = congruence_error(solids[0][0], solids[1][0], net_turn(*solids[0][1:]) -
                                  net_turn(*solids[1][1:]))
    print(f"a {turn:.2f} deg turn leaves {diff:.6f}mm3 of the {solids[0][0].volume():.0f}mm3 "
          f"solid unmatched -- {'ONE part, printed twice' if diff < 1e-3 else 'NOT congruent'}")
    assert diff < 1e-3, "the two caps are not the same part after all"

    solid, face_2d, path = solids[0]
    worst = max(abs(m - fold_margin(z)) for z, m in
                measure_wall_profile(solid, face_2d, path) if z > RELIEF_Z + 0.2)
    print(f"wall face measured off the finished mesh matches the design to {worst:.4f}mm")
    assert worst < 0.05, "the printed wall is not where the design says it is"

    export_stl(solid, "hardware/tests/flat_strip.stl")
    print("\nwrote hardware/tests/flat_strip.stl  (print TWO of these)")
