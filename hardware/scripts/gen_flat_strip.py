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
    gap at the base that showed up on the first print. Now hinge_gap sets
    the bare strip and CONTACT_CLEARANCE sets the contact face, and the
    wall takes whichever of the two is WIDER at each height, so it returns to
    the true miter plane as soon as the hinge no longer needs the room -- and
    so that no part of the section can ever reach the mirror plane before the
    wall top does.

  * BASE_THICKNESS set both the flexibility of the hinge AND the stiffness
    of the face skin. Thin enough to fold meant floppy faces, and the
    floppiness got worse the bigger the model. Now FACE_THICKNESS carries
    the faces and the base is milled down to HINGE_THICKNESS only in the
    bare strip along each fold.

The miter is measured from the real pivot -- the middle of the THINNED
hinge, not the top of the base -- since that is the line the sheet actually
bends about.

SCALE. The net outline is the only thing that scales with the model;
everything else is absolute millimetres, because the hinge is a local
feature whose behaviour depends on the printer and the filament, not on how
big the solid is. So the fold behaves identically at any size -- and so a
finished STL must never be scaled in a slicer, which would scale the hinge
along with it and stop it bending. Ask for another size here instead; the
script re-runs every check at the size it was given.

Size is asked for as PERCEIVED size (mean width -- the average caliper
reading over all orientations), not as an edge length, so that one number
means the same physical object across the six solids. The paper nets are
already sized this way. An edge length is not comparable: a dodecahedron of
22mm edges is nearly three times a tetrahedron of the same.

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

import sys

from polyhedra import (SOLIDS, unfold, compute_fold_edges, shared_verts, edge_key,
                       net_alignment_deg)

SOLID = SOLIDS[(sys.argv[1] if len(sys.argv) > 1 else "icosahedron").lower()]

# --- scale ----------------------------------------------------------------
# The rungs offered on the web page. A ladder rather than a free number
# because each size is a separate print that has to pass its own checks, and
# the window is not wide: below the small end the wall insets eat the face,
# above the large end the strip runs off a 180mm bed. These five sit inside
# the window for all six solids at once, so the same rung exists everywhere.
SIZES_MM = [25.0, 32.0, 40.0, 50.0, 63.0]      # perceived size (mean width)
DEFAULT_SIZE_MM = 40.0

TARGET_SIZE = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIZE_MM
TARGET_EDGE = TARGET_SIZE / SOLID.mean_width()  # mm; scales the net, nothing else
BED_MM = 180.0              # printer bed, checked against the finished strip

# --- base: stiffness and flexibility, separately --------------------------
FACE_THICKNESS = 1.2        # mm, skin under a face
HINGE_THICKNESS = 0.6       # mm, skin in the bare strip along a fold
# The bare strip has to be WIDER on a sharper fold. It curls to a radius of
# about (its width) / (the angle), so the outer fibre is stretched by roughly
# thickness x angle / (2 x width). Hold the width fixed and the tetrahedron's
# 109 degree fold strains the filament two and a half times as hard as the
# icosahedron's 42 degree one -- which is the difference between a hinge that
# bends and one that cracks, from a number that was never about either solid.
# Making the width proportional to the angle asks the same of every fold.
#
# The constant is calibrated on the fold that has actually been printed: the
# icosahedron's, 0.80mm at 41.81 degrees, which bends easily without going
# floppy. So that solid is unchanged and the others are brought into line
# with it.
HINGE_GAP_PER_RAD = 0.80 / math.radians(41.81)   # mm of bare strip per radian
MIN_HINGE_GAP = 0.6         # mm floor -- below a nozzle width it is a crease, not a hinge

# --- walls: bending relief and contact face, separately -------------------
WALL_HEIGHT = 3.0           # mm, rim height above the face
CONTACT_CLEARANCE = 0.08    # mm, how far the contact face sits off the true miter.
                            # Not zero: a wall printing proud would stop the fold
                            # BEFORE the target angle, which is a hard stop you
                            # cannot push past -- worse than a small gap.

FREE_EDGE_MARGIN = 0.4      # mm, inset of the wall on silhouette (non-fold) edges
TOP_MARGIN = 1.0            # mm, extra inset there at the top -- cosmetic taper
OVERLAP = 0.3               # mm, frustum foot embedded INTO the base (touching-but-
                            # not-overlapping solids are ambiguous for mesh booleans)

# Each edge closes through its own angle -- 180 minus that edge's dihedral --
# read off the solid rather than fixed, because a solid need not have only one:
# the d10's kites meet at 59.45 degrees along one class of edge and 73.30 along
# the other, and a single constant would mis-cut half its seams.
FOLD_ANGLES = SOLID.fold_angles()

PIVOT_Z = HINGE_THICKNESS / 2        # the sheet bends about the middle of the thinned band
# The wall starts at the HINGE band, not at the top of the face. Starting it
# higher leaves the stretch between them with nothing but the trench wall,
# which is cut to the trench's full width, so the boundary sat proud there and
# then stepped inward when the wall finally began -- an undercut, 0.02mm on the
# icosahedron and 0.29mm on the cube. Beginning at the hinge means the profile
# is wall_margin(z) the whole way up, with no step to print over.
FOOT_Z = HINGE_THICKNESS - OVERLAP
TOP_Z = FACE_THICKNESS + WALL_HEIGHT

assert 0 < HINGE_THICKNESS <= FACE_THICKNESS, "hinge cannot be thicker than the face"


def miter_of(edge):
    return math.tan(math.radians(FOLD_ANGLES[edge]) / 2)


def hinge_gap(miter):
    """Width of the bare strip left free to flex at a fold, set by the angle
    that fold has to close through -- which the miter already carries, being
    the tangent of its half. Taking it from the miter rather than passing it
    separately keeps every caller's signature as it was: everything that
    shapes the section is a function of the fold angle and the height."""
    return max(MIN_HINGE_GAP, HINGE_GAP_PER_RAD * 2 * math.atan(miter))


def wall_margin(z, miter):
    """How far the material sits from its fold line at height z above the outer
    skin -- for the wall and for the trench cut into the base alike.

    It is the true miter plane through the pivot, offset by CONTACT_CLEARANCE,
    except low down where that plane would run inside the bare strip the base
    needs in order to flex; there the bare strip wins.

    Taking the LARGER of the two rather than ramping between them is what keeps
    the fold honest. The boundary is then never inside the miter plane at any
    height, and since a point at (d, z) reaches the mirror plane at half-angle
    atan(d / (z - pivot)), which falls as z rises along the miter, the wall top
    is guaranteed to be the first thing to touch. That is the contact face, by
    design. Anything that dips inside the miter plane lower down would touch
    sooner and jam the fold short of the angle the solid needs.
    """
    return max(hinge_gap(miter) / 2, CONTACT_CLEARANCE + max(0.0, z - PIVOT_Z) * miter)


def relief_knee(miter):
    """Height at which the bare strip stops being the wider of the two and the
    wall rejoins the miter plane. Only a reporting/ring-placement convenience;
    nothing depends on it being a parameter any more."""
    z = PIVOT_Z + (hinge_gap(miter) / 2 - CONTACT_CLEARANCE) / miter
    return min(max(z, FOOT_Z), TOP_Z)





# Nothing anywhere on the section may reach the mirror plane before the wall
# top does, or the fold jams short of the angle the solid needs. Check it for
# every fold angle this solid uses, across the whole height.
for _a in set(FOLD_ANGLES.values()):
    _m = math.tan(math.radians(_a) / 2)
    _psi = lambda z: math.atan2(wall_margin(z, _m), z - PIVOT_Z)
    _top = _psi(TOP_Z)
    for _z in [HINGE_THICKNESS + 1e-9] + [HINGE_THICKNESS + k * (TOP_Z - HINGE_THICKNESS) / 400
                                          for k in range(401)]:
        assert _psi(_z) >= _top - 1e-12, (
            f"at a {_a:.2f} deg fold, the section at z={_z:.3f}mm would touch at "
            f"{2 * math.degrees(_psi(_z)):.2f} deg, before the wall top's "
            f"{2 * math.degrees(_top):.2f} deg -- the fold would jam early")


def free_margin(z):
    frac = (z - FOOT_Z) / (TOP_Z - FOOT_Z)
    return FREE_EDGE_MARGIN + frac * TOP_MARGIN


def margins_at(z, miters):
    return [free_margin(z) if m is None else wall_margin(z, m) for m in miters]


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
    miters = []
    for k in range(n):
        e = edge_key(global_verts[k], global_verts[(k + 1) % n])
        miters.append(miter_of(e) if e in fold_edges else None)
    knees = sorted({relief_knee(m) for m in miters if m is not None} or {FOOT_Z})
    verts = []
    for z in [FOOT_Z] + knees + [TOP_Z]:
        ring = offset_polygon_per_edge(pts2d, margins_at(z, miters))
        assert inradius(ring) > 0.2, (
            f"the wall insets have eaten this face at TARGET_EDGE={TARGET_EDGE}mm "
            f"(top inradius {inradius(ring):.2f}mm) -- the model is too small")
        verts += [(x, y, z) for x, y in ring]
    return m3d.Manifold.hull_points(verts)


def ccw(pts):
    n = len(pts)
    s = sum(pts[k][0] * pts[(k + 1) % n][1] - pts[(k + 1) % n][0] * pts[k][1] for k in range(n))
    return pts if s > 0 else list(reversed(pts))


TRENCH_OVERSHOOT = 0.05     # mm, see hinge_trench


def hinge_trench(pa, pb, miter):
    """The cut that thins the base along one fold line. Its width spans exactly
    the bare strip between the two wall feet, so it never undercuts a wall.

    Along the fold it runs slightly PAST both ends. A fold edge of a strip has
    both its endpoints on the net's outline, so a trench cut to exactly the
    edge's length ends flush with the boundary -- a degenerate coincidence that
    floating point resolves differently in the two caps, leaving one of them an
    extra end wall of one hinge gap by FACE_THICKNESS. That was 0.96mm2 at the
    time, which is exactly the surface-area difference it produced, and it was
    enough to make two congruent caps fail the one-part check.
    """
    ax, ay = pa
    bx, by = pb
    length = math.hypot(bx - ax, by - ay) + 2 * TRENCH_OVERSHOOT
    depth = FACE_THICKNESS - HINGE_THICKNESS
    if depth <= 0:
        return None
    # sized at the TOP of the trench, not at the wall's foot: the miter plane
    # keeps moving outward with height, so a trench cut to the foot's width
    # leaves the base's top edge poking inside it
    box = m3d.Manifold.cube([length, 2 * wall_margin(FACE_THICKNESS, miter), depth + 1.0], True)
    box = box.rotate([0, 0, math.degrees(math.atan2(by - ay, bx - ax))])
    return box.translate([(ax + bx) / 2, (ay + by) / 2,
                          HINGE_THICKNESS + (depth + 1.0) / 2])


def build_strip(path, edge_len=TARGET_EDGE):
    face_2d = unfold(SOLID, path, edge_len)
    fold_edges = compute_fold_edges(SOLID.faces, path)

    # CrossSection needs consistent CCW winding per contour (a CW contour reads
    # as a hole) -- the zigzag unfolding alternates orientation face to face.
    contours = [ccw([p for _, p in face_2d[fi]]) for fi in path]
    base = m3d.Manifold.extrude(m3d.CrossSection(contours), FACE_THICKNESS)

    trenches = []
    for i in range(len(path) - 1):
        a, b = shared_verts(SOLID.faces, path[i], path[i + 1])
        pts = dict(face_2d[path[i]])
        t = hinge_trench(pts[a], pts[b], miter_of(edge_key(a, b)))
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


def stop_angle_deg(miter):
    """The fold angle at which the two contact faces actually meet. The wall
    top is the leading edge once the fold passes the target, so contact is
    where it reaches the mirror plane."""
    lever = TOP_Z - PIVOT_Z
    return 2 * math.degrees(math.atan((CONTACT_CLEARANCE + lever * miter) / lever))


def congruence_error(a, b, turn_deg):
    """Volume of b that fails to coincide with a after that turn about the
    build plate. Zero means the two caps are literally the same object."""
    mid = lambda m: ((m.bounding_box()[0] + m.bounding_box()[3]) / 2,
                     (m.bounding_box()[1] + m.bounding_box()[4]) / 2)
    ax, ay = mid(a)
    bx, by = mid(b)
    r = b.translate([-bx, -by, 0]).rotate([0, 0, turn_deg])
    rx, ry = mid(r)
    r = r.translate([ax - rx, ay - ry, 0])
    return (a - r).volume() + (r - a).volume()


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
    i = min(len(path) // 2, len(path) - 2)     # a 2-face strip has only one seam
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
    print(f"{SOLID.name}: two strips of {len(paths[0])} faces "
          f"({len(SOLID.faces[paths[0][0]])} sides each)")
    print(f"perceived size {TARGET_SIZE:.1f}mm wants edge {TARGET_EDGE:.2f}mm "
          f"(do NOT rescale the STL afterwards -- it would rescale the hinge)")
    for angle in sorted({round(a, 6) for a in FOLD_ANGLES.values()}):
        m = math.tan(math.radians(angle) / 2)
        print(f"  a {angle:6.2f} deg fold: contact faces meet at {stop_angle_deg(m):6.2f} deg "
              f"({stop_angle_deg(m) - angle:+.2f}), bare strip {hinge_gap(m):.2f}mm wide in a "
              f"{2 * wall_margin(FACE_THICKNESS, m):.2f}mm trench, wall rejoins the miter at "
              f"z={relief_knee(m):.2f}mm")
    print(f"base {FACE_THICKNESS:.2f}mm under the faces, {HINGE_THICKNESS:.2f}mm at the folds")

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
    turn, net_err = net_alignment_deg(solids[0][1], solids[0][2],
                                      solids[1][1], solids[1][2])
    # Judged against the edge length, not in bare millimetres. The residual is
    # round-off accumulated along the unfold chain, so it grows with the model
    # -- about 4e-7mm per mm of edge on the d10 -- and an absolute threshold is
    # a different demand at every size: the same net passed at 22mm edges and
    # failed at 24mm. A real mismatch would be a whole feature, four orders of
    # magnitude above this.
    assert turn is not None and net_err < 1e-5 * TARGET_EDGE, (
        f"the two nets are not congruent (best fit off by {net_err:.6f}mm, "
        f"{net_err / TARGET_EDGE:.1e} of an edge) -- this solid needs two "
        f"different prints")
    diff = congruence_error(solids[0][0], solids[1][0], turn)
    # Read the leftover volume as an average surface displacement, which is
    # both physically meaningful and independent of how big the part is: a
    # real difference between the caps would be a whole feature, microns
    # thick at the very least, while the floor here is the precision of the
    # net alignment the turn was measured from.
    skin = diff / solids[0][0].surface_area()
    print(f"a {turn:.4f} deg turn (found on the nets to "
          f"{net_err / TARGET_EDGE:.1e} of an edge) leaves "
          f"{diff:.6f}mm3 unmatched -- {skin * 1e6:.3f} nanometres of surface, "
          f"{'ONE part, printed twice' if skin < 1e-4 else 'NOT congruent'}")
    assert skin < 1e-4, (
        f"the two caps differ by {skin:.2e}mm of surface -- too much to be "
        f"arithmetic, so they are genuinely not the same part")

    solid, face_2d, path = solids[0]
    i = min(len(path) // 2, len(path) - 2)
    seam = edge_key(*shared_verts(SOLID.faces, path[i], path[i + 1]))
    knee = relief_knee(miter_of(seam))
    worst = max(abs(m - wall_margin(z, miter_of(seam))) for z, m in
                measure_wall_profile(solid, face_2d, path) if z > knee + 0.2)
    print(f"wall face measured off the finished mesh matches the design to {worst:.4f}mm")
    assert worst < 0.05, "the printed wall is not where the design says it is"

    name = SOLID.name.split()[-1].lower()
    out = f"hardware/tests/flat_strip_{name}_{TARGET_SIZE:g}mm.stl"
    export_stl(solid, out)
    print(f"\nwrote {out}  (print TWO of these)")
