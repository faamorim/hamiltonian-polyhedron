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
                       net_alignment_deg, net_alignment_turns, socket_edges)

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
MIN_WALL_HEIGHT = 3.0       # mm, the rim with nothing to accommodate
WALL_HEIGHT = MIN_WALL_HEIGHT   # raised below, where a socket needs more
CONTACT_CLEARANCE = 0.08    # mm, how far the contact face sits off the true miter.
                            # Not zero: a wall printing proud would stop the fold
                            # BEFORE the target angle, which is a hard stop you
                            # cannot push past -- worse than a small gap.

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
def assert_no_early_jam():
    for a in {round(x, 6) for x in FOLD_ANGLES.values()}:
        m = math.tan(math.radians(a) / 2)
        psi = lambda z: math.atan2(wall_margin(z, m), z - PIVOT_Z)
        top = psi(TOP_Z)
        for z in [HINGE_THICKNESS + 1e-9] + [
                HINGE_THICKNESS + k * (TOP_Z - HINGE_THICKNESS) / 400 for k in range(401)]:
            assert psi(z) >= top - 1e-12, (
                f"at a {a:.2f} deg fold, the section at z={z:.3f}mm would touch at "
                f"{2 * math.degrees(psi(z)):.2f} deg, before the wall top's "
                f"{2 * math.degrees(top):.2f} deg -- the fold would jam early")


assert_no_early_jam()


def seam_margin(z, miter):
    """The same miter, for an edge of the net that is NOT a fold: one of the
    Hamiltonian cycle's edges, where this cap meets the OTHER one.

    Every free edge of a cap's net is a seam edge -- that is the theorem the
    whole project rests on, cut edges = V -- so this is the mating face of the
    finished object, and it has to be the bisector of the two faces that meet
    there, exactly as a fold does. The one difference is where the miter is
    measured from. A fold pivots about the middle of its hinge because the
    sheet bends there; a seam does not bend at all. The two caps are rigid
    bodies brought together, touching first at the outer skin, so the bisector
    passes through the edge on the OUTER SURFACE and the miter is measured
    from z = 0.

    This was a plain inward taper before -- 0.4mm at the foot to 1.4mm at the
    top -- a number with no relation to the angle the caps meet at. It left
    the cube's two caps overlapping by 2.7mm of solid at the rim, and the
    base, extruded straight out to the paper outline below it, standing proud
    of the rim by up to 1.2 x miter. So the caps met base-ledge to base-ledge
    and the rims never touched at all.
    """
    return CONTACT_CLEARANCE + z * miter


def margins_at(z, edges):
    """`edges` is one (is_fold, miter) per side of the face, in order."""
    return [wall_margin(z, m) if fold else seam_margin(z, m) for fold, m in edges]


def offset_polygon_per_edge(pts2d, margins):
    """Move each edge inward by its OWN margin and re-intersect consecutive
    edges. A uniform centroid-scale cannot do this, and the whole design needs
    it: every edge tapers with height, but about its own angle and its own
    pivot -- a fold about the middle of its hinge, a seam about the outer
    skin."""
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
    edges = []
    for k in range(n):
        e = edge_key(global_verts[k], global_verts[(k + 1) % n])
        edges.append((e in fold_edges, miter_of(e)))
    knees = sorted({relief_knee(m) for fold, m in edges if fold} or {FOOT_Z})
    verts = []
    for z in [FOOT_Z] + knees + [TOP_Z]:
        ring = offset_polygon_per_edge(pts2d, margins_at(z, edges))
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


# --- magnet sockets at the seam -------------------------------------------
# A 4x2mm magnet stands with its poles across the seam and drops in edgewise
# from the INSIDE face, down a channel in its own plane, behind a thin wall.
# Attraction then presses it INTO that wall -- magnet, wall, body of the face,
# all in compression -- instead of straight out of a hole with only friction
# across it, which is what an earlier version did and why it let go. Glue is
# welcome but never in the load path, and the lid never carries anything.
#
# SOCKET_WALL is the only number here chosen by feel rather than forced: it
# falls out of the fit entirely, because thickening it lowers the top and the
# bottom of the pocket by the same amount. Two of them, one per cap, set the
# gap between the magnets and so the pull.
MAGNET_DIA, MAGNET_H = 4.0, 2.0
MAGNET_FIT = 0.2            # mm added to the pocket so the magnet drops in
SOCKET_WALL = 0.8           # mm between the magnet and the mating face
SOCKET_SKIN = 0.8           # mm of plastic under the pocket, over the outer surface
SOCKET_PROUD = 0.4          # mm every cutter runs past the surface it cuts
LID_T = 0.8                 # mm, the lid and the rebate it sits flush in
LID_GAP = 0.15              # mm clearance round the lid
LID_MARGIN = 1.2            # mm of rebate around the channel's mouth
REBATE_WALL = 0.8           # mm the rebate must leave to the mating face
SOCKET_MARGIN = 1.3         # mm of face around the socket, before a fold

SOCKET_MIN_GAP = 0.5        # mm a socket must keep clear of a fold's miter plane
_WANTED_KEYS, _WANTED_AT = socket_edges(SOLID)


def seam_frame(miter):
    """Square into the material from the mating face, and up that face. The
    magnet's axis is the first; it drops in along the second."""
    sec = math.hypot(1.0, miter)
    return (np.array([0.0, 1.0 / sec, -miter / sec]),
            np.array([0.0, miter / sec, 1.0 / sec]))


def on_seam(t, miter):
    """The point t millimetres up the mating face from the seam edge."""
    return np.array([0.0, CONTACT_CLEARANCE, 0.0]) + t * seam_frame(miter)[1]


def socket_seat(miter):
    """Where the magnet sits, measured up the mating face: far enough that
    the pocket's lowest corner -- the far bottom one, since the pocket leans
    back with the miter -- still has skin over the outer surface."""
    r = (MAGNET_DIA + MAGNET_FIT) / 2
    return (r + (SOCKET_WALL + MAGNET_H + MAGNET_FIT) * miter
            + SOCKET_SKIN * math.hypot(1.0, miter))


def socket_top(miter):
    """How tall the rim has to be for a socket to fit in it.

    The magnet stands MAGNET_DIA up the mating face, and the face is only
    TOP_Z x sec wide, so the rim has to carry it -- plus the lid's rebate on
    top. This was a local pad on the four or six faces that needed one, which
    saved filament and left the rest of the rim alone, but it also left lumps
    standing up on the inside of a finished cap. The inside is seen: pulling
    the two caps apart is the whole point of the object. So the rim goes up
    to this everywhere and the inside comes out flat.

    Nothing below the face moves, so the hinge is untouched; the fold's
    contact face only gets a longer lever, which brings it slightly CLOSER
    to the angle it is aiming for."""
    sec = math.hypot(1.0, miter)
    r = (MAGNET_DIA + MAGNET_FIT) / 2
    return ((socket_seat(miter) + r) / sec - SOCKET_WALL * miter / sec
            + LID_T + 0.2)


def socket_cutter(miter):
    """The magnet's pocket and the channel it slides down, as one swept
    solid: a cylinder dragged up the mating face until it leaves the rim.
    Sweeping a convex body along a line is the hull of its two ends."""
    n, e2 = seam_frame(miter)
    r, h = (MAGNET_DIA + MAGNET_FIT) / 2, MAGNET_H + MAGNET_FIT
    cyl = m3d.Manifold.cylinder(h, r, r, 96)
    cyl = cyl.rotate([math.degrees(math.atan2(-1.0, -miter)), 0, 0])
    lo = cyl.translate(list(on_seam(socket_seat(miter), miter) + SOCKET_WALL * n))
    run = (socket_top(miter) + SOCKET_PROUD) * math.hypot(1.0, miter)
    return m3d.Manifold.batch_hull([lo, lo.translate(list(run * e2))])


_FOOTPRINT = {}


def socket_footprint(miter):
    """The room a socket takes on its face: the channel's own footprint up to
    the top of the rim, plus a wall all round. Measured off the cutter rather
    than chosen -- an earlier version used two fixed numbers, 9 x 7mm, which
    on the icosahedron came within 0.09mm of a fold's miter plane."""
    if miter not in _FOOTPRINT:
        top = socket_top(miter)
        clip = m3d.Manifold.cube([1e3, 1e3, top], True).translate([0, 0, top / 2])
        b = (socket_cutter(miter) ^ clip).bounding_box()
        _FOOTPRINT[miter] = (2 * (max(-b[0], b[3]) + SOCKET_MARGIN), b[4] + SOCKET_MARGIN)
    return _FOOTPRINT[miter]


def socket_parts(miter):
    """The pocket, the rebate, and the lid that fills it -- in a frame with
    the seam edge on the x axis and the face on the +y side."""
    top = socket_top(miter)
    cut = socket_cutter(miter)
    mouth = m3d.CrossSection(cut.slice(top - LID_T).to_polygons())
    rebate2d = mouth.offset(LID_MARGIN, m3d.JoinType.Round, 2.0, 96)

    # Held off the seam by the real plane, in 3D. The rebate is the mouth
    # grown by LID_MARGIN, which is wider than the wall the socket leaves to
    # the mating face, so left alone it opens straight through it -- 4.37mm
    # off the bisector on the icosahedron, on the very face the two caps
    # close on. Clipping it against a 2D line taken at the rebate's foot is
    # not enough either: the plane leans inward as it rises, by miter per mm,
    # so on the tetrahedron's 109 degree joint the top of the rebate came
    # out 5.48mm through. Subtract the plane itself, offset by the wall it
    # has to leave, and the clearance is right at every height.
    sec = math.hypot(1.0, miter)
    plane = seam_cut((-1e3, 0.0), (1e3, 0.0), miter, up_to=top + 2.0)
    seat = top - LID_T
    rebate = m3d.Manifold.extrude(rebate2d, LID_T + SOCKET_PROUD).translate([0, 0, seat])
    rebate -= plane.translate([0, REBATE_WALL * sec, 0])
    lid = m3d.Manifold.extrude(
        rebate2d.offset(-LID_GAP, m3d.JoinType.Round, 2.0, 96), LID_T).translate([0, 0, seat])
    lid -= plane.translate([0, (REBATE_WALL + LID_GAP) * sec, 0])
    return cut, rebate, lid.translate([0, 0, -seat])


def at_edge(part, pa, pb):
    """Swing a part built on the x axis onto a real seam edge of the net."""
    return part.rotate([0, 0, math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))]) \
               .translate([(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, 0])


def socket_clearance(face_2d, path, fold_edges, keys):
    """How close a socket comes to the miter plane of a fold.

    The fold's wall leans further in the higher it goes, and a socket sits
    at the top of the rim, so this is where a face runs out of room. If the
    socket crossed that plane it would open into the fold's contact face.
    Returns the worst margin over every socket and every fold it shares a
    face with; it must stay positive."""
    worst = math.inf if keys else 0.0
    for fi in path:
        face = SOLID.faces[fi]
        pts = dict(face_2d[fi])
        centre = np.mean([q for _, q in face_2d[fi]], axis=0)
        seam = [(edge_key(face[k], face[(k + 1) % len(face)]), k)
                for k in range(len(face))]
        for e, k in seam:
            if e in fold_edges or e not in keys:
                continue
            pa, pb = np.array(pts[face[k]]), np.array(pts[face[(k + 1) % len(face)]])
            u = (pb - pa) / np.linalg.norm(pb - pa)
            nrm = np.array([-u[1], u[0]])
            if nrm @ (centre - pa) < 0:
                u, nrm = -u, -nrm
            mid = (pa + pb) / 2
            w, d = socket_footprint(miter_of(e))
            corners = [mid + su * (w / 2) * u + sv * nrm
                       for su in (-1, 1) for sv in (0.0, d)]
            for e2, k2 in seam:
                if e2 not in fold_edges:
                    continue
                qa = np.array(pts[face[k2]])
                qb = np.array(pts[face[(k2 + 1) % len(face)]])
                v = (qb - qa) / np.linalg.norm(qb - qa)
                fn = np.array([-v[1], v[0]])
                if fn @ (centre - qa) < 0:
                    fn = -fn
                need = wall_margin(socket_top(miter_of(e)), miter_of(e2))
                worst = min(worst, min(fn @ (c - qa) for c in corners) - need)
    return worst


# A pad is absolute millimetres, like the hinge, but the face it stands on
# scales with the model -- so whether a socket fits is a question about THIS
# size, not about the solid. Ask it once, here, rather than discovering it in
# the middle of a boolean: at the small end of the ladder the dodecahedron's
# pentagons and the d10's kites have no room for one.
def _fit_check():
    path = SOLID.strips()[0]
    f2 = unfold(SOLID, path, TARGET_EDGE)
    return socket_clearance(f2, path, compute_fold_edges(SOLID.faces, path), _WANTED_KEYS)


SOCKET_GAP = _fit_check()
SOCKET_FITS = SOCKET_GAP >= SOCKET_MIN_GAP
SOCKET_KEYS = _WANTED_KEYS if SOCKET_FITS else set()
SOCKET_AT = _WANTED_AT if SOCKET_FITS else []

# The rim is as tall as the sockets need and no taller, so the inside of a
# finished cap is one flat surface. Where the sockets do not fit it stays at
# the minimum. Rebound here rather than written as a constant because it is
# a consequence of the magnet, not a choice -- and the fold invariant is
# re-checked against the height that actually results.
WALL_HEIGHT = max([MIN_WALL_HEIGHT] + [socket_top(miter_of(e)) - FACE_THICKNESS
                                       for e in SOCKET_KEYS])
TOP_Z = FACE_THICKNESS + WALL_HEIGHT
assert_no_early_jam()


def seam_cut(pa, pb, miter, up_to=None):
    """The wedge to take off the base along one seam edge.

    The rim above is shaped ring by ring, but the base under it is one flat
    extrusion of the whole net outline, so at a seam it runs straight out to
    the paper edge and stands proud of the rim by up to FACE_THICKNESS x
    miter. That ledge is what met the other cap instead of the mating face.

    So cut the base back to the same bisector plane. A half-space does it:
    rotating a box about the edge by -atan(miter) lays its face on the plane
    y = CONTACT_CLEARANCE + z x miter, and everything outside is removed. The
    box is no deeper than the wedge it has to reach (the plane's distance to
    the far corner of the material) so it cannot reach across the net to some
    other part of a strip that bends back on itself, and it runs past both
    ends of the edge by that same depth so the two planes meeting at a corner
    of the solid leave no sliver between them.
    """
    ax, ay = pa
    bx, by = pb
    # How far the wedge to remove reaches, and so how deep the box must be:
    # measured to the TALLEST thing this plane has to cut through. That is
    # the rim for a strip, but anything standing on the rim -- a pad for a
    # magnet socket, say -- is taller, and a box sized for the rim leaves a
    # sliver of it standing proud, detached from the part.
    up_to = TOP_Z if up_to is None else up_to
    depth = (CONTACT_CLEARANCE + up_to * miter) / math.hypot(1.0, miter) + 0.5
    length = math.hypot(bx - ax, by - ay) + 2 * depth
    tall = 4 * up_to + 8
    box = m3d.Manifold.cube([length, depth, tall], True)
    box = box.translate([0, -depth / 2, 0])          # its y=0 face is the plane
    box = box.rotate([-math.degrees(math.atan(miter)), 0, 0])
    box = box.translate([0, CONTACT_CLEARANCE, 0])

    # the face's own side of the edge is +y before this turn, so point the cut
    # the other way if the edge runs the other way round
    box = box.rotate([0, 0, math.degrees(math.atan2(by - ay, bx - ax))])
    return box.translate([(ax + bx) / 2, (ay + by) / 2, 0])


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
    solid = m3d.Manifold.batch_boolean([base] + frustums, m3d.OpType.Add)

    # Every edge of the net that is not a fold is a seam. Gather them all,
    # each turned so the face lies on its +y side, which is the frame the
    # cut and the socket are built in.
    seams = []
    for fi in path:
        pts = dict(face_2d[fi])
        face = SOLID.faces[fi]
        centre = np.mean([q for _, q in face_2d[fi]], axis=0)
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            e = edge_key(a, b)
            if e in fold_edges:
                continue
            pa, pb = pts[a], pts[b]
            nrm = (-(pb[1] - pa[1]), pb[0] - pa[0])
            if nrm[0] * (centre[0] - pa[0]) + nrm[1] * (centre[1] - pa[1]) < 0:
                pa, pb = pb, pa
            seams.append((e, pa, pb))

    lids = []
    for e, pa, pb in seams:
        if e in SOCKET_KEYS:
            lids.append(socket_parts(miter_of(e))[2])

    # then miter every seam, base and all, so the two caps meet on the
    # bisector instead of on a square ledge
    solid -= m3d.Manifold.batch_boolean(
        [seam_cut(pa, pb, miter_of(e)) for e, pa, pb in seams],
        m3d.OpType.Add)

    # then sink the sockets into the rim
    holes = []
    for e, pa, pb in seams:
        if e not in SOCKET_KEYS:
            continue
        cut, rebate, _ = socket_parts(miter_of(e))
        holes += [at_edge(cut, pa, pb), at_edge(rebate, pa, pb)]
    if holes:
        solid -= m3d.Manifold.batch_boolean(holes, m3d.OpType.Add)
    return solid, face_2d, lids


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


def edge_profile(solid, face_2d, fi, va, vb, z_lo, z_hi, samples=25):
    """Read the face that an edge carries back off the FINISHED mesh, so the
    CSG is checked against the design rather than trusted. Slices the solid at
    a series of heights and measures how far the material starts from the edge
    -- sampling the surface itself, not just the few heights that happen to
    carry mesh vertices, which is what made an earlier version of this read
    14mm wrong: a wall face runs parallel to its own edge, so its only
    vertices are at the far corners of the face."""
    pts = dict(face_2d[fi])
    pa, pb = np.array(pts[va]), np.array(pts[vb])
    mid = (pa + pb) / 2
    u = (pb - pa) / np.linalg.norm(pb - pa)
    nrm = np.array([-u[1], u[0]])
    if np.dot(nrm, np.mean([q for _, q in face_2d[fi]], axis=0) - mid) < 0:
        nrm = -nrm                      # point into the face

    out = []
    for z in np.linspace(z_lo, z_hi, samples):
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


def a_seam_edge(path, fold_edges, face_2d):
    """A seam edge to measure the mating face on -- one carrying a SOCKET by
    preference, since that is where the face can be breached and a bare edge
    would prove nothing about it."""
    found = None
    for fi in path:
        face = SOLID.faces[fi]
        for k in range(len(face)):
            a, b = face[k], face[(k + 1) % len(face)]
            e = edge_key(a, b)
            if e in fold_edges:
                continue
            if e in SOCKET_KEYS:
                return fi, a, b
            found = found or (fi, a, b)
    assert found, "a cap with no seam edge is not a cap"
    return found


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
        solid, face_2d, lids = build_strip(path)
        n = len(solid.decompose())
        assert n == 1, f"cap {i} is split into {n} disconnected shells"
        bb = solid.bounding_box()
        w, h = bb[3] - bb[0], bb[4] - bb[1]
        assert max(w, h) <= BED_MM, (
            f"strip is {w:.0f}x{h:.0f}mm and will not fit a {BED_MM:.0f}mm bed "
            f"at TARGET_EDGE={TARGET_EDGE}mm")
        print(f"cap{i}: volume {solid.volume():8.1f}mm3   footprint {w:.1f} x {h:.1f} x "
              f"{bb[5] - bb[2]:.1f}mm")
        solids.append((solid, face_2d, path, lids))

    # The design is only worth one file if the two caps really are the same
    # part. Comparing vertex lists does not answer that -- the two meshes are
    # tessellated differently (205 vs 201 vertices) even though the solids are
    # identical. Ask the boolean kernel instead: turn one onto the other and
    # measure the volume that fails to overlap.
    print(f"\nboth caps: volume {solids[0][0].volume():.6f} / "
          f"{solids[1][0].volume():.6f}, area {solids[0][0].surface_area():.6f} / "
          f"{solids[1][0].surface_area():.6f}")
    turns = net_alignment_turns(solids[0][1], solids[0][2],
                                solids[1][1], solids[1][2], tol=1e-5)
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
    # Try EVERY turn that aligns the nets, not just the one a search happened
    # to return. The two caps are one part if SOME rigid motion carries one
    # onto the other, and a zigzag strip's outline has a half turn in it, so
    # there are always two candidates. Bare caps matched under either; caps
    # with sockets match under only one, and the search picked the other on
    # the octahedron at 63mm and called a good part not congruent.
    diff, turn = min((congruence_error(solids[0][0], solids[1][0], t), t)
                     for t in (turns or [turn]))
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

    solid, face_2d, path, lids = solids[0]
    if SOCKET_KEYS:
        print(f"{len(SOCKET_KEYS)} magnet sockets at cycle edges {SOCKET_AT}; the rim "
              f"is {WALL_HEIGHT:.2f}mm, raised from {MIN_WALL_HEIGHT:.2f} to take them, "
              f"so the inside is flat. Nearest fold cleared by {SOCKET_GAP:.2f}mm")
    else:
        print(f"no magnet sockets: one would come within {SOCKET_GAP:.2f}mm of a fold's "
              f"miter plane at this size. A socket is absolute millimetres and the "
              f"face is not, so a larger size has room.")

    i = min(len(path) // 2, len(path) - 2)     # a 2-face strip has only one fold
    va, vb = shared_verts(SOLID.faces, path[i], path[i + 1])
    fold = miter_of(edge_key(va, vb))
    knee = relief_knee(fold)
    worst = max(abs(m - wall_margin(z, fold)) for z, m in
                edge_profile(solid, face_2d, path[i], va, vb,
                             FACE_THICKNESS + 0.05, TOP_Z - 0.05) if z > knee + 0.2)
    print(f"wall face measured off the finished mesh matches the design to {worst:.4f}mm")
    assert worst < 0.05, "the printed wall is not where the design says it is"

    # And the mating face, over its WHOLE height -- the base included, which
    # is where it used to run straight out to the paper outline and stop the
    # two caps ever touching along their rims.
    fold_edges = compute_fold_edges(SOLID.faces, path)
    sf, sa, sb = a_seam_edge(path, fold_edges, face_2d)
    seam_m = miter_of(edge_key(sa, sb))
    worst = max(abs(m - seam_margin(z, seam_m)) for z, m in
                edge_profile(solid, face_2d, sf, sa, sb, 0.05, TOP_Z - 0.05, 60))
    print(f"seam face is the bisector of a {FOLD_ANGLES[edge_key(sa, sb)]:.2f} deg "
          f"joint to {worst:.4f}mm, all the way down to the outer skin")
    assert worst < 0.02, (
        "the seam is not on the bisector -- the two caps will not close on it")

    name = SOLID.name.split()[-1].lower()
    out = f"hardware/tests/flat_strip_{name}_{TARGET_SIZE:g}mm.stl"
    # the lids ride along beside the strip: they are the same print, and one
    # strip needs exactly as many as it has sockets
    bb = solid.bounding_box()
    shipped = solid
    for j, lid in enumerate(lids):
        b = lid.bounding_box()
        shipped += lid.translate([bb[0] - b[0] + j * (b[3] - b[0] + 3.0),
                                  bb[4] - b[1] + 4.0, -b[2]])
    export_stl(shipped, out)
    print(f"\nwrote {out}  (print TWO of these: strip + {len(lids)} lids)")
