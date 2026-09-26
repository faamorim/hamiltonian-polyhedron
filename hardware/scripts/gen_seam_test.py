"""A two-piece test of the SEAM: the mitered joint where one cap meets the
other, and the magnet that is to hold it shut.

    python3 hardware/scripts/gen_seam_test.py [solid]

Print TWO. They are the same part -- that is what a bisector buys, each side
cut at half the angle -- so two copies of one tile mate face to face exactly
as two caps do. Print them flat on the bed, base down, in the orientation
they come out in, because everything this tests depends on that.

WHY THE RIM IS TALLER HERE. The mating face is a band TOP_Z / cos(half the
fold) wide, so the shallower the fold, the narrower the face -- 4.50mm on
the icosahedron against 7.27mm on the tetrahedron. A 4mm magnet in a
press-fit bore needs 5.35mm of it once there is 0.6mm of plastic on each
side, which four of the six solids do not have at the 3.0mm rim the strips
are built with today. Raising the rim to 3.8mm gives every one of them
enough. That is the decision this tile is for: it is built at 3.8 so that
what you hold is the thing being proposed, not a version the magnet cannot
fit in. Nothing below the face is touched, so the hinge is unchanged.

Three questions in one print:

  1. Does the mitered seam close flush? Hold the tiles together with no
     magnets first. They should meet along the whole face with no rock and
     no daylight. A gap at the outer skin means the miter is wrong; a gap at
     the inner edge means CONTACT_CLEARANCE is too generous.

  2. Does the bore print? It is a blind hole whose axis lies only
     atan(miter) above horizontal -- 21 degrees on the icosahedron, the
     worst of the six -- so its ceiling is an unsupported arch about 4mm
     across. The tile carries TWO, to settle this by looking rather than by
     arguing: a plain round bore, and a teardrop whose 45-degree roof has no
     flat overhang anywhere. If the round one comes out clean, the teardrop
     is not worth having.

  3. Is the pull right? A 4x2 N42 pair nearly touching runs about 4-5N, and
     the finished solid would have one pair per seam edge, or half that if
     they alternate. Two pairs here is a direct sample: if two feel about
     right in the hand, twelve will not.

Requires: pip install manifold3d numpy
"""

import math
import sys

import manifold3d as m3d
import numpy as np

sys.argv = [sys.argv[0], (sys.argv[1] if len(sys.argv) > 1 else "icosahedron")]
import gen_flat_strip as G          # noqa: E402  (reads the solid off argv)

MAGNET_DIA, MAGNET_H = 4.0, 2.0
BORE_DIA = MAGNET_DIA + 0.15    # a press fit, not a slip fit
BORE_DEPTH = MAGNET_H + 0.3     # so the magnet sits just below the face
OUTER_SKIN = 0.6                # plastic between the bore and the outer surface
BACKING = 0.6                   # plastic behind the bottom of the bore
TEARDROP_VENT = 0.4             # mm the teardrop's tip runs PAST the inner surface
BORE_PROUD = 0.3                # mm the cutter starts OUTSIDE the mating face

TILE_LEN = 44.0                 # mm along the seam
TILE_DEEP = 16.0                # mm back from it, into the face
SEG = 96

# Where the bores sit along the seam. The partner tile is this same part
# turned 180 degrees about the joint, and that turn REVERSES the seam
# direction -- a bore at +x is met by the one the partner had at -x. So the
# positions have to be symmetric or a round bore would face a teardrop and
# the comparison would be between two different things. Round outboard,
# teardrop inboard, two of each.
ROUND_AT = 13.0
TEARDROP_AT = 5.0


def rim_for_magnet(miters):
    """The shortest rim that gives every fold of this solid a mating face the
    bore fits in, with skin on both sides.

    The bore runs square into the mating face, so it DIVES: its axis loses
    miter in z for every 1 it gains going in, and its lowest point is at the
    far end, not at the mouth. Reading the mouth alone says the icosahedron
    needs a 3.80mm rim; the far end of the same bore is then 0.26mm below the
    bed. Taking the whole bore, the face has to hold 2r + depth x miter,
    measured across it, plus skin on both sides."""
    return max((BORE_DIA + BORE_DEPTH * m) / math.hypot(1.0, m)
               for m in miters) + 2 * OUTER_SKIN - G.FACE_THICKNESS


def teardrop_apex(r):
    """How far the teardrop's tip sits from the bore's axis."""
    return r * math.sqrt(2) + TEARDROP_VENT


def bore_centre_t(miter, face_w):
    """Where on the mating face the bore is centred, measured along that face
    from the outer edge.

    Far enough in that the FAR END of the bore, which is the part nearest the
    bed, still has skin under it, and no further in than leaves skin on the
    inner side. Both bounds are measured perpendicular to the outer surface
    and converted along the face, which is where the sec factor comes from."""
    sec = math.hypot(1.0, miter)
    lo = BORE_DIA / 2 + BORE_DEPTH * miter + OUTER_SKIN * sec
    hi = face_w - BORE_DIA / 2 - OUTER_SKIN * sec
    assert lo <= hi + 1e-9, (
        f"a {BORE_DIA:.2f}mm bore does not fit a {face_w:.2f}mm mating face: "
        f"it must sit at least {lo:.2f}mm along it and at most {hi:.2f}mm")
    return (lo + hi) / 2 if hi > lo else lo


def bore(miter, face_w, along, teardrop):
    """One pocket, bored square into the mating face.

    Built pointing up and then laid over, because the axis is not a round
    number: it is the inward normal of the bisector plane, (0, 1, -miter)
    normalised. Turning about x by atan2(-1, -miter) takes local +z onto it,
    and takes local -y onto the world-up direction within that plane -- which
    is where the teardrop's ridge has to go, so the topmost line of the bore
    is a peak rather than a ceiling no layer can start on.
    """
    r = BORE_DIA / 2
    section = m3d.CrossSection.circle(r, SEG)
    if teardrop:
        # the apex sits r*sqrt2 from the centre, so the hull's two straight
        # flanks leave it at 45 degrees and run tangent into the circle. A
        # hull rather than a drawn triangle: a triangle wide enough to reach
        # the apex overhangs the circle at its base, and those little wings
        # are themselves the overhang the shape exists to avoid.
        # The apex would sit at r*sqrt2 for flanks at exactly 45 degrees.
        # Putting it further out only makes them steeper, and it settles a
        # question the geometry otherwise leaves to chance: where the tip
        # lands relative to the inner surface. Left at r*sqrt2 it grazes it
        # on the octahedron -- 0.12mm through, near enough tangent that the
        # boolean returns a zero-volume sliver and the tile stops being one
        # piece. Run it clear past instead: the tip vents into the sealed
        # inside of the model, which nothing sees, on every solid.
        apex = m3d.CrossSection.square([0.02, 0.02], True)
        section = m3d.CrossSection.batch_hull(
            [section, apex.translate([0.0, -teardrop_apex(r)])])
    # Start the cutter BORE_PROUD outside the face rather than exactly on it.
    # A cutter whose end lands flush with the surface it cuts is a coplanar
    # boolean, and what comes back is combinatorially sound but geometrically
    # junk: the face keeps triangles that run straight across the hole. It
    # webbed 431 of 540 probe points inside the round bore's mouth, and put
    # 831 triangles on a rectangle with four holes in it. Same rule as
    # OVERLAP and TRENCH_OVERSHOOT next door, which exist for exactly this.
    body = m3d.Manifold.extrude(section, BORE_DEPTH + BORE_PROUD)
    body = body.translate([0, 0, -BORE_PROUD])
    body = body.rotate([math.degrees(math.atan2(-1.0, -miter)), 0, 0])

    s = math.hypot(1.0, miter)
    t = bore_centre_t(miter, face_w)
    return body.translate([along, G.CONTACT_CLEARANCE + t * miter / s, t / s])


def build_tile():
    angles = sorted({round(a, 6) for a in G.FOLD_ANGLES.values()})
    miters = [math.tan(math.radians(a) / 2) for a in angles]

    # the rim this solid needs for the magnet -- see the module docstring.
    # Everything below FACE_THICKNESS, the hinge included, is untouched.
    G.WALL_HEIGHT = rim_for_magnet(miters)
    G.TOP_Z = G.FACE_THICKNESS + G.WALL_HEIGHT

    ang, miter = angles[0], miters[0]        # the shallowest fold is the tightest
    face_w = G.TOP_Z * math.hypot(1.0, miter)

    tile = m3d.Manifold.cube([TILE_LEN, TILE_DEEP, G.TOP_Z], False)
    tile = tile.translate([-TILE_LEN / 2, 0, 0])
    tile -= G.seam_cut((-TILE_LEN, 0.0), (TILE_LEN, 0.0), miter)
    tile -= m3d.Manifold.batch_boolean(
        [bore(miter, face_w, x, False) for x in (-ROUND_AT, ROUND_AT)]
        + [bore(miter, face_w, x, True) for x in (-TEARDROP_AT, TEARDROP_AT)],
        m3d.OpType.Add)
    return tile, ang, miter, face_w


def partner(tile, miter):
    """The tile that mates with this one: the SAME part, turned 180 degrees
    about the line where the two mating faces meet -- the one direction in
    the bisector plane perpendicular to the seam. It is a proper rotation, so
    the two really are one part printed twice; what it is not is a mirror
    image, and it reverses x, which is why the bores are laid out
    symmetrically."""
    s = math.hypot(1.0, miter)
    k = np.array([0.0, miter / s, 1.0 / s])
    R = 2 * np.outer(k, k) - np.eye(3)          # Rodrigues at 180 degrees
    c = G.CONTACT_CLEARANCE
    return (tile.translate([0, -c, 0])
                .transform(np.hstack([R, np.zeros((3, 1))]))
                .translate([0, c, 0]))


def face_webbing(tile, miter, face_w, mouths):
    """How much of each bore's mouth the mating face has been triangulated
    OVER. Should be none: a hole is a hole.

    Worth checking every build rather than trusting the kernel, because a
    coplanar boolean fails in a way nothing else here would catch -- the mesh
    stays closed, 2-manifold and consistently wound, every edge paired, and
    the seam still measures exactly right. It is only wrong where nothing was
    looking: triangles of the face lying across the opening. They show up in
    a slicer as a web over the hole, and would print as one.
    """
    sec = math.hypot(1.0, miter)
    n = np.array([0.0, 1.0 / sec, -miter / sec])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.cross(n, e1)

    mesh = tile.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    tri = v[np.asarray(mesh.tri_verts)]
    off = (tri[:, :, 1] - miter * tri[:, :, 2] - G.CONTACT_CLEARANCE) / sec
    face = tri[(np.abs(off) < 1e-6).all(axis=1)]
    P = np.stack([face @ e1, face @ e2], axis=-1)

    def covered(p):
        a, b, c = P[:, 0], P[:, 1], P[:, 2]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(d) > 1e-12
        l1 = np.where(ok, ((b[:, 1] - c[:, 1]) * (p[0] - c[:, 0])
                           + (c[:, 0] - b[:, 0]) * (p[1] - c[:, 1])) / np.where(ok, d, 1), -1)
        l2 = np.where(ok, ((c[:, 1] - a[:, 1]) * (p[0] - c[:, 0])
                           + (a[:, 0] - c[:, 0]) * (p[1] - c[:, 1])) / np.where(ok, d, 1), -1)
        return bool(np.any((l1 > 1e-9) & (l2 > 1e-9) & (1 - l1 - l2 > 1e-9)))

    hits = probes = 0
    for m3 in mouths:
        m2 = np.array([m3 @ e1, m3 @ e2])
        for k in range(120):
            th = 2 * math.pi * k / 120
            for frac in (0.35, 0.6, 0.85):
                probes += 1
                hits += covered(m2 + frac * (BORE_DIA / 2)
                                * np.array([math.cos(th), math.sin(th)]))
    return hits, probes, len(face)


def seam_error(tile, miter):
    """Read the mating face back off the finished mesh, away from the bores."""
    worst = 0.0
    for z in np.linspace(0.05, G.TOP_Z - 0.05, 60):
        hits = []
        for poly in tile.slice(float(z)).to_polygons():
            q = np.asarray(poly)
            for k in range(len(q)):
                t = G.ray_hits_segment(np.array([0.0, 0.0]), np.array([0.0, 1.0]),
                                       q[k], q[(k + 1) % len(q)])
                if t is not None:
                    hits.append(t)
        if hits:
            worst = max(worst, abs(min(hits) - G.seam_margin(z, miter)))
    return worst


if __name__ == "__main__":
    tile, ang, miter, face_w = build_tile()
    name = G.SOLID.name
    t = bore_centre_t(miter, face_w)

    print(f"{name}: sharpest seam closes through {ang:.2f} deg, miter {miter:.3f}")
    sec0 = math.hypot(1.0, miter)
    print(f"  rim raised {G.WALL_HEIGHT - 3.0:+.2f}mm to {G.WALL_HEIGHT:.2f}mm so the "
          f"mating face is {face_w:.2f}mm wide -- a {BORE_DIA:.2f}mm bore that dives "
          f"{BORE_DEPTH * miter:.2f}mm over its depth needs "
          f"{BORE_DIA + BORE_DEPTH * miter + 2 * OUTER_SKIN * sec0:.2f}mm of it")
    sec = math.hypot(1.0, miter)
    print(f"  bore centred {t:.2f}mm along that face; its lowest point "
          f"(the far end) clears the outer surface by "
          f"{(t - BORE_DIA / 2) / sec - BORE_DEPTH * miter / sec:.2f}mm, and its "
          f"highest (the mouth) is {G.TOP_Z - (t + BORE_DIA / 2) / sec:.2f}mm "
          f"below the inner surface")
    print(f"  its axis lies {math.degrees(math.atan(miter)):.1f} deg above horizontal, "
          f"so the round one bridges {BORE_DIA:.2f}mm")
    apex = t + teardrop_apex(BORE_DIA / 2)
    print(f"  the round bore is fully enclosed; the teardrop's tip reaches "
          f"{apex:.2f}mm along a {face_w:.2f}mm face, venting "
          f"{apex - face_w:.2f}mm into the sealed inside of the model")
    assert apex > face_w + 0.1, (
        "the teardrop's tip all but grazes the inner surface -- raise "
        "TEARDROP_VENT so the boolean has something definite to cut")
    print(f"  magnet {MAGNET_DIA:.1f} x {MAGNET_H:.1f}mm ends up "
          f"{BORE_DEPTH - MAGNET_H:.1f}mm below the face, so the plastic meets first")
    print(f"  4 bores: round at x = +/-{ROUND_AT:.0f}mm, teardrop at "
          f"+/-{TEARDROP_AT:.0f}mm -- symmetric, because the partner is this part "
          f"turned over and that reverses the seam")

    err = seam_error(tile, miter)
    n = sum(1 for m in tile.decompose() if m.volume() > 1e-6)
    assert n == 1, f"the tile came out in {n} pieces"

    # the two really must be one part: lay the partner on this one and see
    # that nothing of either is left sticking into the other
    clash = (tile ^ partner(tile, miter)).volume()
    assert clash < 1e-6, (
        f"the tile and its partner overlap by {clash:.4f}mm3 -- they do not "
        f"close on the seam")
    assert err < 0.02, f"the mating face is {err:.4f}mm off the bisector"
    print(f"  mating face measured off the mesh: on the bisector to {err * 1000:.1f} um, "
          f"one shell, {2 * tile.volume() / 1000:.1f}cm3 of filament for the pair")
    print(f"  the pair closes with nothing of either inside the other")

    sec = math.hypot(1.0, miter)
    mouths = [np.array([x, G.CONTACT_CLEARANCE + t * miter / sec, t / sec])
              for x in (-ROUND_AT, ROUND_AT, -TEARDROP_AT, TEARDROP_AT)]
    webbed, probes, n_face = face_webbing(tile, miter, face_w, mouths)
    print(f"  bore mouths open: {probes - webbed}/{probes} probe points clear, "
          f"{n_face} triangles on the mating face")
    assert webbed == 0, (
        f"{webbed} of {probes} points inside the bore mouths are covered by a "
        f"triangle of the mating face -- the holes are webbed over")

    out = f"hardware/tests/seam_test_{name.split()[-1].lower()}.stl"
    G.export_stl(tile, out)
    print(f"\nwrote {out}  (print TWO, flat on the bed, base down)")
