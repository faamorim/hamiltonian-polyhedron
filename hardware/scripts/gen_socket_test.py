"""A small two-piece test of the SEAM and its magnet socket.

    python3 hardware/scripts/gen_socket_test.py [solid]

Print TWO wedges and TWO lids. The wedges are the same part -- that is what
a bisector buys, each side cut at half the angle -- so two copies mate face
to face exactly as two caps do.

THE SOCKET. The magnet stands with its poles across the seam and drops in
edgewise from the INSIDE face, down a channel in its own plane. A thin wall
is left in front of it, between the magnet and the mating face. That wall is
the whole point:

  * attraction presses the magnet INTO it, so the load runs magnet -> wall
    -> body of the face. Plastic in compression, backed by the part. The
    earlier design bored straight into the mating face, which pointed the
    same pull straight out of the hole with nothing but friction across it;
    glue there would have been loaded in peel, the worst case you can give
    an adhesive, and it did not hold.
  * repulsion, if someone offers the wrong poles up, presses it the other
    way into solid material. So nothing ever pushes on the lid, which is
    free to be what it is: cosmetic, seen when the two caps are pulled
    apart, which is the whole point of the object.
  * its thickness sets the gap between the two magnets, and so the force.
    It falls out of the size calculation entirely -- thickening it lowers
    the top and the bottom of the pocket by the same amount -- so this is a
    number chosen on feel, not one the geometry forces. 0.8mm here, 1.6mm
    between magnets once both caps have one.

THE BOSS. The magnet needs 4mm measured up the mating face, and at the 3mm
rim the strips use there is only 4.50mm of face to put it in, once skin over
the outer surface is taken off. Rather than raise the rim everywhere -- it
would want 4.34mm, on every face of every solid -- this raises a local pad
on the inside at the socket, and nowhere else. The rim stays 3mm, every fold
keeps the geometry already print-tested, and only four faces per cap carry
a socket at all. Printed flat the pad is a block on top of a flat surface:
no overhang, no support.

WHY FOUR. The cycle has twelve edges but they do not all need magnets -- the
solid holds its own shape, the magnets only align the two rims. Two of the
twelve are where the strips' END faces meet each other, and those are the
ones that matter; two more halfway between keep it from hinging. And the
one-part constraint is kind here: on the icosahedron those two end-to-end
joints form a single orbit, and the two midpoints are fixed points, so
exactly four sockets cut in the one part give four complete pairs, evenly
spaced at cycle edges 0, 3, 6 and 9.

Requires: pip install manifold3d numpy
"""

import math
import sys

import manifold3d as m3d
import numpy as np

sys.argv = [sys.argv[0], (sys.argv[1] if len(sys.argv) > 1 else "icosahedron")]
import gen_flat_strip as G          # noqa: E402  (reads the solid off argv)

MAGNET_DIA, MAGNET_H = 4.0, 2.0
FIT = 0.2                 # mm added to the pocket so the magnet drops in
WALL = 0.8                # mm of plastic between the magnet and the mating face
SKIN = 0.8                # mm of plastic over the outer surface, under the pocket
LID_T = 0.8               # mm, the lid and the rebate it sits flush in
LID_GAP = 0.15            # mm clearance round the lid
PROUD = 0.4               # mm every cutter runs past the surface it cuts

TILE_LEN = 26.0           # mm along the seam
TILE_DEEP = 14.0          # mm back from it, into the face
BOSS_W = 9.0              # mm of pad along the seam
BOSS_D = 7.0              # mm of pad back from the mating face
SEG = 96


def frame(miter):
    """Along the seam, into the material square to the mating face, and up
    that face. The magnet's axis is the second; it drops in along the third."""
    sec = math.hypot(1.0, miter)
    return (np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0 / sec, -miter / sec]),
            np.array([0.0, miter / sec, 1.0 / sec]))


def on_face(t, miter):
    """The point t millimetres up the mating face from the seam edge."""
    _, _, e2 = frame(miter)
    return np.array([0.0, G.CONTACT_CLEARANCE, 0.0]) + t * e2


def seat_t(miter):
    """Where the magnet sits, measured up the mating face.

    Far enough that the pocket's lowest corner -- the far bottom one, since
    the pocket leans back with the miter -- still has skin over the outer
    surface."""
    r, h = (MAGNET_DIA + FIT) / 2, MAGNET_H + FIT
    return r + (WALL + h) * miter + SKIN * math.hypot(1.0, miter)


def pocket_top_z(miter):
    """The highest the magnet reaches, which is what the pad has to clear."""
    sec = math.hypot(1.0, miter)
    r = (MAGNET_DIA + FIT) / 2
    return (seat_t(miter) + r) / sec - WALL * miter / sec


def boss_top_z(miter):
    return pocket_top_z(miter) + LID_T + 0.2


def socket(miter, along):
    """The magnet's pocket and the channel it slides down, as one swept
    solid: a cylinder dragged up the mating face until it leaves the pad.
    Sweeping a convex body along a line is the hull of its two ends."""
    _, n, e2 = frame(miter)
    r, h = (MAGNET_DIA + FIT) / 2, MAGNET_H + FIT
    cyl = m3d.Manifold.cylinder(h, r, r, SEG)
    cyl = cyl.rotate([math.degrees(math.atan2(-1.0, -miter)), 0, 0])
    seat = on_face(seat_t(miter), miter) + WALL * n + np.array([along, 0, 0])
    lo = cyl.translate(list(seat))
    run = (boss_top_z(miter) + PROUD) * math.hypot(1.0, miter)   # more than enough
    return m3d.Manifold.batch_hull([lo, lo.translate(list(run * e2))])


def build(miter):
    top = boss_top_z(miter)

    tile = m3d.Manifold.cube([TILE_LEN, TILE_DEEP, G.TOP_Z], False)
    tile = tile.translate([-TILE_LEN / 2, 0, 0])
    # the pad, sitting on the rim, its seam side carried on by the same cut
    boss = m3d.Manifold.cube([BOSS_W, BOSS_D, top], False)
    boss = boss.translate([-BOSS_W / 2, 0, 0])
    tile += boss
    tile -= G.seam_cut((-TILE_LEN, 0.0), (TILE_LEN, 0.0), miter, up_to=top)

    cut = socket(miter, 0.0)
    tile -= cut

    # a rebate round the channel's mouth so the lid finishes flush, shaped
    # from the mouth itself rather than guessed at
    mouth = cut.slice(top - LID_T).to_polygons()
    rebate2d = m3d.CrossSection(mouth).offset(1.2, m3d.JoinType.Round, 2.0, SEG)
    rebate = m3d.Manifold.extrude(rebate2d, LID_T + PROUD).translate([0, 0, top - LID_T])
    tile -= rebate

    lid2d = rebate2d.offset(-LID_GAP, m3d.JoinType.Round, 2.0, SEG)
    lid = m3d.Manifold.extrude(lid2d, LID_T)
    return tile, lid, rebate2d


def partner(tile, miter):
    """The tile this one mates with: the SAME part, turned 180 degrees about
    the line where the two mating faces meet. A proper rotation, not a
    mirror -- which is why two of one part is all you print."""
    sec = math.hypot(1.0, miter)
    k = np.array([0.0, miter / sec, 1.0 / sec])
    R = 2 * np.outer(k, k) - np.eye(3)
    c = G.CONTACT_CLEARANCE
    return (tile.translate([0, -c, 0])
                .transform(np.hstack([R, np.zeros((3, 1))]))
                .translate([0, c, 0]))


def seam_error(tile, miter, x=0.0):
    """The mating face read off the finished mesh, on the line through the
    socket -- so a pocket that broke the thin wall would show up here."""
    worst = 0.0
    for z in np.linspace(0.05, G.TOP_Z - 0.05, 60):
        hits = []
        for poly in tile.slice(float(z)).to_polygons():
            q = np.asarray(poly)
            for k in range(len(q)):
                t = G.ray_hits_segment(np.array([x, 0.0]), np.array([0.0, 1.0]),
                                       q[k], q[(k + 1) % len(q)])
                if t is not None:
                    hits.append(t)
        if hits:
            worst = max(worst, abs(min(hits) - G.seam_margin(z, miter)))
    return worst


if __name__ == "__main__":
    name = G.SOLID.name
    ang = min(round(a, 6) for a in G.FOLD_ANGLES.values())
    miter = math.tan(math.radians(ang) / 2)
    sec = math.hypot(1.0, miter)
    tile, lid, rebate2d = build(miter)

    face_w = G.TOP_Z * sec
    top = boss_top_z(miter)
    r = (MAGNET_DIA + FIT) / 2
    low_z = (seat_t(miter) - r) / sec - (WALL + MAGNET_H + FIT) * miter / sec

    print(f"{name}: sharpest seam closes through {ang:.2f} deg, miter {miter:.3f}")
    print(f"  {WALL:.2f}mm wall in front of the magnet, so {2 * WALL:.2f}mm between "
          f"the two of them once both caps have one")
    print(f"  magnet seated {seat_t(miter):.2f}mm up the mating face, reaching "
          f"{pocket_top_z(miter):.2f}mm; the face itself only runs to "
          f"{G.TOP_Z:.2f}mm, which is why there is a pad")
    print(f"  pad {top - G.TOP_Z:.2f}mm proud of the {G.WALL_HEIGHT:.2f}mm rim, "
          f"{BOSS_W:.0f} x {BOSS_D:.0f}mm; the rim itself is untouched")
    print(f"  pocket's lowest corner clears the outer surface by {low_z:.2f}mm")
    print(f"  lid {LID_T:.2f}mm thick, flush in its rebate, {LID_GAP:.2f}mm all round")

    assert low_z > SKIN - 1e-6, f"the pocket is {low_z:.2f}mm off the outer surface"
    n = sum(1 for m in tile.decompose() if m.volume() > 1e-6)
    assert n == 1, f"the wedge came out in {n} pieces"

    err = seam_error(tile, miter)
    assert err < 0.02, f"the mating face is {err:.4f}mm off the bisector"
    clash = (tile ^ partner(tile, miter)).volume()
    assert clash < 1e-6, f"the pair overlaps by {clash:.4f}mm3"
    print(f"  mating face on the bisector to {err * 1000:.1f} um straight through "
          f"the socket, so the wall is unbroken; the pair closes with nothing "
          f"of either inside the other")

    # the lid must drop into its rebate without touching the sides
    seated = lid.translate([0, 0, top - LID_T])
    bite = (seated ^ tile).volume()
    assert bite < 1e-6, f"the lid fouls its rebate by {bite:.4f}mm3"
    print(f"  lid seats in the rebate without touching the sides")

    b = lid.bounding_box()
    print(f"  wedge {tile.volume() / 1000:.2f}cm3, lid "
          f"{b[3] - b[0]:.1f} x {b[4] - b[1]:.1f} x {LID_T:.1f}mm "
          f"-- {2 * (tile.volume() + lid.volume()) / 1000:.2f}cm3 for the pair")

    out = f"hardware/tests/socket_test_{name.split()[-1].lower()}.stl"
    G.export_stl(tile + lid.translate([0, TILE_DEEP + 4.0, 0]), out)
    print(f"\nwrote {out}  (print TWO, flat on the bed, base down)")
