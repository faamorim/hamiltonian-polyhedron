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

THE RIM. The magnet needs 4mm measured up the mating face, and the rim has
to be tall enough to hold that plus the lid's rebate. So the rim is raised
to suit -- everywhere, not as a lump on the faces that carry a socket, so
the inside of a finished cap comes out flat. The inside is seen: pulling the
two caps apart is the whole point of the object. Nothing below the face
moves, so the hinge is untouched.

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

# The socket itself lives in gen_flat_strip now, with the part it goes into.
# Named here only so the test and the real thing cannot drift apart.
MAGNET_DIA, MAGNET_H = G.MAGNET_DIA, G.MAGNET_H
FIT, WALL, SKIN = G.MAGNET_FIT, G.SOCKET_WALL, G.SOCKET_SKIN
LID_T, LID_GAP, PROUD = G.LID_T, G.LID_GAP, G.SOCKET_PROUD

TILE_LEN = 26.0           # mm along the seam
TILE_DEEP = 14.0          # mm back from it, into the face
SEG = 96


frame = G.seam_frame
on_face = G.on_seam
seat_t = G.socket_seat
boss_top_z = G.socket_top


def pocket_top_z(miter):
    return G.socket_top(miter) - LID_T - 0.2


def build(miter):
    top = G.TOP_Z
    cut, rebate, lid = G.socket_parts(miter)

    tile = m3d.Manifold.cube([TILE_LEN, TILE_DEEP, top], False)
    tile = tile.translate([-TILE_LEN / 2, 0, 0])
    tile -= G.seam_cut((-TILE_LEN, 0.0), (TILE_LEN, 0.0), miter, up_to=top)
    tile -= cut
    tile -= rebate
    return tile, lid, None


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
          f"{G.TOP_Z:.2f}mm -- which is what sets the rim")
    pw, pd = G.socket_footprint(miter)
    print(f"  rim {G.WALL_HEIGHT:.2f}mm (from {G.MIN_WALL_HEIGHT:.2f}), so the inside "
          f"is flat; the socket takes {pw:.1f} x {pd:.1f}mm of face")
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
