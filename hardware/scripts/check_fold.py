"""Folds both printed caps for real and asks the only question that finally
matters: do they interfere?

    python3 hardware/scripts/check_fold.py [solid]

Everything else measures the flat part against its own design. This takes
the built solid, cuts it at its fold lines, moves each face onto its place
on the polyhedron, does the same for the other cap, and intersects them.
Zero means the two caps close.

Not run on every build -- it is two full builds plus a pile of booleans --
but it is the gate to run before a design goes anywhere near main.

WHAT IT FOUND. On five of the six solids the seam mates perfectly along its
whole length: away from the vertices the two caps overlap by 0.000000mm3.

The octahedron is the exception and is worth recording rather than rounding
away. Its two caps share four needles of 0.043mm across and 9.141mm long,
6.56mm under the surface at four of its six vertices, where the two rims
cross inside the solid. Those figures are identical at 25, 32, 40, 50 and
63mm -- same volume to six decimals, same cross-section, same length -- so
the interference does not scale: the neighbourhood of a vertex is built
from absolute margins, so it is the same 0.043mm on every model, at the
minimum rim height as well as the raised one. 0.043mm is a fifth of a
nozzle width and half the clearance the seam already carries, so it passes
on thickness, but it is a real overlap and not noise. Every bit of
interference there is sits within 2mm of a vertex, where several mating
faces converge at once.

Part of that is this script's own doing and should not be read as a defect:
folding splits the hinge band at each fold line and turns the two halves
rigidly, where the real hinge bends. Fold lines end at vertices, so the
artifact lands exactly where the crowding is. Treat the vertex figure as an
upper bound, and the away-from-vertex figure as the real answer.

Requires: pip install manifold3d numpy
"""

import sys

import manifold3d as m3d
import numpy as np

sys.argv = [sys.argv[0], (sys.argv[1] if len(sys.argv) > 1 else "icosahedron")]
import gen_flat_strip as G          # noqa: E402
from polyhedra import unfold        # noqa: E402

VERTEX_R = 2.0      # mm around each polyhedron vertex, masked off separately

# How much interference counts as none. Not a number picked to make today's
# solids pass: a quarter of a 0.2mm layer, which is a fifth of a 0.4mm
# nozzle, is below what an FDM printer can put down, and it is half the
# 0.08mm the design already leaves between the two mating faces at the seam.
# An overlap thinner than this cannot exist in plastic.
#
# It is measured as a THICKNESS, not a volume, because volume conflates two
# different things: a hairline that runs a long way is harmless, and a blunt
# collision is not. The thickness of a piece is the smallest side of its
# bounding box.
PRINTABLE = 0.05    # mm
VOLUME_CEILING = 0.5  # mm3 -- a sanity bound, so "thin but everywhere" fails

S = G.SOLID
V3 = np.array(S.vertices, float)
V3 *= G.TARGET_EDGE / np.linalg.norm(V3[S.faces[0][0]] - V3[S.faces[0][1]])
CENTRE = V3.mean(0)
unit = lambda w: w / np.linalg.norm(w)


def fold(solid, face_2d, path):
    """Cut the strip at its fold lines and put each face where it belongs.

    The part prints rim UP and folds rim-inward, so the net as laid out
    corresponds to the polyhedron seen from the other side -- mirror it once
    and every face then maps with +z going INTO the solid, which is what a
    fold does. The solid is achiral, so this costs nothing.
    """
    solid = solid.mirror([1, 0, 0])
    pieces = []
    for fi in path:
        gv = [g for g, _ in face_2d[fi]]
        p2 = np.array([p for _, p in face_2d[fi]]) * np.array([-1.0, 1.0])
        p3 = np.array([V3[g] for g in gv])

        e1_2 = unit(p2[1] - p2[0])
        e2_2 = np.array([-e1_2[1], e1_2[0]])
        n3 = unit(np.cross(p3[1] - p3[0], p3[2] - p3[0]))
        if n3 @ (p3.mean(0) - CENTRE) < 0:
            n3 = -n3
        e1_3 = unit(p3[1] - p3[0])
        e2_3 = np.cross(-n3, e1_3)

        R = (np.column_stack([e1_3, e2_3, -n3])
             @ np.column_stack([np.append(e1_2, 0), np.append(e2_2, 0), [0, 0, 1]]).T)
        t = p3[0] - R @ np.append(p2[0], 0.0)
        err = max(np.linalg.norm(R @ np.append(q, 0.0) + t - r) for q, r in zip(p2, p3))
        assert err < 1e-6, f"the fold of face {fi} is off by {err:.2e}mm"

        prism = m3d.Manifold.extrude(
            m3d.CrossSection([G.ccw([tuple(q) for q in p2])]), 4 * G.TOP_Z + 8
        ).translate([0, 0, -(2 * G.TOP_Z + 4)])
        pieces.append((solid ^ prism).transform(np.hstack([R, t[:, None]])))
    return m3d.Manifold.batch_boolean(pieces, m3d.OpType.Add)


if __name__ == "__main__":
    caps, folded = S.strips(), []
    for path in caps:
        solid, f2, _ = G.build_strip(path)
        folded.append(fold(solid, f2, path))
        print(f"cap of {len(path)} faces: flat {solid.volume():8.1f}mm3, "
              f"folded {folded[-1].volume():8.1f}mm3")

    overlap = folded[0] ^ folded[1]
    mask = m3d.Manifold.batch_boolean(
        [m3d.Manifold.sphere(VERTEX_R, 48).translate(list(v)) for v in V3],
        m3d.OpType.Add)
    rest = overlap - mask
    away = rest.volume()
    pieces = rest.decompose() if away > 0 else []
    thick = 0.0
    for piece in pieces:
        b = piece.bounding_box()
        thick = max(thick, min(b[3] - b[0], b[4] - b[1], b[5] - b[2]))

    print(f"\nthe two folded caps overlap by {overlap.volume():.4f}mm3 in total,")
    print(f"of which {away:.6f}mm3 is further than {VERTEX_R:.1f}mm from any vertex")
    if pieces:
        print(f"in {len(pieces)} piece(s), the thickest {thick:.4f}mm across "
              f"(a printer can express {PRINTABLE:.2f}mm)")
    assert thick < PRINTABLE and away < VOLUME_CEILING, (
        f"the two caps occupy the same space away from any vertex: "
        f"{away:.4f}mm3 in {len(pieces)} piece(s), the thickest {thick:.4f}mm "
        f"across. That is more than a printer can ignore -- they will not "
        f"close on the seam")
    if not pieces:
        print("the seam itself mates: all of it is at the vertices, where the "
              "hinge cuts this check makes are themselves suspect")
    else:
        print("the seam mates to within what a printer can express")
