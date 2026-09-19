"""Generates a two-piece dovetail slide-joint test coupon as STL files.

This is a standalone test of the connector mechanism planned for joining
the printed polyhedron faces: a short dovetail key that lets one piece
slide onto its neighbor along their shared edge, without adding anything
outside the piece's own wall thickness.

Requires: pip install manifold3d

Run from the repo root:
    python3 hardware/scripts/gen_dovetail_test.py
"""

import manifold3d as m3d

EDGE_LEN = 24.0    # mm, length of the test joint along the "edge"
WALL_T = 3.0        # mm, matches the real piece's wall/extrusion depth -- this
                     # is the hard ceiling the dovetail's width must stay under
BLOCK_N = 6.0        # mm, extra material behind the joint face, just to make a holdable test block
NECK_W = 1.0         # mm, dovetail width at its base (narrow end, at the mating face)
TIP_W = 1.6          # mm, dovetail width at its tip (wide end, buried in the neighbor)
PROTRUDE = 1.6       # mm, how far the ridge sticks out past the mating face
CLEARANCE = 0.15     # mm, extra half-gap added per side on the groove for FDM slop

# The tongue's tip (plus clearance) must leave real shoulder material on both
# sides within the wall thickness -- otherwise the pocket cut for the groove
# breaches clean through the wall and disconnects the material past that
# point from the rest of the piece (this happened in an earlier version:
# TIP_W=3.0 with WALL_T=3.0 sliced the groove piece into 3 floating shells).
assert TIP_W + 2 * CLEARANCE < WALL_T, "dovetail tip (with clearance) would breach the wall thickness"


def dovetail_profile(neck_w, tip_w, protrude):
    # Trapezoid in (N, T): narrow at N=0 (the mating face), wide at N=protrude.
    # Counter-clockwise winding -- manifold3d treats a clockwise polygon as
    # zero/negative area under the default fill rule.
    return m3d.CrossSection([[
        (0.0, -neck_w / 2), (protrude, -tip_w / 2),
        (protrude, tip_w / 2), (0.0, neck_w / 2),
    ]])


def ridge_piece():
    # Bulk material in X < 0; the tongue pokes into X in [0, PROTRUDE], which
    # is the neighbor's territory once assembled.
    block = m3d.Manifold.cube([BLOCK_N, WALL_T, EDGE_LEN]).translate([-BLOCK_N, -WALL_T / 2, 0])
    key = m3d.Manifold.extrude(dovetail_profile(NECK_W, TIP_W, PROTRUDE), EDGE_LEN)
    return block + key


def groove_piece():
    # Bulk material in X > 0; the pocket is cut into the SAME world region
    # ([0, PROTRUDE+slop]) the tongue occupies, so the two mating faces
    # (both at world X=0) sit flush when assembled.
    block = m3d.Manifold.cube([BLOCK_N, WALL_T, EDGE_LEN]).translate([0, -WALL_T / 2, 0])
    pocket = m3d.Manifold.extrude(
        dovetail_profile(NECK_W + 2 * CLEARANCE, TIP_W + 2 * CLEARANCE, PROTRUDE + 0.3),
        EDGE_LEN,
    )
    return block - pocket


def export_stl(manifold_obj, path):
    mesh = manifold_obj.to_mesh()
    verts = mesh.vert_properties
    tris = mesh.tri_verts
    with open(path, "w") as f:
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
    rp = ridge_piece()
    gp = groove_piece()

    # Sanity checks before trusting the geometry enough to print it:
    #  - each piece must be a single connected shell, not several floating
    #    fragments (what a too-wide dovetail produces, per the assert above)
    #  - the two pieces must not overlap in space once clearance is honored
    rp_parts = len(rp.decompose())
    gp_parts = len(gp.decompose())
    assert rp_parts == 1, f"ridge piece is split into {rp_parts} disconnected shells"
    assert gp_parts == 1, f"groove piece is split into {gp_parts} disconnected shells"

    overlap = m3d.Manifold.batch_boolean([rp, gp], m3d.OpType.Intersect).volume()
    assert overlap < 1e-6, f"ridge and groove overlap by {overlap} mm^3 -- clearance is too tight"

    export_stl(rp, "hardware/tests/dovetail_test_ridge.stl")
    export_stl(gp, "hardware/tests/dovetail_test_groove.stl")
    print("OK: both pieces are single connected shells with no overlap.")
    print("Wrote hardware/tests/dovetail_test_ridge.stl and dovetail_test_groove.stl")
