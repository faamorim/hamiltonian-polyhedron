"""Generates half of a solid with its vertices flattened, for the string
activity: tap a tack into each flat, then wind string from tack to tack and
try to find a Hamiltonian cycle on the real thing.

    python3 hardware/scripts/gen_peg_solid.py [solid]

Truncating is not decoration. A vertex is a point where several faces meet,
so there is nowhere to seat a tack and nothing for its head to sit flush
against; cutting each vertex back gives a small flat facing straight out.

Why half. Printed whole and resting on a face, the vertex flats low down
face downward-and-outward and become near-ceiling overhangs. Cut in two and
stood on the cut, both halves print with no support at all.

Which cut matters, and not the obvious one. Splitting across a 2-fold axis
gives a perfectly flat silhouette and zero overhang, but on most solids the
plane passes exactly through four vertices, so four tack holes end up split
lengthwise down the seam. Cutting across the axis THROUGH two opposite
vertices avoids that -- no vertex touches the plane, so no hole is ever
divided -- at the cost of a few degrees of overhang, far inside what prints
cleanly. The script searches the solid's own symmetry directions for the
gentlest cut that leaves the seam clear, and checks the result.

The two halves come out the same part on every solid here except the
tetrahedron, so one file is printed twice.

Requires: pip install manifold3d numpy

Run from the repo root.
"""

import math
import sys

import manifold3d as m3d
import numpy as np

from polyhedra import SOLIDS, face_edges

TARGET_MEAN_WIDTH_MM = 77.87  # the figure the fold-up nets are sized to, so a
                              # printed set all reads as the same size
TRUNCATE_FRAC = 0.12          # cut each edge back by this much of its length at
                              # each end; sets how big the tack flats come out
# A cone in the middle of each flat, purely so the tack point has somewhere to
# start instead of skating off a sloped face. It is a centre punch and nothing
# more -- driven or heat-set, the shank's grip comes from the material around
# it, and a dimple this size adds no strength worth counting.
#
# Deeper and narrower prints better than shallow and wide, which is the wrong
# way round from intuition: the cone widens as it rises, so its wall is an
# overhang at atan(radius / depth) from vertical. At 1.6 x 1.0mm that is 38.7
# degrees, comfortably inside what prints clean; flattening it to 1.6 x 0.4
# would make it 63 degrees and come out rough.
DIMPLE_DIA = 1.6              # mm across where it meets the flat
DIMPLE_DEPTH = 1.0            # mm to the point
DIMPLE_PROUD = 0.2            # mm the cut is carried past the surface, so the
                              # boolean never has to resolve a tangent face

SEAM_KEY_DIA = 0.0            # alignment pins across the seam, if ever wanted

BIG = 1e3                     # half-space boxes, comfortably larger than the part


def scaled_vertices(solid):
    verts = np.array(solid.vertices, float)
    verts -= verts.mean(axis=0)
    return verts * (TARGET_MEAN_WIDTH_MM / solid.mean_width() / solid.reference_edge())


def truncated(verts, edges, frac):
    """Every vertex cut back to a flat. The corners of the result are exactly
    the points where the cutting planes meet the edges, so the convex hull of
    those points IS the truncated solid -- no boolean subtraction needed."""
    pts = []
    for a, b in edges:
        pts.append(verts[a] + frac * (verts[b] - verts[a]))
        pts.append(verts[b] + frac * (verts[a] - verts[b]))
    return m3d.Manifold.hull_points([tuple(p) for p in pts]), np.array(pts)


def mesh_faces(man):
    mesh = man.to_mesh()
    return np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts)]


def rotation_taking(axis, target=(0.0, 0.0, 1.0)):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    b = np.asarray(target, float)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + math.sin(math.atan2(s, c)) * k + (1 - c) * (k @ k)


def flat_points(verts, edges, v, frac):
    """Corners of the tack flat cut at vertex v."""
    return np.array([verts[v] + frac * (verts[u] - verts[v])
                     for a, b in edges for u in ([b] if a == v else [a] if b == v else [])])


def splits_a_flat(verts, edges, z, frac):
    """True if the cut plane would saw through any tack flat. Checking the
    flats themselves, not just their corners: on some axes a flat's centre sits
    exactly ON the plane with all its corners off it, and a corner count would
    wave that through."""
    for v in range(len(verts)):
        h = flat_points(verts, edges, v, frac) @ z
        if h.min() < -1e-9 and h.max() > 1e-9:
            return True
    return False


def worst_overhang(part):
    """Measured on the half that will actually be printed, standing on its cut
    face -- not predicted from the whole solid, which misjudges every face the
    plane passes through."""
    t = mesh_faces(part)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    keep = area > 1e-9
    n, t, area = n[keep] / (2 * area[keep])[:, None], t[keep], area[keep]
    bed = t[:, :, 2].min()
    on_bed = np.abs(t[:, :, 2].max(axis=1) - bed) < 1e-6      # the cut face itself
    down = (n[:, 2] < -1e-9) & ~on_bed
    lean = np.degrees(np.arcsin(np.clip(-n[down, 2], 0, 1))) if down.any() else np.array([0.0])
    return float(lean.max()), float(area[down].sum())


def flat_distance(verts, edges, v, frac):
    """How far vertex v's flat sits from the centre, along that vertex's own
    outward direction."""
    r = verts[v] / np.linalg.norm(verts[v])
    return float(flat_points(verts, edges, v, frac)[0] @ r)


def dimple(direction, surface, dia, depth, proud):
    """A cone sunk into one flat, apex inward, carried slightly past the face."""
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    height = depth + proud
    cone = m3d.Manifold.cylinder(height, 0.0, (dia / 2) * height / depth, 64, False)
    rot = rotation_taking(d).T            # takes +Z onto this flat's outward normal
    shift = (d * (surface - depth)).reshape(3, 1)
    return cone.transform(np.hstack([rot, shift]).astype(np.float32))


def with_dimples(whole, verts, edges, frac):
    cones = [dimple(verts[v], flat_distance(verts, edges, v, frac),
                    DIMPLE_DIA, DIMPLE_DEPTH, DIMPLE_PROUD)
             for v in range(len(verts))]
    return whole - m3d.Manifold.batch_boolean(cones, m3d.OpType.Add)


def best_cut(solid, whole, verts, edges, frac):
    """The gentlest cut, among the solid's own symmetry directions, that leaves
    every tack flat whole. Each candidate is actually built and measured, since
    predicting the overhang from the uncut solid gets it wrong."""
    raw = [verts[i] for i in range(len(verts))]
    raw += [(verts[a] + verts[b]) / 2 for a, b in edges]
    raw += [np.mean([verts[i] for i in f], axis=0) for f in solid.faces]
    seen, cands = set(), []
    for ax in raw:                                    # dedupe by direction, up to sign
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        z = np.asarray(ax, float) / n
        key = tuple(np.round(z if z[np.argmax(np.abs(z))] > 0 else -z, 6))
        if key not in seen:
            seen.add(key)
            cands.append(z)
    best = None
    for z in cands:
        if splits_a_flat(verts, edges, z, frac):
            continue
        lean, down_area = worst_overhang(half(whole, z, True))
        if best is None or lean < best[0] - 1e-9:
            best = (lean, z, down_area)
    assert best is not None, "every candidate cut would saw through a tack flat"
    return best


def export_stl(man, path_out):
    mesh = man.to_mesh()
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


def half(man, z, keep_upper=True):
    """Everything on one side of the plane through the origin normal to z."""
    box = m3d.Manifold.cube([BIG, BIG, BIG], True).translate([0, 0, BIG / 2])
    rot = rotation_taking(z)
    # rotate the PART so the cut normal becomes +Z, then trim with a half-space
    turned = man.transform(np.hstack([rot, np.zeros((3, 1))]).astype(np.float32))
    if not keep_upper:
        turned = turned.rotate([180, 0, 0])
    return turned ^ box


if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "icosahedron").lower()
    if name not in SOLIDS:
        sys.exit(f"unknown solid {name!r}; choose one of {', '.join(SOLIDS)}")
    solid = SOLIDS[name]

    verts = scaled_vertices(solid)
    edges = sorted({e for f in solid.faces for e in face_edges(f)})
    whole, corners = truncated(verts, edges, TRUNCATE_FRAC)

    # how big each tack flat comes out
    flats = []
    for v in range(len(verts)):
        pts = [verts[v] + TRUNCATE_FRAC * (verts[u] - verts[v])
               for a, b in edges for u in ([b] if a == v else [a] if b == v else [])]
        flats.append(max(np.linalg.norm(p - q) for p in pts for q in pts))
    across = float(np.linalg.norm(verts, axis=1).max() * 2)
    print(f"{solid.name}: {across:.0f}mm across, edges {np.linalg.norm(verts[edges[0][0]] - verts[edges[0][1]]):.1f}mm")
    print(f"  vertices cut back {TRUNCATE_FRAC * 100:.0f}% of an edge -> "
          f"{len(verts)} flats, {min(flats):.1f}-{max(flats):.1f}mm across")


    # The cut is chosen on the plain solid. Dimpling first would have the
    # search judging 1.6mm cone facets instead of the faces that actually
    # decide print quality, and it picks a worse axis when it does.
    worst, z, down_area = best_cut(solid, whole, verts, edges, TRUNCATE_FRAC)
    assert not splits_a_flat(verts, edges, z, TRUNCATE_FRAC)
    print(f"  cut chosen from {len(verts) + len(edges) + len(solid.faces)} symmetry "
          f"directions: worst overhang {worst:.1f} deg over {down_area:.0f}mm2, "
          f"no tack flat split")
    assert worst < 45.0, f"worst overhang {worst:.1f} deg needs support"

    plain_volume = whole.volume()
    whole = with_dimples(whole, verts, edges, TRUNCATE_FRAC)
    cut_away = plain_volume - whole.volume()
    ideal = len(verts) * math.pi * (DIMPLE_DIA / 2) ** 2 * DIMPLE_DEPTH / 3
    assert abs(cut_away - ideal) < 0.05 * ideal, (
        f"the dimples removed {cut_away:.3f}mm3, not the {ideal:.3f}mm3 that "
        f"{len(verts)} cones of that size should")
    print(f"  centring dimple {DIMPLE_DIA:.1f} x {DIMPLE_DEPTH:.1f}mm in each flat "
          f"({math.degrees(math.atan2(DIMPLE_DIA / 2, DIMPLE_DEPTH)):.1f} deg wall, "
          f"{cut_away:.2f}mm3 removed in all)")

    upper = half(whole, z, True)
    lower = half(whole, z, False)
    n = len(upper.decompose())
    assert n == 1, f"the half is {n} disconnected shells"

    # is the other half the same part? then one file, printed twice
    mid = lambda m: [(m.bounding_box()[i] + m.bounding_box()[i + 3]) / 2 for i in range(2)]
    def mismatch(deg):
        r = lower.translate([-mid(lower)[0], -mid(lower)[1], 0]).rotate([0, 0, float(deg)])
        r = r.translate([mid(upper)[0] - mid(r)[0], mid(upper)[1] - mid(r)[1], 0])
        return (upper - r).volume() + (r - upper).volume()

    best = min((mismatch(d), d) for d in np.arange(0, 360, 1.0))
    step = 1.0                                        # refine: a coarse sweep alone
    for _ in range(7):                                # reads as a mismatch of microns
        step /= 5
        best = min((mismatch(d), d) for d in np.arange(best[1] - 3 * step, best[1] + 3 * step, step))
    same = best[0] / upper.surface_area() < 1e-4
    print(f"  halves match after a {best[1]:.4f} deg turn to "
          f"{best[0] / upper.surface_area() * 1e6:.1f} nanometres of surface -- "
          f"{'ONE part, printed twice' if same else 'two different parts'}")

    bb = upper.bounding_box()
    print(f"  half: {bb[3] - bb[0]:.1f} x {bb[4] - bb[1]:.1f} x {bb[5] - bb[2]:.1f}mm, "
          f"volume {upper.volume() / 1000:.1f}cm3")

    out = f"hardware/tests/peg_half_{name}.stl"
    export_stl(upper, out)
    print(f"wrote {out}" + ("  (print TWO of these)" if same else ""))

    # ---- render -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    BG = "#f7f7f5"
    fig = plt.figure(figsize=(17, 5.0), facecolor=BG)
    views = [(26, -60, "as it prints, cut face down", None),
             (6, -60, "from the side", None),
             (78, -60, "from above", None),
             (34, -60, "one flat, close up", 9.0)]
    t = mesh_faces(upper)
    apex = t.reshape(-1, 3)[np.argmax(t.reshape(-1, 3)[:, 2])]
    for k, (elev, azim, title, zoom) in enumerate(views):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d", facecolor=BG)
        ax.set_proj_type("ortho")
        nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        light = np.array([0.32, 0.45, 0.83])
        lam = np.clip(nrm @ light / np.linalg.norm(light), 0, 1)
        sh = 0.24 + 0.76 * lam
        cols = np.stack([sh * 0.42, sh * 0.55, sh * 0.72, np.ones_like(sh)], 1)
        ax.add_collection3d(Poly3DCollection(t, facecolors=cols, edgecolors="#20304a",
                                             linewidths=0.25))
        p = t.reshape(-1, 3)
        lo, hi = p.min(0), p.max(0)
        if zoom is not None:
            lo, hi = apex - zoom, apex + zoom
        for setter, a, b in ((ax.set_xlim, lo[0], hi[0]), (ax.set_ylim, lo[1], hi[1]),
                             (ax.set_zlim, lo[2], hi[2])):
            setter(a, b)
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, color="0.28", pad=0)
    fig.suptitle(f"{solid.name} — half with flattened vertices, {across:.0f}mm across, "
                 f"{len(verts)} tack flats {min(flats):.1f}mm wide, each with a "
                 f"{DIMPLE_DIA:.1f} × {DIMPLE_DEPTH:.1f}mm centring dimple",
                 fontsize=11.5, color="0.15")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = f"hardware/tests/peg_half_{name}.png"
    fig.savefig(png, dpi=150, facecolor=BG)
    print(f"wrote {png}")
