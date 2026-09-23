"""Generates a print-ready PDF template of both caps' flat, unfolded nets --
wireframe only (no 3D relief), for cutting out of paper/cardboard and
folding by hand as a cheap side-prototype of the folding geometry.

    python3 hardware/scripts/gen_flat_strip_pdf.py [icosahedron|dodecahedron]

Both solids work the same way: a Hamiltonian cycle on the vertex graph cuts
the surface into two caps, and because every vertex lies on the cut, no
vertex is left inside a cap -- so each cap's faces form a strip that rolls
out flat. The icosahedron gives two 10-triangle strips, the dodecahedron two
6-pentagon ones. (All 30 of the dodecahedron's Hamiltonian cycles split it
that way, so its cycle was picked for how well the two nets nest on a page,
not for its structure.)

Solid lines are each strip's outer silhouette (cut here) -- including the
handle tabs, whose outline is drawn in full, the line where a tab meets its
triangle included, so each tab reads as a distinct piece rather than melting
into the face. That line is deliberately NOT dashed: the tabs are grips, not
anything the polyhedron needs, so there is nothing to fold there. Dashed
lines are the internal edges between faces (fold here, mountain-fold so the
strip curls toward you the way the printed 3D version will).

Must be printed at 100% / "Actual Size" -- NOT "Fit to page". The edge
length is auto-scaled to the largest size that fits both nets on one
landscape sheet (with a safety margin) -- this is a standalone
paper/cardboard test of the fold geometry, not meant to match the 3D
prototype's scale (bigger is easier to fold and see clearly by hand).

Each solid's two caps have congruent nets (verified separately: rotating one
onto the other leaves zero error -- 120 degrees for the icosahedron, 288 for
the dodecahedron -- even though no rigid motion of the whole solid maps one
cap's actual faces onto the other's; unfolding keeps only the local turn
pattern, not which global vertices are involved). Since they are the same
shape, this is a genuine 2-piece nesting problem: cap 1 is rendered
pre-rotated into the SAME orientation as cap 0, then the two are slid
together (translation search only, rotation fixed at the now-optimal 0
degrees relative) to interlock rather than just sit side by side.

Requires: pip install matplotlib numpy shapely

Run from the repo root.
"""

import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate
from shapely.ops import unary_union

from polyhedra import SOLIDS, unfold, compute_fold_edges, edge_key

# Landscape US Letter -- swap for (297, 210) for landscape A4.
PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
MM_PER_INCH = 25.4
MARGIN_MM = 10    # safe margin on all sides (most printers can't print edge-to-edge)
HEADER_MM = 15    # vertical space reserved for the title at the top
CUT_GAP_MM = 3.0  # minimum real clearance wanted between the two cut outlines


def net_bbox(path, face_2d):
    pts = [p for fi in path for _, p in face_2d[fi]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def centroid(face_2d, fi):
    pts = [p for _, p in face_2d[fi]]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def alignment_rotation_deg(path_a, face_2d_a, path_b, face_2d_b):
    """The angle to rotate net B's points by so it lands in the same
    orientation as net A (only meaningful/exact if the two nets are
    congruent -- verified true for cap0/cap1 by the zero-error check this
    reproduces the logic of)."""
    def step(path, face_2d, i):
        ca, cb = centroid(face_2d, path[i]), centroid(face_2d, path[i + 1])
        return (cb[0] - ca[0], cb[1] - ca[1])

    sa, sb = step(path_a, face_2d_a, 0), step(path_b, face_2d_b, 0)
    return math.degrees(math.atan2(sa[1], sa[0]) - math.atan2(sb[1], sb[0]))


def net_polygon(path, face_2d):
    faces = [Polygon([p for _, p in face_2d[fi]]) for fi in path]
    merged = unary_union(faces)
    # Unfolding a strip is not guaranteed to be injective -- a long enough band
    # curls round and can land back on itself, and whether EVERY convex solid
    # has some non-overlapping net is still an open question (Duerer's
    # problem). If it happened here the printed net would be unusable, so
    # check rather than assume: any overlap loses area against the faces.
    loose = sum(f.area for f in faces)
    assert abs(merged.area - loose) < 1e-6 * loose, (
        f"this net overlaps itself ({loose - merged.area:.3f} mm2 of {loose:.3f} lost) "
        f"-- it cannot be cut and folded as drawn")
    return merged


# Handle tabs: small flags a student folds up perpendicular to the paper and
# pinches to hold the assembled bowl, instead of touching the fragile
# curved surface directly (a "stick marionette" rather than gripping the
# model itself). One tab at each extremity of the strip, so a cap is held
# from both of its ends rather than from two adjacent points.
TAB_BASE_FRAC = 0.55   # width where it meets the face, as a fraction of the edge length
TAB_TOP_FRAC = 0.40    # width at the outer tip
TAB_HEIGHT_FRAC = 0.45  # how far the tab protrudes, as a fraction of the edge length


def pick_tab_edges(solid, path, face_2d):
    """One tab at each END of the strip, so a cap is held from both of its
    extremities rather than from two adjacent points. Of an end face's free
    edges, the tab goes on the one NEAREST the strip's own centre, which
    tucks it alongside the body of the net rather than flaring it off the
    far tip -- both easier to grip and tighter to nest on the page.

    Measuring against the strip's centre rather than against the adjoining
    face is not a stylistic choice: an end face is symmetric about the fold
    edge that joins it to the strip, so its free edges are exactly
    equidistant from that neighbour and a rule phrased that way is a tie
    decided by nothing. Against the strip's centre the two differ clearly
    (1.73 vs 2.18 edge lengths on the icosahedron), and picking the nearer
    reproduces all four tab positions that were placed by hand there.
    """
    folds = compute_fold_edges(solid.faces, path)
    strip_centre = np.mean([q for f in path for _, q in face_2d[f]], axis=0)
    chosen = []
    for end in (path[0], path[-1]):
        pts = dict(face_2d[end])
        face = solid.faces[end]
        free = [(face[k], face[(k + 1) % len(face)]) for k in range(len(face))
                if edge_key(face[k], face[(k + 1) % len(face)]) not in folds]
        assert free, f"face {end} has no free edge to put a tab on"
        a, b = min(free, key=lambda ab: np.linalg.norm(
            (np.array(pts[ab[0]]) + np.array(pts[ab[1]])) / 2 - strip_centre))
        chosen.append((end, a, b))
    return chosen


def tab_polygon_points(face_2d, fi, a, b):
    """The tab's 4 outer corners in the same LOCAL (pre-page-transform) 2D
    frame as the rest of that net, so it can go through the same rotate/
    translate pipeline as everything else."""
    pts = dict(face_2d[fi])
    pa, pb = np.array(pts[a]), np.array(pts[b])
    c = np.mean([p for _, p in face_2d[fi]], axis=0)
    edge_len = np.linalg.norm(pb - pa)
    d = (pb - pa) / edge_len
    n = np.array([-d[1], d[0]])
    m = (pa + pb) / 2
    if np.dot(n, m - c) < 0:
        n = -n
    base_w, top_w, h = edge_len * TAB_BASE_FRAC, edge_len * TAB_TOP_FRAC, edge_len * TAB_HEIGHT_FRAC
    base1, base2 = m - d * base_w / 2, m + d * base_w / 2
    tip1, tip2 = m + n * h - d * top_w / 2, m + n * h + d * top_w / 2
    return [tuple(base1), tuple(tip1), tuple(tip2), tuple(base2)]


def net_polygon_with_tabs(path, face_2d, tab_edges):
    poly = net_polygon(path, face_2d)
    for fi, a, b in tab_edges:
        poly = unary_union([poly, Polygon(tab_polygon_points(face_2d, fi, a, b))])
    return poly


def best_nesting_offset(poly_a, poly_b_aligned, usable_w, usable_h, gap_mm, step=0.1):
    """poly_b_aligned is already in the same orientation as poly_a (0 deg
    relative rotation, established separately as optimal for two congruent
    copies of this shape). Slide it around on a grid and keep the
    translation that lets both fit at the largest scale."""
    minxA, minyA, maxxA, maxyA = poly_a.bounds
    minxB, minyB, maxxB, maxyB = poly_b_aligned.bounds
    wA, hA = maxxA - minxA, maxyA - minyA
    wB, hB = maxxB - minxB, maxyB - minyB

    best = None
    for dx in np.arange(-wB - 0.5, wA + 0.5, step):
        for dy in np.arange(-hB - 0.5, hA + 0.5, step):
            cand = shapely_translate(poly_b_aligned, xoff=dx - minxB, yoff=dy - minyB)
            d_unit = poly_a.distance(cand)
            if d_unit <= 1e-6:
                continue
            cb = unary_union([poly_a, cand]).bounds
            cw, ch = cb[2] - cb[0], cb[3] - cb[1]
            if cw <= 0 or ch <= 0:
                continue
            e_pagefit = min(usable_w / cw, usable_h / ch)
            if gap_mm / d_unit > e_pagefit:
                continue
            if best is None or e_pagefit > best[0]:
                best = (e_pagefit, dx, dy)
    return best


FOLD_STYLE = dict(color="0.5", linestyle=(0, (4, 3)), linewidth=1.1)
CUT_STYLE = dict(color="black", linestyle="-", linewidth=1.6)


SOLID_TOL_MM = 1e-6


def point_on_segment(q, seg):
    (x0, y0), (x1, y1) = seg
    dx, dy = x1 - x0, y1 - y0
    span = math.hypot(dx, dy)
    if span < SOLID_TOL_MM:
        return False
    if abs((q[0] - x0) * dy - (q[1] - y0) * dx) / span > SOLID_TOL_MM:
        return False
    t = ((q[0] - x0) * dx + (q[1] - y0) * dy) / span ** 2
    return -SOLID_TOL_MM <= t <= 1 + SOLID_TOL_MM


def verify_no_loose_ends(cut_segments):
    """A cut template is only usable if no solid line simply stops in open
    paper. An end is fine when another solid line ends at the same point OR
    runs through it -- a tab's side ends partway along the edge it is mounted
    on, which is supported without being a shared corner. This is exactly the
    symptom of drawing a tabbed edge as a fold: the tab's sides are then left
    hanging off nothing.
    """
    for i, seg in enumerate(cut_segments):
        for q in seg:
            if not any(point_on_segment(q, other)
                       for j, other in enumerate(cut_segments) if j != i):
                raise AssertionError(f"cut line stops in open paper at {q}")


def draw_net(ax, path, face_2d, fold_edges, rotate_deg, pivot, ox, oy, tab_edges=()):
    c, s = math.cos(math.radians(rotate_deg)), math.sin(math.radians(rotate_deg))
    px, py = pivot

    def to_page(p):
        x, y = p[0] - px, p[1] - py
        return (x * c - y * s + px + ox, x * s + y * c + py + oy)

    def pt_of(fi, gv):
        for g, p in face_2d[fi]:
            if g == gv:
                return to_page(p)

    cut_segments = []
    drawn = set()  # keyed by edge_id ALONE, so a shared edge is drawn once
    # total, not once per owning face (drawing it twice, from each face's
    # own point order, put the dash pattern out of phase between the two
    # copies -- overlaid, they filled each other's gaps and looked solid)
    for fi in path:
        verts = [gv for gv, _ in face_2d[fi]]
        for k in range(len(verts)):
            a, b = verts[k], verts[(k + 1) % len(verts)]
            eid = edge_key(a, b)
            if eid in drawn:
                continue
            drawn.add(eid)
            pa, pb = pt_of(fi, a), pt_of(fi, b)
            if eid in fold_edges:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **FOLD_STYLE)
            else:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **CUT_STYLE)
                cut_segments.append((pa, pb))

    for fi, a, b in tab_edges:
        base1, tip1, tip2, base2 = [to_page(p) for p in tab_polygon_points(face_2d, fi, a, b)]
        for q0, q1 in ((base1, tip1), (tip1, tip2), (tip2, base2)):
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]], **CUT_STYLE)
            cut_segments.append((q0, q1))

    verify_no_loose_ends(cut_segments)


if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "icosahedron").lower()
    if name not in SOLIDS:
        sys.exit(f"unknown solid {name!r}; choose one of {', '.join(SOLIDS)}")
    solid = SOLIDS[name]

    paths = solid.strips()
    fold_edges = [compute_fold_edges(solid.faces, p) for p in paths]
    print(f"{solid.name}: two strips of {len(paths[0])} faces "
          f"({len(solid.faces[paths[0][0]])} sides each)")

    usable_w = PAGE_W_MM - 2 * MARGIN_MM
    usable_h = PAGE_H_MM - 2 * MARGIN_MM - HEADER_MM

    # search for the tightest nesting at unit (1mm edge) scale, then solve
    # for the actual edge length once
    unit_2ds = [unfold(solid.faces, p, 1.0) for p in paths]
    tabs = [pick_tab_edges(solid, paths[i], unit_2ds[i]) for i in (0, 1)]
    align_deg = alignment_rotation_deg(paths[0], unit_2ds[0], paths[1], unit_2ds[1])

    poly0 = net_polygon_with_tabs(paths[0], unit_2ds[0], tabs[0])
    cx1, cy1 = centroid(unit_2ds[1], paths[1][0])  # pivot for cap1's rotation
    poly1_aligned = shapely_rotate(
        net_polygon_with_tabs(paths[1], unit_2ds[1], tabs[1]), align_deg, origin=(cx1, cy1))

    best = best_nesting_offset(poly0, poly1_aligned, usable_w, usable_h, CUT_GAP_MM)
    assert best is not None, "no nesting of the two strips fits on the page"
    edge_len, off_dx, off_dy = best
    print(f"alignment rotation for cap1: {align_deg:.2f} deg")
    print(f"using edge length {edge_len:.1f}mm (nested fit, {PAGE_W_MM:.0f}x{PAGE_H_MM:.0f}mm landscape)")

    face_2ds = [unfold(solid.faces, p, edge_len) for p in paths]

    # Do ALL placement bookkeeping in shapely (rotate/translate/bounds), then
    # hand draw_net just the final (rotation, pivot, offset) each net needs --
    # avoids re-deriving rotated coordinates by hand a second time.
    poly0 = net_polygon_with_tabs(paths[0], face_2ds[0], tabs[0])  # cap0 stays unrotated

    rot_pivot = centroid(face_2ds[1], paths[1][0])
    poly1_rotated = shapely_rotate(
        net_polygon_with_tabs(paths[1], face_2ds[1], tabs[1]), align_deg, origin=rot_pivot)
    # slot cap1 (now in cap0's orientation) into its nested position
    slot_x, slot_y = off_dx * edge_len, off_dy * edge_len
    r1minx, r1miny, _, _ = poly1_rotated.bounds
    cap1_dx, cap1_dy = slot_x - r1minx, slot_y - r1miny
    poly1_final = shapely_translate(poly1_rotated, xoff=cap1_dx, yoff=cap1_dy)
    gap = poly0.distance(poly1_final)
    assert gap >= CUT_GAP_MM - 1e-6, f"strips are only {gap:.2f}mm apart"
    print(f"the two strips nest with {gap:.2f}mm of clear paper between them "
          f"(wanted at least {CUT_GAP_MM:.1f}mm)")

    combined_bounds = unary_union([poly0, poly1_final]).bounds
    cw, ch = combined_bounds[2] - combined_bounds[0], combined_bounds[3] - combined_bounds[1]
    center_ox = (PAGE_W_MM - cw) / 2 - combined_bounds[0]
    center_oy = (MARGIN_MM + usable_h / 2) - (combined_bounds[1] + combined_bounds[3]) / 2

    fig = plt.figure(figsize=(PAGE_W_MM / MM_PER_INCH, PAGE_H_MM / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, PAGE_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_net(ax, paths[0], face_2ds[0], fold_edges[0], 0, (0, 0),
             center_ox, center_oy, tab_edges=tabs[0])
    draw_net(ax, paths[1], face_2ds[1], fold_edges[1], align_deg, rot_pivot,
             cap1_dx + center_ox, cap1_dy + center_oy, tab_edges=tabs[1])


    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 15,
            f"Hamiltonian Polyhedron - {solid.name}",
            ha="center", fontsize=11, weight="bold")

    out = f"hardware/tests/flat_strip_template_{name}.pdf"
    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"wrote {out}")
