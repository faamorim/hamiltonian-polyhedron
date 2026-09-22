"""Generates a print-ready PDF template of both caps' flat, unfolded nets --
wireframe only (no 3D relief), for cutting out of paper/cardboard and
folding by hand as a cheap side-prototype of the folding geometry.

Solid lines are each strip's outer silhouette (cut here). Dashed lines are
the internal edges between faces (fold here, mountain-fold so the strip
curls toward you the way the printed 3D version will).

Must be printed at 100% / "Actual Size" -- NOT "Fit to page". The edge
length is auto-scaled to the largest size that fits both nets on one
landscape sheet (with a safety margin) -- this is a standalone
paper/cardboard test of the fold geometry, not meant to match the 3D
prototype's scale (bigger is easier to fold and see clearly by hand).

The two caps' flat nets are congruent (verified separately: rotating one by
120 degrees maps it onto the other with zero error, even though no rigid
motion of the whole icosahedron maps cap 0's actual faces onto cap 1's --
unfolding keeps only the local turn pattern, not which global vertices are
involved). Since they're the same shape, this is a genuine 2-piece nesting
problem: cap 1 is rendered pre-rotated into the SAME orientation as cap 0
(undoing its natural 120-degree offset), then the two are slid together
(translation search only, rotation fixed at the now-optimal 0 degrees
relative) to interlock rather than just sit side by side with a fixed gap.

Requires: pip install matplotlib numpy shapely

Run from the repo root:
    python3 hardware/scripts/gen_flat_strip_pdf.py
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry import Polygon
from shapely.affinity import rotate as shapely_rotate, translate as shapely_translate
from shapely.ops import unary_union

from gen_flat_strip import (
    FACES, CYCLE,
    internal_adjacency, split_into_caps, walk_path, unfold, edge_key,
)

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
    return (sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3)


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
    tris = [Polygon([p for _, p in face_2d[fi]]) for fi in path]
    return unary_union(tris)


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


def draw_net(ax, path, face_2d, rotate_deg, pivot, ox, oy):
    c, s = math.cos(math.radians(rotate_deg)), math.sin(math.radians(rotate_deg))
    px, py = pivot

    def to_page(p):
        x, y = p[0] - px, p[1] - py
        return (x * c - y * s + px + ox, x * s + y * c + py + oy)

    fold_edges = set()
    for i in range(len(path) - 1):
        fa, fb = path[i], path[i + 1]
        shared = [v for v in FACES[fa] if v in FACES[fb]]
        fold_edges.add(edge_key(*shared))

    def pt_of(fi, gv):
        for g, p in face_2d[fi]:
            if g == gv:
                return to_page(p)

    drawn = set()  # keyed by edge_id ALONE, so a shared edge is drawn once
    # total, not once per owning face (drawing it twice, from each face's
    # own point order, put the dash pattern out of phase between the two
    # copies -- overlaid, they filled each other's gaps and looked solid)
    for fi in path:
        verts = [gv for gv, _ in face_2d[fi]]
        for k in range(3):
            a, b = verts[k], verts[(k + 1) % 3]
            eid = edge_key(a, b)
            if eid in drawn:
                continue
            drawn.add(eid)
            pa, pb = pt_of(fi, a), pt_of(fi, b)
            is_fold = eid in fold_edges
            style = dict(color="0.5", linestyle=(0, (4, 3)), linewidth=1.1) if is_fold \
                else dict(color="black", linestyle="-", linewidth=1.6)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], **style)


if __name__ == "__main__":
    adj = internal_adjacency(FACES, CYCLE)
    caps = split_into_caps(FACES, adj)
    paths = [walk_path(cap, adj) for cap in caps]

    usable_w = PAGE_W_MM - 2 * MARGIN_MM
    usable_h = PAGE_H_MM - 2 * MARGIN_MM - HEADER_MM

    # search for the tightest nesting at unit (1mm edge) scale, then solve
    # for the actual edge length once
    unit_2ds = [unfold(p, 1.0) for p in paths]
    align_deg = alignment_rotation_deg(paths[0], unit_2ds[0], paths[1], unit_2ds[1])

    poly0 = net_polygon(paths[0], unit_2ds[0])
    cx1, cy1 = centroid(unit_2ds[1], paths[1][0])  # pivot for cap1's rotation
    poly1_aligned = shapely_rotate(net_polygon(paths[1], unit_2ds[1]), align_deg, origin=(cx1, cy1))

    best = best_nesting_offset(poly0, poly1_aligned, usable_w, usable_h, CUT_GAP_MM)
    edge_len, off_dx, off_dy = best
    print(f"alignment rotation for cap1: {align_deg:.2f} deg")
    print(f"using edge length {edge_len:.1f}mm (nested fit, {PAGE_W_MM:.0f}x{PAGE_H_MM:.0f}mm landscape)")

    face_2ds = [unfold(p, edge_len) for p in paths]

    # Do ALL placement bookkeeping in shapely (rotate/translate/bounds), then
    # hand draw_net just the final (rotation, pivot, offset) each net needs --
    # avoids re-deriving rotated coordinates by hand a second time.
    poly0 = net_polygon(paths[0], face_2ds[0])  # cap0 stays at its natural, unrotated placement

    rot_pivot = centroid(face_2ds[1], paths[1][0])
    poly1_rotated = shapely_rotate(net_polygon(paths[1], face_2ds[1]), align_deg, origin=rot_pivot)
    # slot cap1 (now in cap0's orientation) into its nested position
    slot_x, slot_y = off_dx * edge_len, off_dy * edge_len
    r1minx, r1miny, _, _ = poly1_rotated.bounds
    cap1_dx, cap1_dy = slot_x - r1minx, slot_y - r1miny
    poly1_final = shapely_translate(poly1_rotated, xoff=cap1_dx, yoff=cap1_dy)

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

    draw_net(ax, paths[0], face_2ds[0], 0, (0, 0), center_ox, center_oy)
    draw_net(ax, paths[1], face_2ds[1], align_deg, rot_pivot, cap1_dx + center_ox, cap1_dy + center_oy)

    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 15,
             "Hamiltonian Polyhedron - Icosahedron",
             ha="center", fontsize=11, weight="bold")

    with PdfPages("hardware/tests/flat_strip_template.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print("wrote hardware/tests/flat_strip_template.pdf")
