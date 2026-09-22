"""Generates a print-ready PDF template of both caps' flat, unfolded nets --
wireframe only (no 3D relief), for cutting out of paper/cardboard and
folding by hand as a cheap side-prototype of the folding geometry.

Solid lines are each strip's outer silhouette (cut here). Dashed lines are
the internal edges between faces (fold here, mountain-fold so the strip
curls toward you the way the printed 3D version will).

Must be printed at 100% / "Actual Size" -- NOT "Fit to page". The edge
length is auto-scaled to the largest size that fits both nets side by side
on one landscape sheet (with a safety margin) -- this is a standalone
paper/cardboard test of the fold geometry, not meant to match the 3D
prototype's scale (bigger is easier to fold and see clearly by hand).

Requires: pip install matplotlib

Run from the repo root:
    python3 hardware/scripts/gen_flat_strip_pdf.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gen_flat_strip import (
    FACES, CYCLE,
    internal_adjacency, split_into_caps, walk_path, unfold, edge_key,
)

# Landscape US Letter -- swap for (297, 210) for landscape A4.
PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
MM_PER_INCH = 25.4
GAP_MM = 15       # horizontal gap between the two strips on the page
MARGIN_MM = 10    # safe margin on all sides (most printers can't print edge-to-edge)
HEADER_MM = 25    # vertical space reserved for the title/instructions at the top


def net_bbox(path, face_2d):
    pts = [p for fi in path for _, p in face_2d[fi]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def draw_net(ax, path, face_2d, ox, oy):
    def to_page(p):
        return (p[0] + ox, p[1] + oy)

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


def largest_edge_that_fits(paths, usable_w, usable_h):
    """Both nets scale linearly with edge length, so measure their footprint
    at 1mm edges and solve directly for the biggest edge length that keeps
    the side-by-side layout within the page."""
    unit_2ds = [unfold(p, 1.0) for p in paths]
    unit_boxes = [net_bbox(paths[i], unit_2ds[i]) for i in range(2)]
    unit_w = [b[1] - b[0] for b in unit_boxes]
    unit_h = [b[3] - b[2] for b in unit_boxes]

    edge_for_width = (usable_w - GAP_MM) / sum(unit_w)
    edge_for_height = usable_h / max(unit_h)
    return min(edge_for_width, edge_for_height)


if __name__ == "__main__":
    adj = internal_adjacency(FACES, CYCLE)
    caps = split_into_caps(FACES, adj)
    paths = [walk_path(cap, adj) for cap in caps]

    usable_w = PAGE_W_MM - 2 * MARGIN_MM
    usable_h = PAGE_H_MM - 2 * MARGIN_MM - HEADER_MM
    edge_len = largest_edge_that_fits(paths, usable_w, usable_h)
    print(f"using edge length {edge_len:.1f}mm (largest that fits {PAGE_W_MM:.0f}x{PAGE_H_MM:.0f}mm landscape)")

    face_2ds = [unfold(p, edge_len) for p in paths]
    bboxes = [net_bbox(paths[i], face_2ds[i]) for i in range(2)]
    widths = [b[1] - b[0] for b in bboxes]
    heights = [b[3] - b[2] for b in bboxes]

    total_w = sum(widths) + GAP_MM
    max_h = max(heights)

    fig = plt.figure(figsize=(PAGE_W_MM / MM_PER_INCH, PAGE_H_MM / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, PAGE_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    start_x = (PAGE_W_MM - total_w) / 2
    y_center = MARGIN_MM + usable_h / 2  # center within the space below the header
    cursor_x = start_x
    for i in range(2):
        xmin, xmax, ymin, ymax = bboxes[i]
        ox = cursor_x - xmin
        oy = y_center - (ymin + ymax) / 2
        draw_net(ax, paths[i], face_2ds[i], ox, oy)
        ax.text(cursor_x + widths[i] / 2, y_center - heights[i] / 2 - 8,
                 f"cap {i}  ({len(paths[i])} faces)", ha="center", fontsize=9, color="0.3")
        cursor_x += widths[i] + GAP_MM

    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 15,
             f"Hamiltonian Polyhedron -- flat nets, both caps ({edge_len:.0f}mm edges)",
             ha="center", fontsize=11, weight="bold")
    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 22,
             "Print at 100% / Actual Size (not \"Fit to page\"). "
             "Solid = cut. Dashed = fold (mountain fold, toward you).",
             ha="center", fontsize=8, color="0.3")

    with PdfPages("hardware/tests/flat_strip_template.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print("wrote hardware/tests/flat_strip_template.pdf")
