"""Generates a print-ready PDF template of both caps' flat, unfolded nets --
wireframe only (no 3D relief), for cutting out of paper/cardboard and
folding by hand as a cheap side-prototype of the folding geometry.

Solid lines are each strip's outer silhouette (cut here). Dashed lines are
the internal edges between faces (fold here, mountain-fold so the strip
curls toward you the way the printed 3D version will).

Must be printed at 100% / "Actual Size" -- NOT "Fit to page" -- since the
triangle edges are drawn at true millimeter scale, matching the 3D-printed
prototype from gen_flat_strip.py exactly.

Requires: pip install matplotlib

Run from the repo root:
    python3 hardware/scripts/gen_flat_strip_pdf.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gen_flat_strip import (
    FACES, CYCLE, TARGET_EDGE,
    internal_adjacency, split_into_caps, walk_path, unfold, edge_key,
)

PAGE_W_MM, PAGE_H_MM = 215.9, 279.4  # US Letter; swap for A4 (210 x 297) if you prefer
MM_PER_INCH = 25.4
GAP_MM = 15  # horizontal gap between the two strips on the page


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


if __name__ == "__main__":
    adj = internal_adjacency(FACES, CYCLE)
    caps = split_into_caps(FACES, adj)
    paths = [walk_path(cap, adj) for cap in caps]
    face_2ds = [unfold(p, TARGET_EDGE) for p in paths]
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
    y_center = (PAGE_H_MM - 30) / 2  # leave room for the header at the top
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
             f"Hamiltonian Polyhedron -- flat nets, both caps ({TARGET_EDGE:.0f}mm edges)",
             ha="center", fontsize=11, weight="bold")
    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 22,
             "Print at 100% / Actual Size (not \"Fit to page\"). "
             "Solid = cut. Dashed = fold (mountain fold, toward you).",
             ha="center", fontsize=8, color="0.3")

    with PdfPages("hardware/tests/flat_strip_template.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print("wrote hardware/tests/flat_strip_template.pdf")
