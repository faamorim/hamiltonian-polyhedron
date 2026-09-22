"""Generates a print-ready PDF worksheet of the icosahedron's Schlegel diagram
-- the whole solid's edge graph drawn flat on one page, with every vertex
marked, so it can be handed out for tracing Hamiltonian cycles by pencil.

What a Schlegel diagram is: put your eye just outside ONE face of the solid
and look in through it. Because the solid is convex, every other vertex and
edge appears inside that face's outline, and no two edges cross. The face
you looked through becomes the unbounded region outside the drawing -- so
all 20 faces are present (19 drawn, 1 as the surrounding page), all 12
vertices, all 30 edges. It is the solid's surface flattened without tearing
or lying about which vertex touches which.

That is exactly what this script does, literally: it places a viewpoint a
short distance out along one face's normal and perspective-projects all 12
vertices onto that face's plane. Nothing about the layout is hand-drawn or
faked -- move the viewpoint and the drawing breathes like the real solid.

How far out the eye sits is not a free choice of taste. It is measured as
a fraction of the largest distance that still keeps the eye "beyond" only
that one face -- past that limit it crosses a neighbouring face's plane and
the projection stops being a Schlegel diagram at all. Both ends of the
legal range draw badly: near 0 the far side of the solid collapses into a
speck in the middle, and near the limit the three vertices behind the view
face flatten onto the outer triangle's own sides, leaving pairs of edges
almost on top of each other. So the script sweeps the range and keeps the
viewpoint that maximises the smallest gap between a vertex and an edge it
does not belong to -- the one where a pencil line can most easily tell two
edges apart.

Even at its best, though, a literal perspective view bunches the icosahedron
up: its 12 vertices land on four concentric rings of 3, at exact 30/90/150/
210/270/330-degree angles, but at radii 1.155 / 0.411 / 0.277 / 0.112 --
three quarters of the vertices crammed into the middle third. So the script
keeps the angles and the ring membership the projection found, and respaces
only the RADII, choosing them to maximise the inradius of the tightest of
the 19 drawn triangles (the biggest circle that fits in the worst region --
a number that punishes both cramped regions and thin slivers, unlike area
or vertex spacing alone, which each fix one and wreck the other).

That respacing is a redrawing, not a different solid: every candidate is put
back through the same two checks as the raw projection -- no two edges cross,
and every vertex not on the view face lies strictly inside its triangle --
so the result is still a planar straight-line drawing of the icosahedron's
graph with the same face on the outside, which is what a Schlegel diagram is
for. It is the textbook concentric-ring picture, with the ring radii earned
rather than eyeballed.

Must be printed at 100% / "Actual Size", same as the strip template, so the
two sheets belong to the same set.

Requires: pip install matplotlib numpy

Run from the repo root:
    python3 hardware/scripts/gen_schlegel_pdf.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gen_flat_strip import VERTICES, FACES, edge_key

# Landscape US Letter -- matches hardware/scripts/gen_flat_strip_pdf.py so the
# two worksheets print as a set.
PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
MM_PER_INCH = 25.4
MARGIN_MM = 10     # safe margin on all sides (most printers can't print edge-to-edge)
HEADER_MM = 26     # vertical space reserved for the title block at the top

VIEW_FACE = 0      # which face we look in through; it becomes the outer triangle

VERTEX_R_MM = 2.6  # radius of the circle drawn at each vertex

EDGE_STYLE = dict(color="0.35", linestyle="-", linewidth=1.1, zorder=1, solid_capstyle="round")
VERTEX_STYLE = dict(facecolor="white", edgecolor="black", linewidth=1.3, zorder=2)


def face_plane(face, verts, center):
    """Centroid and outward unit normal of one face."""
    p = np.array([verts[i] for i in face], dtype=float)
    centroid = p.mean(axis=0)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    n /= np.linalg.norm(n)
    if np.dot(centroid - center, n) < 0:
        n = -n
    return centroid, n


def max_view_distance(verts, faces, view_face, centroid, normal, center):
    """Largest distance along the face normal that keeps the eye strictly
    inside every OTHER face's plane -- the condition for the projection to be
    a genuine Schlegel diagram rather than a view that sees round the side."""
    best = np.inf
    for fi, f in enumerate(faces):
        if fi == view_face:
            continue
        c_other, n_other = face_plane(f, verts, center)
        # eye(t) = centroid + t*normal; it leaves this face's half-space when
        # dot(eye - c_other, n_other) = 0.
        denom = np.dot(normal, n_other)
        if denom <= 1e-12:
            continue  # moving away from / parallel to that plane: never crosses
        t = np.dot(c_other - centroid, n_other) / denom
        if t > 0:
            best = min(best, t)
    assert np.isfinite(best), "no bounding face found -- solid is not closed?"
    return best


def schlegel_2d(verts, faces, view_face, view_frac):
    verts = np.array(verts, dtype=float)
    center = verts.mean(axis=0)
    centroid, normal = face_plane(faces[view_face], verts, center)

    t_max = max_view_distance(verts, faces, view_face, centroid, normal, center)
    eye = centroid + (view_frac * t_max) * normal

    # In-plane orthonormal basis, oriented so the outer triangle points up.
    e1 = verts[faces[view_face][1]] - verts[faces[view_face][0]]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    out = []
    for v in verts:
        d = v - eye
        denom = np.dot(d, normal)
        assert denom < -1e-12, "vertex is not behind the view plane"
        s = -np.dot(eye - centroid, normal) / denom
        hit = eye + s * d
        rel = hit - centroid
        out.append((np.dot(rel, e1), np.dot(rel, e2)))
    return np.array(out)


def rotate_apex_up(pts2d, apex_index):
    """Spin the drawing so one outer vertex sits at the top of the page."""
    ang = np.arctan2(pts2d[apex_index][1], pts2d[apex_index][0])
    turn = np.pi / 2 - ang
    c, s = np.cos(turn), np.sin(turn)
    return pts2d @ np.array([[c, s], [-s, c]])


def rings_and_angles(pts2d):
    """The projection lands the icosahedron's vertices on concentric rings.
    Return each vertex's ring index (0 = outermost) and its polar angle."""
    r = np.hypot(pts2d[:, 0], pts2d[:, 1])
    ang = np.arctan2(pts2d[:, 1], pts2d[:, 0])
    levels = sorted({round(v, 6) for v in r}, reverse=True)
    return np.array([levels.index(round(v, 6)) for v in r]), ang, levels


def place_on_rings(ring, ang, radii):
    rr = np.array([radii[k] for k in ring])
    pts = np.column_stack([rr * np.cos(ang), rr * np.sin(ang)])
    return pts / (pts[:, 0].max() - pts[:, 0].min())   # normalise to unit width


def min_inradius(pts2d, faces, outer_face):
    """Radius of the largest circle fitting inside the tightest drawn region."""
    worst = np.inf
    for fi, (a, b, c) in enumerate(faces):
        if fi == outer_face:
            continue          # this one is the unbounded region, not a drawn triangle
        A, B, C = pts2d[a], pts2d[b], pts2d[c]
        semi = (np.linalg.norm(B - A) + np.linalg.norm(C - B) + np.linalg.norm(A - C)) / 2
        worst = min(worst, abs(cross2(B - A, C - A)) / 2 / semi)
    return worst


def best_ring_radii(ring, ang, radii, faces, outer_face, edges):
    """Respace the rings for legibility, rejecting anything that stops being a
    valid Schlegel drawing (see module docs). Coarse grid, then local refine."""
    def score(cand):
        pts = place_on_rings(ring, ang, cand)
        try:
            verify_planar(pts, edges)
            verify_nesting(pts, faces[outer_face])
        except AssertionError:
            return -1.0
        return min_inradius(pts, faces, outer_face)

    best = [1.0] + [r / radii[0] for r in radii[1:]]
    top = score(best)
    grid = np.arange(0.04, 1.0, 0.02)
    for r1 in grid:
        for r2 in grid[grid < r1]:
            for r3 in grid[grid < r2]:
                cand = [1.0, r1, r2, r3]
                s = score(cand)
                if s > top:
                    top, best = s, cand
    for _ in range(80):
        for i in (1, 2, 3):
            for step in (0.01, -0.01, 0.002, -0.002, 0.0005, -0.0005):
                cand = list(best)
                cand[i] += step
                if not 1.0 > cand[1] > cand[2] > cand[3] > 0:
                    continue
                s = score(cand)
                if s > top:
                    top, best = s, cand
    return best, top


def min_vertex_edge_gap(pts2d, edges):
    """Smallest distance from a vertex to an edge it is not an endpoint of,
    as a fraction of the drawing's width. This is the legibility bottleneck:
    when it is small, two edges run so close together that a hand-drawn line
    cannot follow one without touching the other."""
    worst = np.inf
    for a, b in edges:
        A, B = pts2d[a], pts2d[b]
        d = B - A
        for i, P in enumerate(pts2d):
            if i in (a, b):
                continue
            t = np.clip(np.dot(P - A, d) / np.dot(d, d), 0.0, 1.0)
            worst = min(worst, np.linalg.norm(P - (A + t * d)))
    return worst / (pts2d[:, 0].max() - pts2d[:, 0].min())


def best_view_frac(verts, faces, view_face, edges):
    """Pick the viewpoint that draws the clearest diagram (see module docs)."""
    candidates = np.arange(0.15, 0.95, 0.01)
    return max(candidates,
               key=lambda f: min_vertex_edge_gap(schlegel_2d(verts, faces, view_face, f), edges))


def all_edges(faces):
    return sorted({edge_key(f[k], f[(k + 1) % 3]) for f in faces for k in range(3)})


def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


def segments_cross(p, q, r, s):
    """True if open segments pq and rs properly cross (shared endpoints don't
    count -- edges of a graph are allowed to meet at vertices)."""
    def side(a, b, c):
        return np.sign(cross2(b - a, c - a))
    if len({tuple(p), tuple(q)} & {tuple(r), tuple(s)}):
        return False
    d1, d2 = side(r, s, p), side(r, s, q)
    d3, d4 = side(p, q, r), side(p, q, s)
    return d1 * d2 < 0 and d3 * d4 < 0


def verify_planar(pts2d, edges):
    """A Schlegel diagram has no crossings; if any two edges cross, the
    viewpoint was wrong and the drawing would be a lie about the solid."""
    bad = [(a, b) for i, a in enumerate(edges) for b in edges[i + 1:]
           if segments_cross(pts2d[a[0]], pts2d[a[1]], pts2d[b[0]], pts2d[b[1]])]
    assert not bad, f"{len(bad)} crossing edge pairs -- not a valid Schlegel diagram"


def verify_nesting(pts2d, outer_face):
    """Every vertex not on the view face must land strictly inside its triangle."""
    tri = [pts2d[i] for i in outer_face]
    for i, p in enumerate(pts2d):
        if i in outer_face:
            continue
        signs = [cross2(tri[(k + 1) % 3] - tri[k], p - tri[k]) for k in range(3)]
        assert all(s > 0 for s in signs) or all(s < 0 for s in signs), \
            f"vertex {i} projected outside the outer triangle"


if __name__ == "__main__":
    edges = all_edges(FACES)
    assert len(VERTICES) == 12 and len(FACES) == 20 and len(edges) == 30
    assert len(VERTICES) - len(edges) + len(FACES) == 2, "Euler characteristic is not 2"
    degree = [sum(v in e for e in edges) for v in range(len(VERTICES))]
    assert degree == [5] * 12, f"icosahedron must be 5-regular, got {degree}"

    view_frac = best_view_frac(VERTICES, FACES, VIEW_FACE, edges)
    pts = schlegel_2d(VERTICES, FACES, VIEW_FACE, view_frac)
    pts = rotate_apex_up(pts, FACES[VIEW_FACE][0])
    verify_planar(pts, edges)
    verify_nesting(pts, FACES[VIEW_FACE])
    print(f"valid Schlegel diagram: {len(VERTICES)} vertices, {len(edges)} edges, "
          f"{len(FACES)} faces (19 drawn + 1 outer), no crossings")
    print(f"viewpoint at {view_frac:.2f} of the maximum valid distance")

    ring, ang, radii = rings_and_angles(pts)
    assert len(radii) == 4 and all(np.sum(ring == k) == 3 for k in range(4)), \
        "expected four concentric rings of three vertices"
    before = min_inradius(place_on_rings(ring, ang, radii), FACES, VIEW_FACE)
    radii, after = best_ring_radii(ring, ang, radii, FACES, VIEW_FACE, all_edges(FACES))
    pts = place_on_rings(ring, ang, radii)
    verify_planar(pts, edges)
    verify_nesting(pts, FACES[VIEW_FACE])
    print(f"ring radii respaced to {[round(r, 3) for r in radii]}: tightest region "
          f"grew from {before * 100:.2f}% to {after * 100:.2f}% of the drawing width; "
          f"closest a vertex comes to an edge it is not on: "
          f"{min_vertex_edge_gap(pts, edges) * 100:.1f}%")

    # Scale to the largest size that fits the page, leaving room for the vertex
    # circles (which stick out past the raw edge geometry by their radius).
    usable_w = PAGE_W_MM - 2 * MARGIN_MM - 2 * VERTEX_R_MM
    usable_h = PAGE_H_MM - 2 * MARGIN_MM - HEADER_MM - 2 * VERTEX_R_MM
    span_x = pts[:, 0].max() - pts[:, 0].min()
    span_y = pts[:, 1].max() - pts[:, 1].min()
    scale = min(usable_w / span_x, usable_h / span_y)
    pts = pts * scale
    print(f"drawing spans {span_x * scale:.1f} x {span_y * scale:.1f}mm")

    ox = PAGE_W_MM / 2 - (pts[:, 0].max() + pts[:, 0].min()) / 2
    oy = (MARGIN_MM + usable_h / 2 + VERTEX_R_MM) - (pts[:, 1].max() + pts[:, 1].min()) / 2
    pts = pts + np.array([ox, oy])

    fig = plt.figure(figsize=(PAGE_W_MM / MM_PER_INCH, PAGE_H_MM / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, PAGE_W_MM)
    ax.set_ylim(0, PAGE_H_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], **EDGE_STYLE)
    for x, y in pts:
        ax.add_patch(plt.Circle((x, y), VERTEX_R_MM, **VERTEX_STYLE))

    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 15,
            "Hamiltonian Polyhedron - Icosahedron",
            ha="center", fontsize=11, weight="bold")
    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 21,
            "Schlegel Diagram",
            ha="center", fontsize=9, style="italic", color="0.3")

    with PdfPages("hardware/tests/schlegel_icosahedron.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print("wrote hardware/tests/schlegel_icosahedron.pdf")
