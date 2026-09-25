"""Generates a print-ready PDF worksheet of a solid's Schlegel diagram -- its
whole edge graph drawn flat on one page, every vertex marked, to be handed
out for tracing Hamiltonian cycles by pencil.

    python3 hardware/scripts/gen_schlegel_pdf.py [solid]

What a Schlegel diagram is: put your eye just outside ONE face and look in
through it. Because the solid is convex, every other vertex and edge appears
inside that face's outline, and no two edges cross. The face you looked
through becomes the unbounded region outside the drawing, so nothing is
lost: every vertex, every edge, every face. It is the surface flattened
without tearing or lying about which vertex touches which -- which is what
makes a cycle traced on the sheet a real cycle on the solid.

There is no per-solid drawing here and none is needed. Two constructions
both work for any convex polyhedron, and the script builds both, checks
both, and keeps whichever draws more clearly:

  PROJECTION -- the literal definition. Put a viewpoint out along a face's
  normal and perspective-project every vertex onto that face's plane. How
  far out is not free: past a limit the eye crosses a neighbouring face's
  plane and it stops being a Schlegel diagram, while near either end of the
  legal range the drawing bunches up. So the range is swept.

  TUTTE -- Tutte's 1963 theorem. Pin the view face's vertices to a convex
  polygon and put every other vertex at the plain average of its neighbours,
  by solving one linear system. For a 3-connected planar graph -- which
  every convex polyhedron's graph is, by Steinitz's theorem -- the result is
  guaranteed crossing-free with every face convex. It needs no geometry at
  all, only which vertex touches which, so it is unbothered by the d10 being
  irregular. It is also what you would draw by hand: it puts a single node
  inside a triangle for the tetrahedron, and a square inside a square for
  the cube.

Where a projection lands its vertices on clean concentric rings, a third
candidate respaces those rings' radii while keeping their angles, since a
literal projection crowds the far side of the solid into the middle.

"More clearly" is measured, not judged: the radius of the largest circle
fitting inside the tightest drawn region. That punishes both cramped regions
and thin slivers, where area or vertex spacing alone each fix one and wreck
the other. Every candidate is put through the two checks that make a drawing
a Schlegel diagram at all -- no two edges cross, and every vertex off the
view face lies strictly inside it -- so a construction that fails is
discarded rather than drawn.

Requires: pip install matplotlib numpy

Run from the repo root.
"""

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from polyhedra import SOLIDS, face_edges

PAGE_W_MM, PAGE_H_MM = 279.4, 215.9
MM_PER_INCH = 25.4
MARGIN_MM = 10     # safe margin on all sides (most printers can't print edge-to-edge)
HEADER_MM = 26     # vertical space reserved for the title block at the top

VIEW_FACE = 0      # which face we look in through; it becomes the outer outline
VERTEX_R_MM = 2.6  # radius of the circle drawn at each vertex

EDGE_STYLE = dict(color="0.35", linestyle="-", linewidth=1.1, zorder=1, solid_capstyle="round")
VERTEX_STYLE = dict(facecolor="white", edgecolor="black", linewidth=1.3, zorder=2)


def all_edges(faces):
    return sorted({e for f in faces for e in face_edges(f)})


def cross2(u, v):
    return u[0] * v[1] - u[1] * v[0]


# ---------------------------------------------------------------- geometry --

def face_plane(face, verts, centre):
    pts = np.array([verts[i] for i in face], dtype=float)
    mid = pts.mean(axis=0)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n /= np.linalg.norm(n)
    return mid, (n if np.dot(mid - centre, n) > 0 else -n)


def max_view_distance(verts, faces, view_face, centroid, normal, centre):
    """Farthest the eye can sit along the face normal and still be 'beyond'
    only that one face -- past it, the projection is not a Schlegel diagram.

    This can legitimately be infinite. Moving out along the cube's bottom
    normal, its four side planes are parallel to the direction of travel and
    its top plane is behind you, so no plane is ever crossed; the tetrahedron
    is the same, every other face leaning away. Those solids are then limited
    only by the drawing going degenerate as the view approaches orthographic
    -- for the cube, the top face projecting exactly onto the bottom one --
    which the sweep finds by measurement.
    """
    best = np.inf
    for fi, f in enumerate(faces):
        if fi == view_face:
            continue
        c_other, n_other = face_plane(f, verts, centre)
        denom = np.dot(normal, n_other)
        if denom <= 1e-12:
            continue                      # moving away from that plane; never crosses
        t = np.dot(c_other - centroid, n_other) / denom
        if t > 0:
            best = min(best, t)
    return best                       # np.inf when nothing bounds it


def eye_range(solid, view_face, span=20.0, steps=70):
    """Eye distances worth trying, in absolute units: from just off the face
    out to either the limit above or `span` circumradii, whichever is nearer.
    Swept geometrically, because the drawing changes fast up close and slowly
    far away."""
    verts = np.array(solid.vertices, dtype=float)
    centre = verts.mean(axis=0)
    centroid, normal = face_plane(solid.faces[view_face], verts, centre)
    radius = float(np.linalg.norm(verts - centre, axis=1).max())
    limit = max_view_distance(verts, solid.faces, view_face, centroid, normal, centre)
    return np.geomspace(0.02 * radius, min(limit * 0.999, span * radius), steps), radius


def projection(solid, view_face, eye_dist):
    verts = np.array(solid.vertices, dtype=float)
    centre = verts.mean(axis=0)
    centroid, normal = face_plane(solid.faces[view_face], verts, centre)
    eye = centroid + eye_dist * normal

    e1 = verts[solid.faces[view_face][1]] - verts[solid.faces[view_face][0]]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    out = []
    for v in verts:
        d = v - eye
        denom = np.dot(d, normal)
        assert denom < -1e-12, "vertex is not behind the view plane"
        hit = eye + (-np.dot(eye - centroid, normal) / denom) * d
        rel = hit - centroid
        out.append((np.dot(rel, e1), np.dot(rel, e2)))
    return np.array(out)


def tutte(solid, view_face):
    """Tutte's barycentric embedding -- see the module docstring."""
    n = len(solid.vertices)
    adj = {v: set() for v in range(n)}
    for a, b in all_edges(solid.faces):
        adj[a].add(b)
        adj[b].add(a)
    outer = solid.faces[view_face]
    pos = np.zeros((n, 2))
    for i, v in enumerate(outer):
        t = 2 * math.pi * i / len(outer) + math.pi / 2
        pos[v] = (math.cos(t), math.sin(t))
    inner = [v for v in range(n) if v not in outer]
    if inner:
        idx = {v: i for i, v in enumerate(inner)}
        a_mat = np.zeros((len(inner), len(inner)))
        rhs = np.zeros((len(inner), 2))
        for v in inner:
            a_mat[idx[v], idx[v]] = len(adj[v])
            for u in adj[v]:
                if u in idx:
                    a_mat[idx[v], idx[u]] = -1.0
                else:
                    rhs[idx[v]] += pos[u]
        for v, p in zip(inner, np.linalg.solve(a_mat, rhs)):
            pos[v] = p
    return pos


def rings_of(pts2d, tol=6):
    r = np.hypot(pts2d[:, 0], pts2d[:, 1])
    levels = sorted({round(float(v), tol) for v in r}, reverse=True)
    return np.array([levels.index(round(float(v), tol)) for v in r]), \
        np.arctan2(pts2d[:, 1], pts2d[:, 0]), levels


def on_rings(ring, ang, radii):
    rr = np.array([radii[k] for k in ring])
    return np.column_stack([rr * np.cos(ang), rr * np.sin(ang)])


# ------------------------------------------------------------ measurement --

def min_inradius(pts2d, faces, outer_face):
    """Largest circle fitting inside the tightest drawn region, as a fraction
    of the drawing's width. Works for any polygon, not just triangles."""
    worst = np.inf
    for fi, f in enumerate(faces):
        if fi == outer_face:
            continue                      # this one is the page, not a region
        p = np.array([pts2d[i] for i in f])
        k = len(p)
        area = abs(sum(cross2(p[i], p[(i + 1) % k]) for i in range(k))) / 2
        per = sum(np.linalg.norm(p[i] - p[(i + 1) % k]) for i in range(k))
        worst = min(worst, 2 * area / per)
    return worst / (pts2d[:, 0].max() - pts2d[:, 0].min())


def segments_cross(p, q, r, s):
    """Proper crossing only -- edges of a graph may meet at shared vertices."""
    if {tuple(np.round(p, 9)), tuple(np.round(q, 9))} & \
       {tuple(np.round(r, 9)), tuple(np.round(s, 9))}:
        return False
    side = lambda a, b, c: np.sign(cross2(b - a, c - a))
    return (side(r, s, p) * side(r, s, q) < 0) and (side(p, q, r) * side(p, q, s) < 0)


def verify(pts2d, faces, edges, outer_face):
    bad = [(a, b) for i, a in enumerate(edges) for b in edges[i + 1:]
           if segments_cross(pts2d[a[0]], pts2d[a[1]], pts2d[b[0]], pts2d[b[1]])]
    assert not bad, f"{len(bad)} crossing edge pairs -- not a Schlegel diagram"
    hull = [pts2d[i] for i in faces[outer_face]]
    k = len(hull)
    for i, p in enumerate(pts2d):
        if i in faces[outer_face]:
            continue
        signs = [cross2(hull[(j + 1) % k] - hull[j], p - hull[j]) for j in range(k)]
        assert all(s > 0 for s in signs) or all(s < 0 for s in signs), \
            f"vertex {i} projected outside the outer face"


def score(pts2d, solid, edges):
    try:
        verify(pts2d, solid.faces, edges, VIEW_FACE)
    except AssertionError:
        return -np.inf
    return min_inradius(pts2d, solid.faces, VIEW_FACE)


# -------------------------------------------------------------- candidates --

def candidates(solid, edges):
    out = {}
    dists, radius = eye_range(solid, VIEW_FACE)
    best = max(((score(projection(solid, VIEW_FACE, t), solid, edges), t) for t in dists),
               key=lambda t: t[0])
    if np.isfinite(best[0]):
        proj = projection(solid, VIEW_FACE, best[1])
        out[f"projection (eye {best[1] / radius:.2f} radii out)"] = proj

        ring, ang, radii = rings_of(proj)
        sizes = [int((ring == k).sum()) for k in range(len(radii))]
        if 2 <= len(radii) <= 6 and len(set(sizes)) == 1:
            r = [v / radii[0] for v in radii]
            top = score(on_rings(ring, ang, r), solid, edges)
            for _ in range(60):                       # coordinate ascent on the radii
                moved = False
                for i in range(1, len(r)):
                    for d in (0.08, -0.08, 0.02, -0.02, 0.005, -0.005):
                        cand = list(r)
                        cand[i] += d
                        if not 0 < cand[i] < 1:
                            continue
                        s = score(on_rings(ring, ang, cand), solid, edges)
                        if s > top:
                            top, r, moved = s, cand, True
                if not moved:
                    break
            out[f"projection, {len(radii)} rings respaced"] = on_rings(ring, ang, r)

    t = tutte(solid, VIEW_FACE)
    if np.isfinite(score(t, solid, edges)):
        out["Tutte embedding"] = t
    return out


def upright(pts2d, apex):
    turn = math.pi / 2 - math.atan2(pts2d[apex][1], pts2d[apex][0])
    c, s = math.cos(turn), math.sin(turn)
    return pts2d @ np.array([[c, s], [-s, c]])


if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "icosahedron").lower()
    if name not in SOLIDS:
        sys.exit(f"unknown solid {name!r}; choose one of {', '.join(SOLIDS)}")
    solid = SOLIDS[name]
    edges = all_edges(solid.faces)
    v, e, f = len(solid.vertices), len(edges), len(solid.faces)
    assert v - e + f == 2, f"Euler characteristic is {v - e + f}, not 2"

    options = candidates(solid, edges)
    assert options, f"no construction produced a valid diagram for {solid.name}"
    print(f"{solid.name}: {v} vertices, {e} edges, {f} faces "
          f"({f - 1} drawn + 1 as the page)")
    for label, pts in options.items():
        print(f"   {label:38s} tightest region {score(pts, solid, edges) * 100:5.2f}% of width")
    label = max(options, key=lambda k: score(options[k], solid, edges))
    pts = upright(options[label], solid.faces[VIEW_FACE][0])
    verify(pts, solid.faces, edges, VIEW_FACE)
    print(f"   -> drawing the {label}")

    usable_w = PAGE_W_MM - 2 * MARGIN_MM - 2 * VERTEX_R_MM
    usable_h = PAGE_H_MM - 2 * MARGIN_MM - HEADER_MM - 2 * VERTEX_R_MM
    scale = min(usable_w / np.ptp(pts[:, 0]), usable_h / np.ptp(pts[:, 1]))
    pts = pts * scale
    pts = pts + np.array([PAGE_W_MM / 2 - (pts[:, 0].max() + pts[:, 0].min()) / 2,
                          (MARGIN_MM + usable_h / 2 + VERTEX_R_MM)
                          - (pts[:, 1].max() + pts[:, 1].min()) / 2])

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
    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 15, f"Hamiltonian Polyhedron - {solid.name}",
            ha="center", fontsize=11, weight="bold")
    ax.text(PAGE_W_MM / 2, PAGE_H_MM - 21, "Schlegel Diagram",
            ha="center", fontsize=9, style="italic", color="0.3")

    out = f"hardware/tests/schlegel_{name}.pdf"
    with PdfPages(out) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"wrote {out}")
