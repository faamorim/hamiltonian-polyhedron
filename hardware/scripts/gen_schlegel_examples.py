"""Three Schlegel diagrams of one solid for a slide: a Hamiltonian cycle, a
Hamiltonian path that is not a cycle, and a start that traps itself.

    python3 hardware/scripts/gen_schlegel_examples.py [solid]

Made to be shown AFTER a class has tried tracing cycles by hand, so that the
three things that actually happen at the desks all appear on one slide and
nobody's attempt reads as simply wrong:

  CYCLE   every vertex once and back to the start. The cut this project is
          built on.
  PATH    every vertex once, but the two ends are not joined by an edge, so
          it cannot close. Complete as a walk, useless as a cut -- and worth
          showing, because it is the near miss.
  TRAPPED stuck with vertices still unvisited: every neighbour of where the
          pencil stopped has already been used. Nothing is wrong with any
          single step; the route was decided several steps earlier.

The layout is the same construction gen_schlegel_pdf.py draws, imported
rather than repeated, so the diagram on the slide is the diagram on the
worksheet. Every route is searched for in the real edge graph and checked
against it before it is drawn -- no route here is hand-drawn or assumed.

Requires: pip install matplotlib numpy

Run from the repo root.
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polyhedra import SOLIDS
import gen_schlegel_pdf as S

INK = "#1c2026"
FAINT = "#c3c8d2"
CYCLE_C = "#1f8a70"
PATH_C = "#2f6fb5"
STUCK_C = "#c0392b"


def neighbours(solid):
    adj = {i: set() for i in range(len(solid.vertices))}
    for a, b in S.all_edges(solid.faces):
        adj[a].add(b)
        adj[b].add(a)
    return adj


def hamiltonian_path(adj, n):
    """A path over every vertex whose ends are NOT adjacent, so that it is
    genuinely not a cycle rather than a cycle drawn with one edge missing."""
    best = None

    def walk(path, seen):
        nonlocal best
        if best:
            return
        if len(path) == n:
            if path[-1] not in adj[path[0]]:
                best = list(path)
            return
        for nb in sorted(adj[path[-1]]):
            if nb not in seen:
                walk(path + [nb], seen | {nb})
                if best:
                    return

    for start in range(n):
        walk([start], {start})
        if best:
            return best
    return None


def trapped_walk(adj, n, want_left=2):
    """A walk that cannot be continued while vertices remain. Preferred short
    and leaving few vertices out, because a route that dies two steps from
    the end makes the point better than one that dies immediately."""
    found = []

    def walk(path, seen):
        if len(found) > 400:
            return
        end = path[-1]
        if len(path) < n and not (adj[end] - seen):
            found.append(list(path))
            return
        for nb in sorted(adj[end]):
            if nb not in seen:
                walk(path + [nb], seen | {nb})

    for start in range(n):
        walk([start], {start})
    if not found:
        return None
    exact = [p for p in found if n - len(p) == want_left]
    return max(exact or found, key=len)


def check(route, adj, n, closed):
    assert len(set(route)) == len(route), "a vertex is visited twice"
    for a, b in zip(route, route[1:]):
        assert b in adj[a], f"{a}-{b} is not an edge of the solid"
    if closed:
        assert route[-1] in adj[route[0]], "the loop does not close on an edge"
        assert len(route) == n, "the cycle misses a vertex"


def draw(ax, pts, edges, route, colour, closed, visited_all, trapped=False):
    for a, b in edges:
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=FAINT, lw=1.6, zorder=1, solid_capstyle="round")
    seq = route + [route[0]] if closed else route
    for a, b in zip(seq, seq[1:]):
        ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                color=colour, lw=5.0, zorder=2, solid_capstyle="round")
    seen = set(route)
    for i, (x, y) in enumerate(pts):
        on = i in seen
        ax.add_patch(plt.Circle((x, y), 0.030,
                                facecolor=colour if on else "white",
                                edgecolor=colour if on else STUCK_C,
                                linewidth=2.0 if on else 2.4,
                                linestyle="-" if on else (0, (1.6, 1.4)),
                                zorder=3))
    def label(v, text):
        """Pushed straight out from the middle of the drawing, so it never
        lands on the vertex it names or on an edge running past it."""
        x, y = pts[v]
        d = np.array([x, y], float)
        d = d / (np.linalg.norm(d) or 1.0)
        lx, ly = x + d[0] * 0.105, y + d[1] * 0.105
        ax.text(lx, ly, text, ha="center", va="center", fontsize=12,
                color=colour, weight="bold", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.92))

    if not closed:
        x, y = pts[route[0]]
        ax.add_patch(plt.Circle((x, y), 0.052, facecolor="none",
                                edgecolor=colour, linewidth=2.2, zorder=4))
        label(route[0], "start")
        ex, ey = pts[route[-1]]
        if trapped:
            d = 0.030
            for s in (1, -1):
                ax.plot([ex - d, ex + d], [ey - s * d, ey + s * d],
                        color="white", lw=2.6, zorder=4, solid_capstyle="round")
            label(route[-1], "stuck")
        else:
            label(route[-1], "end")
    ax.set_aspect("equal")
    ax.axis("off")
    # Room for the labels and for a dashed marker sitting on the outline.
    # From the drawing's own box, not symmetric about the origin: a Schlegel
    # diagram is a triangle here, so symmetric limits hang it off centre and
    # waste half the panel.
    pad = 0.17
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)


if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "icosahedron").lower()
    solid = SOLIDS[name]
    edges = S.all_edges(solid.faces)
    n = len(solid.vertices)
    adj = neighbours(solid)

    options = S.candidates(solid, edges)
    label = max(options, key=lambda k: S.score(options[k], solid, edges))
    pts = S.upright(options[label], solid.faces[S.VIEW_FACE][0])
    S.verify(pts, solid.faces, edges, S.VIEW_FACE)
    pts = pts / max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    pts = pts - pts.mean(0)

    cycle = list(solid.cycle)
    path = hamiltonian_path(adj, n)
    stuck = trapped_walk(adj, n)
    check(cycle, adj, n, closed=True)
    check(path, adj, n, closed=False)
    check(stuck, adj, n, closed=False)
    assert len(path) == n and path[-1] not in adj[path[0]]
    assert len(stuck) < n and not (adj[stuck[-1]] - set(stuck))

    print(f"{solid.name}: {n} vertices, {len(edges)} edges, drawing the {label}")
    print(f"  cycle   {len(cycle)} vertices, closes")
    print(f"  path    {len(path)} vertices, ends {path[0]} and {path[-1]} "
          f"are not adjacent")
    print(f"  trapped {len(stuck)} vertices, {n - len(stuck)} never reached: "
          f"{sorted(set(range(n)) - set(stuck))}")

    panels = [
        (cycle, CYCLE_C, True, "Hamiltonian cycle",
         "every vertex once, and it closes"),
        (path, PATH_C, False, "Hamiltonian path",
         "every vertex once, but the ends do not meet"),
        (stuck, STUCK_C, False, "trapped",
         f"{n - len(stuck)} vertices never reached, and nowhere left to go"),
    ]
    for i, (route, colour, closed, title, sub) in enumerate(panels):
        fig = plt.figure(figsize=(6.6, 6.4))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.82])
        draw(ax, pts, edges, route, colour, closed, n, trapped=(i == 2))
        fig.text(0.5, 0.955, title, ha="center", fontsize=19,
                 weight="bold", color=INK)
        fig.text(0.5, 0.905, sub, ha="center", fontsize=13, color="#5b6474")
        out = f"hardware/tests/schlegel_{name}_{['cycle','path','trapped'][i]}.png"
        fig.savefig(out, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"  wrote {out}")

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.0))
    for ax, (route, colour, closed, title, sub) in zip(axes, panels):
        draw(ax, pts, edges, route, colour, closed, n,
             trapped=(title == "trapped"))
        ax.set_title(f"{title}\n{sub}", fontsize=15, color=INK, pad=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.05)
    out = f"hardware/tests/schlegel_{name}_outcomes.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")
