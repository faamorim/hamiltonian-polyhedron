"""Draws the seam test tile: one on the bed, two of them closed, and each
bore in section, so the difference between a round ceiling and a teardrop's
ridge is visible before anything is printed.

    python3 hardware/scripts/render_seam_test.py [solid]

The sections are read off the finished mesh the way the printer meets it --
horizontal is into the tile, vertical is layer height -- by slicing the
solid at many heights and asking where the material lies on the line through
the bore. That catches what the design intent cannot: whether a hole really
closes over, and how much of it hangs in the air.
"""

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gen_seam_test as T
import gen_flat_strip as G

BG = "#f7f7f5"
MAT = "#a9bfdd"
EDGE = "#2f4f7f"


def tris_of(man):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    return v[np.asarray(mesh.tri_verts)]


def shaded(ax, tris, colour, light=(0.45, -0.75, 0.6)):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    lam = np.clip(n @ (np.array(light) / np.linalg.norm(light)), 0, 1)
    cols = np.array(colour)[None, :] * (0.3 + 0.7 * lam)[:, None]
    ax.add_collection3d(Poly3DCollection(tris, facecolors=np.clip(cols, 0, 1),
                                         edgecolors="none"))


def fit(ax, pts, elev, azim, title):
    """True proportions, not a forced cube -- the tile is 44 x 16 x 6 and a
    cube aspect shrinks it to a sliver in the middle of the panel."""
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    for setter, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        setter(lo[k] - 0.02 * span[k], hi[k] + 0.02 * span[k])
    ax.set_box_aspect(tuple(span / span.max()))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9.5, color="0.2")


def spans_at(tile, y, z):
    """The x intervals where the tile has material, on the line (y, z)."""
    edges = []
    for poly in tile.slice(float(z)).to_polygons():
        q = np.asarray(poly)
        for k in range(len(q)):
            a, b = q[k], q[(k + 1) % len(q)]
            if (a[1] - y) * (b[1] - y) < 0:
                edges.append(a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]))
    edges.sort()
    return list(zip(edges[0::2], edges[1::2]))


def section(ax, tile, y, centre, title, steps=320):
    """Across the bore, not along it. The bridging happens over the bore's
    WIDTH, which runs along the seam, so the view that answers the question
    is the one the nozzle sees: width across, layer height up."""
    for z in np.linspace(0.004, G.TOP_Z - 0.004, steps):
        for x0, x1 in spans_at(tile, y, z):
            ax.plot([x0 - centre, x1 - centre], [z, z], color=MAT, lw=1.3,
                    solid_capstyle="butt", zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(-5.2, 5.2)
    ax.set_ylim(-0.3, G.TOP_Z + 0.5)
    ax.set_xlabel("mm along the seam", fontsize=8)
    ax.set_ylabel("mm above the bed", fontsize=8)
    ax.set_title(title, fontsize=9.5, color="0.2")
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


if __name__ == "__main__":
    tile, ang, miter, face_w = T.build_tile()
    name = G.SOLID.name
    tris = tris_of(tile)

    other = T.partner(tile, miter)

    fig = plt.figure(figsize=(12.6, 8.8), facecolor=BG)
    ax = fig.add_subplot(2, 2, 1, projection="3d", facecolor=BG)
    shaded(ax, tris, (0.35, 0.50, 0.79))
    fit(ax, tris.reshape(-1, 3), 20, -74,
        "the tile, seam face toward you -- four bores")

    ax2 = fig.add_subplot(2, 2, 2, projection="3d", facecolor=BG)
    ta, tb = tris, tris_of(other)
    shaded(ax2, ta, (0.35, 0.50, 0.79))
    shaded(ax2, tb, (0.80, 0.43, 0.35))
    fit(ax2, np.concatenate([ta, tb]).reshape(-1, 3), 14, -58,
        f"two of them, closed on the seam ({ang:.2f}°)")

    # cut through the middle of each bore's length, along the seam
    sec = math.hypot(1.0, miter)
    t = T.bore_centre_t(miter, face_w)
    y_mid = (G.CONTACT_CLEARANCE + t * miter / sec
             + (T.BORE_DEPTH / 2) * (1.0 / sec))
    ax3 = fig.add_subplot(2, 2, 3)
    section(ax3, tile, y_mid, T.ROUND_AT, "round bore -- its ceiling is a bridge")
    ax4 = fig.add_subplot(2, 2, 4)
    section(ax4, tile, y_mid, T.TEARDROP_AT,
            "teardrop -- a ridge, nothing unsupported")

    fig.suptitle(
        f"Seam test — {name}: {ang:.2f}° joint, {T.BORE_DIA:.2f}mm bore "
        f"{math.degrees(math.atan(miter)):.0f}° above horizontal, "
        f"rim {G.WALL_HEIGHT:.2f}mm", fontsize=12, color="0.15")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"hardware/tests/seam_test_{name.split()[-1].lower()}.png"
    fig.savefig(out, dpi=140, facecolor=BG)
    print(f"wrote {out}")
