"""Draws the socket test wedge: the section straight through the socket,
which is the only view that shows what the design is actually about, plus
the pair closed on the seam.

    python3 hardware/scripts/render_socket_test.py [solid]

The sections are read off the finished mesh, not drawn from the design:
sliced at many heights and asked where the material lies on one line.
"""

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MPoly
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gen_socket_test as T
import gen_flat_strip as G

BG = "#f7f7f5"
MAT = "#a9bfdd"
MAGNET = "#c0392b"
INK = "0.25"


def tris_of(man):
    mesh = man.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    return v[np.asarray(mesh.tri_verts)]


def spans(tile, fixed, z, axis):
    """Material intervals on one line at height z. `axis` is the coordinate
    the intervals run along -- 1 for y, a cut ACROSS the seam at fixed x; 0
    for x, a cut ALONG it at fixed y -- and `fixed` is the other one."""
    other = 1 - axis
    out = []
    for poly in tile.slice(float(z)).to_polygons():
        q = np.asarray(poly)
        for k in range(len(q)):
            a, b = q[k], q[(k + 1) % len(q)]
            # half-open: an edge with a vertex exactly ON the line counts
            # once, not twice and not never. The wedge is symmetric about
            # x = 0, so cutting there puts vertices on the line constantly,
            # and a strict product test drops them and mispairs everything
            # after -- whole slabs of solid read as empty.
            if (a[other] > fixed) != (b[other] > fixed):
                f = (fixed - a[other]) / (b[other] - a[other])
                out.append(a[axis] + f * (b[axis] - a[axis]))
    out.sort()
    return list(zip(out[0::2], out[1::2]))


def profile(ax, tile, fixed, axis, top, steps=420):
    for z in np.linspace(0.003, top - 0.003, steps):
        for p, q in spans(tile, fixed, z, axis):
            ax.plot([p, q], [z, z], color=MAT, lw=1.25,
                    solid_capstyle="butt", zorder=1)


def magnet_outline(miter):
    """Where the magnet ends up, in the plane of the cut across the seam."""
    n, e2 = T.frame(miter)
    c = T.on_face(T.seat_t(miter), miter) + T.WALL * n
    r, h = T.MAGNET_DIA / 2, T.MAGNET_H
    pts = [c - r * e2, c + r * e2, c + r * e2 + h * n, c - r * e2 + h * n]
    return np.array([[p[1], p[2]] for p in pts])


if __name__ == "__main__":
    name = G.SOLID.name
    ang = min(round(a, 6) for a in G.FOLD_ANGLES.values())
    miter = math.tan(math.radians(ang) / 2)
    tile, lid, _ = T.build(miter)
    top = T.boss_top_z(miter)

    fig = plt.figure(figsize=(12.8, 7.4), facecolor=BG)

    # --- across the seam, straight through the socket --------------------
    ax = fig.add_subplot(1, 2, 1)
    profile(ax, tile, 0.0, 1, top + 0.1)
    ax.add_patch(MPoly(magnet_outline(miter), closed=True, facecolor=MAGNET,
                       alpha=0.85, edgecolor="none", zorder=3))
    zz = np.linspace(0, top, 2)
    ax.plot(G.CONTACT_CLEARANCE + miter * zz, zz, color="#c0392b", lw=1.0,
            ls=(0, (5, 4)), zorder=4)
    ax.annotate("the seam: this face meets the other cap",
                xy=(G.CONTACT_CLEARANCE + miter * top * 0.72, top * 0.72),
                xytext=(4.6, 5.9), fontsize=7.5, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.9))
    ax.annotate(f"{T.WALL:.1f}mm wall\n(the load path)", xy=(1.35, 3.1),
                xytext=(4.6, 2.3), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate("solid behind it", xy=(4.3, 3.0), xytext=(6.4, 1.2),
                fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.annotate("drops in from here,\nlid sits flush", xy=(3.0, top - 0.1),
                xytext=(6.0, top + 0.5), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax.axhline(G.TOP_Z, color="0.6", lw=0.8, ls=":")
    ax.text(11.2, G.TOP_Z + 0.12, f"rim, {G.WALL_HEIGHT:.1f}mm (unchanged)",
            fontsize=7, color="0.45", ha="right")
    ax.axhline(top, color="0.6", lw=0.8, ls=":")
    ax.text(11.2, top + 0.12, f"pad, +{top - G.TOP_Z:.2f}mm",
            fontsize=7, color="0.45", ha="right")
    ax.set_xlim(-0.4, 11.4); ax.set_ylim(-0.4, top + 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("mm into the face from the seam edge", fontsize=8)
    ax.set_ylabel("mm above the bed", fontsize=8)
    ax.set_title("across the seam, through the socket", fontsize=10, color=INK)

    # --- along the seam ---------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2)
    n, _ = T.frame(miter)
    y_mid = (T.on_face(T.seat_t(miter), miter) + (T.WALL + T.MAGNET_H / 2) * n)[1]
    profile(ax2, tile, y_mid, 0, top + 0.1)
    ax2.set_xlim(-7.5, 7.5); ax2.set_ylim(-0.3, top + 0.5)
    ax2.set_aspect("equal")
    ax2.set_xlabel("mm along the seam", fontsize=8)
    ax2.set_title("along the seam: the channel, open at the top",
                  fontsize=10, color=INK)

    # --- the pair ---------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 4, projection="3d", facecolor=BG)
    a, b = tris_of(tile), tris_of(T.partner(tile, miter))
    for t, col in ((a, (0.35, 0.50, 0.79)), (b, (0.80, 0.43, 0.35))):
        nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-12)[:, None]
        lam = np.clip(nn @ np.array([0.4, -0.75, 0.55]) / 1.0, 0, 1)
        ax3.add_collection3d(Poly3DCollection(
            t, facecolors=np.clip(np.array(col)[None, :] * (0.32 + 0.68 * lam)[:, None], 0, 1),
            edgecolors="none"))
    pts = np.concatenate([a, b]).reshape(-1, 3)
    lo, hi = pts.min(0), pts.max(0)
    span = np.maximum(hi - lo, 1e-6)
    ax3.set_xlim(lo[0], hi[0]); ax3.set_ylim(lo[1], hi[1]); ax3.set_zlim(lo[2], hi[2])
    ax3.set_box_aspect(tuple(span / span.max()))
    ax3.view_init(elev=14, azim=-62); ax3.set_axis_off()
    ax3.set_title(f"two of them, closed ({ang:.2f}°)", fontsize=10, color=INK)

    for a_ in (ax, ax2):
        a_.tick_params(labelsize=7)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    fig.suptitle(f"Socket test — {name}: {T.MAGNET_DIA:.0f}x{T.MAGNET_H:.0f}mm "
                 f"magnet, {2 * T.WALL:.1f}mm between the pair",
                 fontsize=12, color="0.15")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"hardware/tests/socket_test_{name.split()[-1].lower()}.png"
    fig.savefig(out, dpi=140, facecolor=BG)
    print(f"wrote {out}")
