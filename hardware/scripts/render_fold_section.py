"""Draws a close-up section across one fold line, flat and then folded to the
angle at which the two sides first touch -- the view that shows whether the
hinge actually closes to the angle the solid needs, or jams early on some
other feature.

    python3 hardware/scripts/render_fold_section.py [solid]

Every boundary is measured off the finished mesh by slicing it and casting a
ray from the fold line, so this is the real part, not the design intent.
"""

import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import gen_flat_strip as G
from polyhedra import shared_verts, edge_key

BG = "#f7f7f5"
MAT = "#5b7fb5"
EDGE = "#2f4f7f"


def section(solid, face_2d, path, steps=1400):
    """The material boundary d(z) across the middle fold of the strip."""
    i = min(len(path) // 2, len(path) - 2)
    va, vb = shared_verts(G.SOLID.faces, path[i], path[i + 1])
    pts = dict(face_2d[path[i]])
    pa, pb = np.array(pts[va]), np.array(pts[vb])
    mid = (pa + pb) / 2
    u = (pb - pa) / np.linalg.norm(pb - pa)
    nrm = np.array([-u[1], u[0]])
    if np.dot(nrm, np.mean([q for _, q in face_2d[path[i]]], axis=0) - mid) < 0:
        nrm = -nrm
    prof = []
    for z in np.linspace(0.002, G.TOP_Z - 0.002, steps):
        hits = []
        for poly in solid.slice(float(z)).to_polygons():
            q = np.asarray(poly)
            for k in range(len(q)):
                t = G.ray_hits_segment(mid, nrm, q[k], q[(k + 1) % len(q)])
                if t is not None:
                    hits.append(t)
        prof.append((float(z), min(hits) if hits else 0.0))
    return prof, G.miter_of(edge_key(va, vb)), G.FOLD_ANGLES[edge_key(va, vb)]


def side(prof, psi, sign):
    """One half of the section, turned by psi about the pivot."""
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for z, d in prof:
        x, y = sign * d, z - G.PIVOT_Z
        out.append((sign * (abs(x) * c - y * s), x * 0 + (abs(x) * s + y * c) + G.PIVOT_Z))
    return out


def draw(ax, prof, psi, title, far=6.0):
    for sign in (+1, -1):
        pts = side(prof, psi, sign)
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        ax.plot(xs, zs, color=EDGE, lw=1.8, zorder=3)
        ax.fill_betweenx(zs, xs, sign * far, color=MAT, alpha=0.38, lw=0)
    ax.axvline(0, color="0.72", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_title(title, fontsize=10, color="0.2")
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("0.82")
    ax.tick_params(labelsize=7, colors="0.5")


if __name__ == "__main__":
    path = G.SOLID.strips()[0]
    solid, face_2d = G.build_strip(path)
    prof, miter, target = section(solid, face_2d, path)

    live = [(z, d) for z, d in prof if z > G.HINGE_THICKNESS + 1e-6 and d > 0]
    psi_stop, z_hit, d_hit = min(
        (math.degrees(math.atan2(d, z - G.PIVOT_Z)), z, d) for z, d in live)
    psi = math.radians(psi_stop)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), facecolor=BG)
    for ax in axes:
        ax.set_facecolor(BG)

    draw(axes[0], prof, 0.0, "flat, as printed")
    for z, lab in ((G.HINGE_THICKNESS, "hinge band top"), (G.FACE_THICKNESS, "face top"),
                   (G.relief_knee(miter), "wall rejoins miter"), (G.TOP_Z, "wall top")):
        axes[0].axhline(z, color="0.62", lw=0.7, ls=(0, (1, 2)), zorder=2)
        axes[0].text(3.4, z + 0.07, lab, fontsize=7, color="0.42", ha="right")
    axes[0].axhline(G.PIVOT_Z, color="#b5651d", lw=1.0, ls=(0, (3, 2)), zorder=2)
    axes[0].text(3.4, G.PIVOT_Z + 0.07, "pivot", fontsize=7, color="#b5651d", ha="right")
    axes[0].set_xlim(-3.5, 3.5)
    axes[0].set_ylim(0, G.TOP_Z + 0.25)
    axes[0].set_xlabel("mm from the fold line", fontsize=8)

    draw(axes[1], prof, psi, f"folded until something touches: {2 * psi_stop:.2f}°")
    axes[1].plot([d_hit * math.cos(psi) - (z_hit - G.PIVOT_Z) * math.sin(psi)],
                 [d_hit * math.sin(psi) + (z_hit - G.PIVOT_Z) * math.cos(psi) + G.PIVOT_Z],
                 "o", ms=9, mfc="none", mec="#c0392b", mew=2.0, zorder=5)
    axes[1].set_xlim(-3.5, 3.5)
    axes[1].set_ylim(0, G.TOP_Z + 0.25)
    axes[1].set_xlabel("mm from the fold line", fontsize=8)

    verdict = ("closes correctly" if abs(2 * psi_stop - target) < 3
               else "JAMS EARLY" if 2 * psi_stop < target else "overshoots")
    fig.suptitle(f"{G.SOLID.name} — fold wanted {target:.2f}°, "
                 f"first contact at {2 * psi_stop:.2f}°  ({verdict})",
                 fontsize=11.5, color="0.15")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"hardware/tests/fold_section_{G.SOLID.name.split()[-1].lower()}.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out}  (first contact at z={z_hit:.3f}, d={d_hit:.4f})")
