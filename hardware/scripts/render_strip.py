"""Renders the flat strip STL for eyeballing: a shaded 3D view of the whole
part, and a cross-section straight through one fold line -- which is where
the interesting geometry is, since the hinge relief, the thinned base band
and the contact face are all invisible from outside.

Run from the repo root:
    python3 hardware/scripts/render_strip.py
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import gen_flat_strip as G
from polyhedra import shared_verts, edge_key

BG = "#f7f7f5"


def shaded(ax, tris, light=(0.35, 0.5, 0.79), two_sided=False):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-12)[:, None]
    lam = n @ np.array(light) / np.linalg.norm(light)
    # an open surface like a bare cap has triangles facing both ways; lighting
    # it one-sided leaves half of them black
    lam = np.clip(np.abs(lam) if two_sided else lam, 0, 1)
    shade = 0.25 + 0.75 * lam
    colours = np.stack([shade * 0.42, shade * 0.55, shade * 0.72, np.ones_like(shade)], 1)
    ax.add_collection3d(Poly3DCollection(tris, facecolors=colours, edgecolors="none"))


def view(ax, tris, elev, azim, title, light=(0.35, 0.5, 0.79)):
    shaded(ax, tris, light)
    p = tris.reshape(-1, 3)
    lo, hi = p.min(0), p.max(0)
    mid, span = (lo + hi) / 2, (hi - lo)
    r = max(span[0], span[1]) / 2 * 1.02
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(lo[2], max(hi[2], lo[2] + 1e-6))
    # true scale: z is only a few mm against ~90mm in x and y, so the box has
    # to be told that, or the strip renders as if it were a deep box
    ax.set_box_aspect((2 * r, 2 * r, max(span[2], 1e-6)))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, color="0.25", pad=0)


if __name__ == "__main__":
    path = G.SOLID.strips()[0]
    solid, face_2d = G.build_strip(path)
    mesh = solid.to_mesh()
    tris = np.asarray(mesh.vert_properties)[:, :3][np.asarray(mesh.tri_verts)]

    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    for k, (elev, azim, name) in enumerate(
            [(90, -90, "flat on the bed -- as printed"), (26, -62, "raking view")]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d", facecolor=BG)
        ax.set_proj_type("ortho")
        view(ax, tris, elev, azim, name)

    # --- what it becomes: this cap's faces on the real icosahedron ---------
    V = np.array(G.SOLID.vertices, float)
    V *= G.TARGET_EDGE / np.linalg.norm(V[G.SOLID.faces[0][0]] - V[G.SOLID.faces[0][1]])
    cap = np.array([[V[i] for i in G.SOLID.faces[fi]] for fi in path])
    ax = fig.add_subplot(2, 2, 3, projection="3d", facecolor=BG)
    ax.set_proj_type("ortho")
    axis = np.cross(cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0]).sum(0)
    axis /= np.linalg.norm(axis)
    # stand the cap on its own axis so it reads as a bowl rather than a
    # silhouette, then look at it from slightly above the rim
    zhat = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, zhat)
    sin, cos = np.linalg.norm(v), float(np.dot(axis, zhat))
    if sin < 1e-9:
        R = np.eye(3) if cos > 0 else -np.eye(3)
    else:
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / sin
        R = np.eye(3) + math.sin(math.atan2(sin, cos)) * k + (1 - cos) * (k @ k)
    cap = cap @ R.T
    shaded(ax, cap, (0.30, 0.42, 0.86), two_sided=True)
    lo, hi = cap.reshape(-1, 3).min(0), cap.reshape(-1, 3).max(0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.view_init(elev=30, azim=-62)
    ax.set_axis_off()
    ax.set_title("folded -- the cap it becomes (exact, from the solid)",
                 fontsize=9, color="0.25", pad=0)

    # --- cross-section through one fold line -----------------------------
    i = min(len(path) // 2, len(path) - 2)
    va, vb = shared_verts(G.SOLID.faces, path[i], path[i + 1])
    miter = G.miter_of(edge_key(va, vb))
    pts = dict(face_2d[path[i]])
    pa, pb = np.array(pts[va]), np.array(pts[vb])
    mid = (pa + pb) / 2
    u = (pb - pa) / np.linalg.norm(pb - pa)
    nrm = np.array([-u[1], u[0]])

    ax = fig.add_subplot(2, 2, 4, facecolor=BG)
    zs = np.linspace(0.005, G.TOP_Z - 0.005, 420)
    for sign in (+1, -1):
        xs, keep = [], []
        for z in zs:
            hits = []
            for poly in solid.slice(float(z)).to_polygons():
                q = np.asarray(poly)
                for k in range(len(q)):
                    t = G.ray_hits_segment(mid, sign * nrm, q[k], q[(k + 1) % len(q)])
                    if t is not None:
                        hits.append(t)
            if hits:
                xs.append(sign * min(hits))
                keep.append(z)
        ax.plot(xs, keep, color="#2f4f7f", lw=2.0)
        ax.fill_betweenx(keep, xs, sign * 6, color="#5b7fb5", alpha=0.35, lw=0)

    ax.axvline(0, color="0.7", lw=0.8, ls=(0, (4, 3)))
    ax.axhline(G.PIVOT_Z, color="#b5651d", lw=1.0, ls=(0, (2, 2)))
    ax.text(3.9, G.PIVOT_Z + 0.06, "pivot", color="#b5651d", fontsize=7, ha="right")
    ax.axhline(G.FACE_THICKNESS, color="0.6", lw=0.7, ls=(0, (1, 2)))
    ax.axhline(G.RELIEF_Z, color="0.6", lw=0.7, ls=(0, (1, 2)))
    ax.text(3.9, G.FACE_THICKNESS + 0.06, "face top", color="0.45", fontsize=7, ha="right")
    ax.text(3.9, G.RELIEF_Z + 0.06, "relief ends", color="0.45", fontsize=7, ha="right")

    # where the two walls will meet once folded
    for sign in (+1, -1):
        zc = np.linspace(G.RELIEF_Z, G.TOP_Z, 2)
        ax.plot(sign * (G.CONTACT_CLEARANCE + (zc - G.PIVOT_Z) * miter), zc,
                color="#c0392b", lw=1.4)
    ax.plot([], [], color="#c0392b", lw=1.4, label="contact face (true miter)")
    ax.legend(loc="upper center", fontsize=7, frameon=False)

    ax.set_xlim(-4, 4)
    ax.set_ylim(0, G.TOP_Z + 0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("mm from the fold line", fontsize=8)
    ax.set_title("section through a fold", fontsize=9, color="0.25")
    for sp in ax.spines.values():
        sp.set_color("0.8")
    ax.tick_params(labelsize=7, colors="0.5")

    fig.suptitle(f"Hamiltonian Polyhedron - {G.SOLID.name.lower()} cap strip, "
                 f"{G.TARGET_EDGE:.0f}mm edges",
                 fontsize=11, color="0.2")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"hardware/tests/strip_render_{G.SOLID.name.split()[-1].lower()}.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(f"wrote {out}")
