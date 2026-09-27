"""Counts the net corners where two fold lines cross, and measures what the
thinned base has to do there.

Every fold line of a strip ends on the net's outline, so wherever two faces of
the strip share a corner their two hinge trenches cross and that patch of
0.6mm base bends about both fold axes at once. This reports how much of the
hinge that is and how hard it is asked to work, per solid, because the answer
decides whether the patch is worth cutting away -- see hinge_trench in
gen_flat_strip.py for why it currently is not.

Curvature adds as a tensor, so the largest principal curvature at a crossing
is not twice a single fold's. The stretching the patch also needs is measured
as the integral of Gaussian curvature over the overlap, which comes out as
fold_i x fold_j x sin(crossing angle) and so does not depend on the hinge
width at all. That integral is a LINEARIZED figure and overestimates: the same
sum over a closed icosahedron vertex gives 4.6sr where the true angle defect
is 1.047sr, a factor of 4.4, so divide by a few before believing it.

The strain figures are indices, not strains. 27% of engineering strain would
have broken PLA, and the printed icosahedron hinge bends happily, because the
bend spreads out of the trench into the 1.2mm faces. Read them against each
other, not against a datasheet.

Requires: pip install numpy shapely

Run from the repo root:
    python3 hardware/scripts/check_corners.py
"""

import collections
import math

import numpy as np
from shapely import affinity
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from polyhedra import SOLIDS, shared_verts, unfold

# Copied rather than imported: gen_flat_strip builds a solid at import time for
# one named solid, and this walks all six.
FACE_THICKNESS, HINGE_THICKNESS = 1.2, 0.6
OVERLAP, CONTACT_CLEARANCE = 0.3, 0.08
FOOT_Z = HINGE_THICKNESS - OVERLAP
HINGE_GAP_PER_RAD = 0.80 / math.radians(41.81)
MIN_HINGE_GAP, TRENCH_OVERSHOOT = 0.6, 0.05
SIZE_MM = 40.0


def fold_angle(solid, a, b):
    """180 degrees minus the dihedral, off the two faces sharing the edge."""
    normals = []
    for f in [f for f in solid.faces if a in f and b in f]:
        p = np.array([solid.vertices[i] for i in f], float)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n /= np.linalg.norm(n)
        normals.append(n if n @ p.mean(0) > 0 else -n)
    return math.acos(max(-1.0, min(1.0, float(normals[0] @ normals[1]))))


def trenches(solid, path, face_2d):
    """Each fold's trench footprint, as gen_flat_strip cuts it."""
    out = []
    for i in range(len(path) - 1):
        a, b = shared_verts(solid.faces, path[i], path[i + 1])
        pts = dict(face_2d[path[i]])
        pa, pb = np.array(pts[a]), np.array(pts[b])
        angle = fold_angle(solid, a, b)
        width = max(MIN_HINGE_GAP, HINGE_GAP_PER_RAD * angle)
        half = max(width / 2, CONTACT_CLEARANCE
                   + (FACE_THICKNESS - FOOT_Z) * math.tan(angle / 2))
        length = float(np.linalg.norm(pb - pa)) + 2 * TRENCH_OVERSHOOT
        rect = box(-length / 2, -half, length / 2, half)
        rect = affinity.rotate(rect, math.degrees(math.atan2(*(pb - pa)[::-1])))
        out.append(dict(a=pa, b=pb, angle=angle, width=width, vertices=(a, b),
                        poly=affinity.translate(rect, *(pa + pb) / 2)))
    return out


def report(solid):
    print(f"--- {solid.name}, {SIZE_MM:.0f}mm perceived")
    crossing_cap = {}                 # solid vertex -> the caps that cross there
    for ci, path in enumerate(solid.strips()):
        face_2d = unfold(solid, path, SIZE_MM / solid.mean_width())
        net = unary_union([Polygon([q for _, q in face_2d[fi]])
                           for fi in path]).buffer(0)
        folds = trenches(solid, path, face_2d)
        thin = unary_union([f["poly"] for f in folds]).intersection(net)

        ends = collections.defaultdict(list)
        for k, f in enumerate(folds):
            for gv, p in zip(f["vertices"], (f["a"], f["b"])):
                ends[(round(p[0], 4), round(p[1], 4), gv)].append(k)

        rows = []
        for (x, y, gv), ks in sorted(ends.items()):
            if len(ks) < 2:
                continue
            crossing_cap.setdefault(gv, []).append(ci)
            patch = unary_union([folds[i]["poly"].intersection(folds[j]["poly"])
                                 for i in ks for j in ks if i < j]).intersection(net)
            # interior angle of the net at this corner, to say what kind it is
            interior = 0.0
            for fi in path:
                pts = [np.array(q) for _, q in face_2d[fi]]
                for t in range(len(pts)):
                    if abs(pts[t][0] - x) > 1e-4 or abs(pts[t][1] - y) > 1e-4:
                        continue
                    u = pts[t - 1] - pts[t]
                    v = pts[(t + 1) % len(pts)] - pts[t]
                    interior += math.acos(max(-1.0, min(1.0, float(
                        u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))))
            # curvature tensor of the folds meeting here, and the worst pair
            M = np.zeros((2, 2))
            worst = 0.0
            dirs = {}
            for i in ks:
                d = folds[i]["b"] - folds[i]["a"]
                d /= np.linalg.norm(d)
                dirs[i] = d
                M += (folds[i]["angle"] / folds[i]["width"]) * np.outer([-d[1], d[0]],
                                                                        [-d[1], d[0]])
            for i in ks:
                for j in ks:
                    if i >= j:
                        continue
                    cross = math.acos(abs(max(-1.0, min(1.0, float(dirs[i] @ dirs[j])))))
                    worst = max(worst, folds[i]["angle"] * folds[j]["angle"] * math.sin(cross))
            kmax = max(abs(w) for w in np.linalg.eigvalsh(M))
            reach = max(Point(x, y).distance(Point(c)) for g in
                        (patch.geoms if patch.geom_type != "Polygon" else [patch])
                        for c in g.exterior.coords)
            rows.append((gv, len(ks), math.degrees(interior), patch.area, reach,
                         kmax, math.degrees(worst)))

        if not rows:
            print("  no two fold lines share a net corner -- nothing to report")
            return
        single = folds[0]["angle"] / folds[0]["width"]
        print(f"  cap{ci}: {len(folds)} folds, hinge {thin.area:6.2f}mm2, "
              f"single-fold strain index {100 * HINGE_THICKNESS * single / 2:.0f}%")
        for gv, n, interior, area, reach, kmax, intK in rows:
            print(f"    vertex {gv:2d}  {n} folds  outline {interior:6.2f}deg  "
                  f"patch {area:5.3f}mm2 reaching {reach:4.2f}mm  "
                  f"strain index {100 * HINGE_THICKNESS * kmax / 2:3.0f}%  "
                  f"stretch {intK:5.1f}deg")

    both = {v: c for v, c in crossing_cap.items() if len(c) > 1}
    assert not both, (
        f"{solid.name}: {both} -- a vertex is crossed by BOTH caps, so cutting "
        "the patches away would leave a hole through the shell. That should be "
        "impossible below vertex degree 6")
    print(f"  every crossing is carried by one cap alone ({len(crossing_cap)} of "
          f"{len(solid.vertices)} vertices crossed), so the mating cap always "
          "backs the patch")


if __name__ == "__main__":
    for name in ["tetrahedron", "cube", "octahedron", "dodecahedron",
                 "icosahedron", "trapezohedron"]:
        report(SOLIDS[name])
