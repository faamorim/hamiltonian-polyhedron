"""Generates a fused pair of adjacent icosahedron face pieces as one STL,
ready to import into Bambu Studio's Cut tool to split apart with a native
connector (dovetail / snap-fit / etc).

Why fused, not pre-connected: Bambu Studio's connectors are calibrated to
your actual printer + filament profile, so there's no reason to hand-tune
our own dovetail clearance -- we just need to hand it a single watertight
solid with the two faces meeting at the polyhedron's real dihedral angle,
and let Bambu's Cut tool do one planar cut with a connector at that seam.

Why one pair at a time, not the whole 10-face strip at once: Bambu's Cut
tool splits a solid along an infinite plane. The strip is curled around
the icosahedron in 3D, not flat, so a cut plane aligned with one internal
seam could also clip through some other, unrelated piece elsewhere in the
curl. Doing it pair by pair keeps each cut unambiguous.

Geometry approach: each face's solid "wedge" is the convex hull of its
outer triangle plus the same triangle uniformly scaled toward the
icosahedron's center. Two adjacent faces scaled by the same factor share
the exact same (scaled) edge points, so their wedges are automatically
flush along the whole shared wall -- not just touching at the outer edge,
which is what a naive per-face straight-normal extrusion would give you.

Requires: pip install manifold3d

Run from the repo root:
    python3 hardware/scripts/gen_face_pair.py
"""

import math
import manifold3d as m3d

PHI = (1 + 5 ** 0.5) / 2

# Same icosahedron data as index.html (vendor/three-free, hand-derived earlier
# in this project): 12 vertices, 20 triangular faces, and the 12-edge
# Hamiltonian cycle that splits them into two 10-face caps.
VERTICES = [
    [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
    [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
    [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
]
FACES = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
]
CYCLE = [0, 11, 5, 1, 7, 6, 3, 8, 9, 4, 2, 10]

TARGET_EDGE = 22.0          # mm, prototype face edge length (smaller than the
                             # ~57mm "Rubik's cube" final target, but the wall
                             # depth ratio below is fixed, not scaled down --
                             # so this prototype tests the same proportions)
WALL_DEPTH_RATIO = 0.90      # inner (hollow-side) triangle at 90% of outer radius


def internal_adjacency(faces, cycle):
    """Face-to-face adjacency within one cap: shared edges that are NOT part
    of the Hamiltonian cycle (those are the seam between the two caps)."""
    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    cycle_edges = {edge_key(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))}
    face_edge_map = {}
    for fi, f in enumerate(faces):
        for k in range(3):
            key = edge_key(f[k], f[(k + 1) % 3])
            face_edge_map.setdefault(key, []).append(fi)

    adj = {i: [] for i in range(len(faces))}
    for key, lst in face_edge_map.items():
        if len(lst) == 2 and key not in cycle_edges:
            adj[lst[0]].append(lst[1])
            adj[lst[1]].append(lst[0])
    return adj


def scaled_vertices(target_edge):
    raw_edge = math.dist(VERTICES[FACES[0][0]], VERTICES[FACES[0][1]])
    scale = target_edge / raw_edge
    return [[c * scale for c in v] for v in VERTICES]


def wedge(face_idx, points, wall_depth_ratio):
    outer = [points[i] for i in FACES[face_idx]]
    inner = [[c * wall_depth_ratio for c in p] for p in outer]
    return m3d.Manifold.hull_points(outer + inner)


def export_stl(manifold_obj, path):
    mesh = manifold_obj.to_mesh()
    verts = mesh.vert_properties
    tris = mesh.tri_verts
    with open(path, "w") as f:
        f.write("solid part\n")
        for tri in tris:
            v0, v1, v2 = (verts[i][:3] for i in tri)
            ux, uy, uz = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
            vx, vy, vz = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
            nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
            norm = (nx ** 2 + ny ** 2 + nz ** 2) ** 0.5 or 1.0
            f.write(f"  facet normal {nx/norm:.6f} {ny/norm:.6f} {nz/norm:.6f}\n")
            f.write("    outer loop\n")
            for v in (v0, v1, v2):
                f.write(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid part\n")


def make_pair(face_a, face_b, points):
    wa = wedge(face_a, points, WALL_DEPTH_RATIO)
    wb = wedge(face_b, points, WALL_DEPTH_RATIO)
    fused = wa + wb
    parts = fused.decompose()
    assert len(parts) == 1, (
        f"faces {face_a},{face_b} did not fuse into one solid "
        f"({len(parts)} disconnected shells) -- they may not actually be edge-adjacent"
    )
    return fused


if __name__ == "__main__":
    adj = internal_adjacency(FACES, CYCLE)
    points = scaled_vertices(TARGET_EDGE)

    face_a = 0
    face_b = adj[face_a][0]
    print(f"pair: face {face_a} {FACES[face_a]}  <->  face {face_b} {FACES[face_b]}")

    fused = make_pair(face_a, face_b, points)
    print("fused volume:", fused.volume(), "bbox:", fused.bounding_box())

    export_stl(fused, f"hardware/tests/icosahedron_face_pair_{face_a}_{face_b}.stl")
    print(f"wrote hardware/tests/icosahedron_face_pair_{face_a}_{face_b}.stl")
