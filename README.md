# Hamiltonian Cycles on Polyhedra

A Hamiltonian cycle visits every vertex of a graph exactly once. Drawn on the
surface of a polyhedron it becomes something you can hold: a closed seam that
cuts the solid into **two halves, each of which still carries every vertex**.

Live: **https://faamorim.github.io/hamiltonian-polyhedron/**

This repository makes that cut in three forms — a 3D visualizer in the
browser, printable paper nets, and 3D-printable models that come apart in
your hands and magnet back together.

## Why the two halves work

Cut a convex polyhedron's surface along a Hamiltonian cycle and three things
fall out, none of them accidental:

- **Every vertex survives on both halves.** The cycle passes *through* each
  vertex rather than around it, so the vertex ends up on both rims.
- **Exactly `V` edges are cut** — that is Euler's formula rearranged, since
  `E − F + 2 = V`.
- **Each half unfolds flat.** It has to: the cycle puts two cut edges at
  every vertex, so the fan of faces around each vertex is broken into an arc
  and no half ever closes a wheel around one. With no interior vertices, the
  faces of a half form a tree, and a tree rolls out flat whatever its shape.
- **The halves we ship also form a single *strip*** — a path of faces rather
  than a branching tree — which is what the printed part is: one linear
  ladder on living hinges. That is a property of this cut, not of cuts in
  general, and it is about the hinge design rather than about flatness.
  Whether a net overlaps itself is a separate question that has to be
  measured; `hardware/scripts/check_unfold.py` measures it.

The two halves are also **interlocked**: their mating faces point in enough
different directions that no straight pull separates them. They come apart
by hinging, not by lifting.

## What you can make

| | |
|---|---|
| **Visualizer** | `index.html` — pick a solid, pull the two caps apart, watch the cycle draw itself and split the solid as it goes. |
| **Paper net** | One PDF per solid: both strips, fold lines, glue tabs, printed at 100% on A4 or Letter with a 50mm ruler on the page to prove the scale. |
| **3D-printed model** | One STL per solid per size. The strip prints flat and folds up on living hinges; magnets at the seam hold the halves together. |

Six solids: tetrahedron, cube, octahedron, dodecahedron, icosahedron, and the
pentagonal trapezohedron (the d10 — not Platonic, but it has a Hamiltonian
cycle and it makes a good object).

## Printing the model

**One part, printed twice.** The two halves are not related by any symmetry of
the solid, but their flat nets are congruent, so the same file is both halves.
The build checks this rather than assuming it, by turning one onto the other
and measuring the volume that fails to overlap.

**Pick the size on the page, not in the slicer.** Sizes are offered as a
ladder (25, 32, 40, 50, 63mm) because the hinge is a fixed thickness in
millimetres while the faces scale — scaling a finished STL scales the hinge
too, and it stops bending. Each rung is generated and checked at its own size.

You will also need, per solid:

- **4 × 2mm cylindrical magnets**, glued into sockets that open on the inside
  face. The magnet's pole presses into a thin wall of the mating face, so the
  load is compression and the glue is never in the load path. Small sizes take
  no magnets — the faces have no room for a socket — and the page says how
  many each rung takes.
- **A second filament colour** for the inlays: a small triangle on each of the
  two faces where a strip ends, in the other half's colour. Four per solid.
  They drop into shallow pockets and glue flush.

## Assembling one

The STL holds the strip, its lids and its inlays side by side, and you print
the same file twice. Print **one copy in each colour** — the two prints then
contain exactly the parts each other needs:

1. Print the file twice, once in each filament. Flat on the bed, no supports,
   no raft under the strip. Do not scale it.
2. **Swap the inlays between the two prints.** Each copy comes with two, and
   each body needs two of the other colour, so the swap is exact. Glue them
   into the shallow pockets on the outside of the two faces at the ends of
   each strip; they sit flush.
3. Glue a magnet into each socket, entered from the inside face. Check the
   polarity against its partner across the seam **before** the glue sets —
   two sockets that repel are a solid that will not close.
4. Press the lids into the rebates over the sockets. They are cosmetic; the
   magnet is already held by plastic and glue.
5. Fold each strip along its thinned hinges, one fold at a time, until the
   walls meet. Take your time on the first fold of each hinge: the plastic
   yields once and then stays bent.
6. Bring the two halves together. They interlock — no straight pull separates
   them, so close them by hinging one into the other, and the magnets will
   pull the last of the way.

## Credit

The object this extends is ***Desconstrucció d'un dodecàedre*** by
**[Josep Rey Nadal](https://gallery.bridgesmathart.org/exhibitions/bridges-2026-exhibition-of-mathematical-art/josep-rey-nadal)**, a wooden
dodecahedron that unzips along a continuous path into two complementary
halves, each inlaid with the other's wood — a yin-yang in three dimensions.
Our extension moves it to the icosahedron and to 3D printing. Made for a
mathematics-and-art course project (EDCP 342).

## Repository

```
index.html              the visualizer; three.js is vendored in vendor/
polyhedra.js            generated — the solids and their cycle census
downloads.js            generated — what the page offers, and its real figures
hardware/scripts/       every generator and check
hardware/tests/         the generated STLs, PDFs and renders
ROADMAP.md              what is coming, and what was turned down and why
```

Everything in `hardware/tests/` is generated. To rebuild all of it:

```sh
pip install manifold3d numpy shapely matplotlib
python3 hardware/scripts/gen_downloads.py
```

That writes every STL and PDF **and** `downloads.js` **and** `polyhedra.js` in
the same run, so neither the sizes quoted on the page nor the cut it draws can
drift from the files you download. The solids are defined once, in
`hardware/scripts/polyhedra.py`, which checks each one — Euler's formula, the
cycle visiting every vertex exactly once, every step of it a real edge, every
face planar — and the page reads what passed.

It also carries each solid's census, enumerated by
`hardware/scripts/census.py`: every Hamiltonian cycle, how many are
essentially different once the solid's own symmetries are divided out, how
many of those cut the solid into two *strips* rather than two branching
trees, and whether the cut we ship has congruent halves. That last one is why
a single STL can be printed twice. The enumeration agrees with the literature
where it has an opinion — 30 cycles on the dodecahedron, Hamilton's own
icosian game, and 1280 undirected on the icosahedron — and it runs in about
two seconds for all six.

`check_unfold.py` is the companion that answers the question `census.py` only
appears to: it rolls every cap of every cycle out flat and measures whether
the net overlaps itself. It establishes that every cap lies flat, strip or
not, and that overlap is a separate property that some strips have and most
branching caps do not.

The checks are the interesting part of the scripts. Each generator measures
its result off the finished mesh instead of trusting the design: the seam is
verified to sit on the bisector of the joint, the wall is ray-cast and
compared to where the design says it should be, and `check_fold.py` folds both
halves for real and intersects them to ask whether they actually close. That
habit exists because several bugs here passed every check that only looked at
intent — a hole webbed over by the surface it was cut into, a seam that
measured correct but stood on a ledge that kept the two halves apart.
