# Roadmap

What is coming, what is parked, and — most usefully — what was considered and
turned down, so it does not get relitigated.

## Next

In this order. Each step is chosen so the one after it is not built on
something we already intend to change.

1. ~~**One source for the polyhedra.**~~ *(done)* `polyhedra.py` is canonical
   and self-verifying; the page reads a generated `polyhedra.js`.
2. ~~**Tabs.**~~ *(done)* Solid · Diagram · Make it. Per-tab sidebar, which
   retired the greying-out. Downloads moved from the 280px sidebar into the
   main area. Trace and paint became separate controls, camera-follow stayed
   a sibling rather than a child. Reset view in a canvas corner. URL state
   (`#solid/tab/palette`).
3. ~~**Colour in the diagram.**~~ *(done)* The two caps stay filled at the
   same strength they have on the solid — half opacity over a near-black
   background turned out to cost almost no saturation (0.66 → 0.57) and
   nearly half the brightness (0.77 → 0.44), which reads as washed out
   rather than as further away — and the outer region —
   the face we looked *through*, which a Schlegel diagram turns inside out —
   is washed in its own cap's colour, strongest against the outline and dying
   to the background well before the canvas edge. A flat fill would have read
   as a backdrop; the falloff reads as a region continuing outward. It is
   painted by the same `cut()` the faces use, so it greys until the seam
   claims it, and it takes the cap it genuinely belongs to — which is why
   along a stretch of outline the cycle does *not* run, the wash and the face
   inside it are the same colour. That is the point: there the outer face is
   still joined to its cap, and the wash is brightest exactly at those
   joins and quietest along the seam. A census line under the stats, from
   `hardware/scripts/census.py`. And a third palette, **plain**: one colour
   for both halves, so the diagram can be read as a drawing before it is read
   as an answer. The seam still runs through it, so the cut is visible and
   only the verdict is withheld. Its legend collapses to a single row, since
   two swatches of one colour name nothing.
4. ~~**Drawing the cycle.**~~ *(the drawing itself is done)* Drag across
   vertices in the Diagram tab, or click them one at a time; retracing onto
   the previous vertex steps back. Legal next vertices are marked, because
   that is the board rather than the move — but vertices that have become
   *unreachable* are deliberately not marked, since getting stranded is the
   whole lesson. When the loop closes the solid re-splits along it: the
   partition, the caps, the stats and the census all fall out of the code
   that already read the shipped cycle. While a line is open the old seam is
   hidden, because the cut on screen belongs to the cycle being replaced. The
   caps are **not** blanked, though: faces whose side is already settled take
   their colour as the line reaches them, and only the undecided ones stay
   neutral. The settled ones are settled for good — see below.

   Still to come, and the reason this step is only half done: **"Make it"
   does not yet explain why a drawn cut is or is not printable.** It warns
   that the files belong to the shipped cycle and offers a way back, which
   is honest but not yet useful. Making it useful needs `is_strip` in
   JavaScript. Note that the right test is **not** the strip test: it is
   whether each cap's net lays out without overlapping itself, which is a
   different and stronger question — see *What we know*. The strip test
   answers "does the printed ladder fit this cut", which is also worth
   saying, but it is a second question and not a substitute for the first.
5. **The unfold.** The Make it tab's main area becomes a canvas, and the
   solid unfolds into its two flat strips.

### Why a half-drawn line can already colour faces

A Hamiltonian cycle visits every vertex exactly once, so every vertex ends up
carrying exactly **two** cycle edges. The instant the line passes *through* a
vertex — in and out, not stopping on it — that vertex has both of them, and
every other edge at it is barred from the cycle for good, because the quota is
full. Two permanent facts follow:

- across a drawn edge the two faces are in **different** caps, since a cut edge
  is exactly where the two caps meet;
- across a non-drawn edge at a **visited** vertex the two faces are in the
  **same** cap, since that edge can never join the cycle now.

Union-find over those two constraints gives every face whose side is already
decided. Nothing is claimed at the line's two loose ends, where one edge is
still open, nor anywhere the line has not reached — and that is the honest
answer, because an open path does not separate a sphere: you can always walk
around a loose end. What it *can* say, it says at once and never retracts.

Checked by taking every prefix of 40 cycles per solid and comparing each
assignment against the finished partition: **16,765 assignments, 0 wrong.**

The colours are anchored on the first edge drawn rather than on face 0, and
`computeFacePartition`'s labelling is flipped to match when the loop closes,
so nothing swaps under the reader at the moment of closing.

## The printed part is built to one person's constraints

Raised by the first outside user: he has a bigger printer than the A1 Mini the
ladder was tuned for, did not want magnets, and did not want the inlay pockets
— and the STL gives him all three whether he likes them or not.

- **A plain variant, and taller rungs.** Nearly free, because the generator
  already anticipated it: `SOCKET_FITS` gates the sockets, the keys, the
  placements *and* the wall height (with no sockets the rim drops back to its
  minimum, so the part gets better rather than merely stripped), and
  `INLAY_FACES` is documented in the source as "the whole switch" — it takes
  `"ends"` or `"all"` and wants a `"none"`. The ladder is a loop, and an STL's
  size is triangle count rather than millimetres, so taller rungs cost nothing
  and a plain part costs less.
- **A parametric hinge.** `HINGE_THICKNESS` is one constant, and the right
  value depends on the filament and the size — PLA at 25mm and PETG at 100mm
  do not want the same skin. Worth exposing rather than deciding for people.
  Note `HINGE_GAP_PER_RAD` and `MIN_HINGE_GAP` are tied to it: the gap has to
  grow with the thickness or the walls collide before the fold closes, so this
  is a parameter with a *check*, not a free number.
- **Emit a parametric `.scad`.** The computed strip as `polyhedron()` data,
  the features as parameters at the top. Hands the magnet-dimension problem to
  the people who have opinions about it, for the cost of one generator.
- **Generate in the browser, with manifold3d's WASM build.** The honest
  long-term answer. Prebuilding does not scale — 6 solids x 5 sizes x magnets
  x inlays is 120 files, and that is before magnet *diameter*, which is
  continuous and cannot be enumerated at all. manifold3d is the same library
  the Python already uses, so this is porting our 1124 lines onto the same
  kernel rather than reimplementing a CSG engine. And `check_fold` is itself
  manifold operations, so **the verification travels with the generator**: the
  page could fold both halves, measure the overlap and refuse to hand over a
  file that fails — checking the exact bytes the user receives, which is
  better than today, where we verify a representative set and ship those.

  This supersedes the "migrating the generators to JavaScript" entry below in
  one respect: the argument that the checks must be able to fail a build was
  weaker than it was stated. It still holds for the *build* — artefacts in the
  repo should stay verified offline — but it is not an argument against the
  page also being able to generate.

**Untested physical claim:** the hinge is a fixed thickness in millimetres, so
a much larger model puts proportionally more load on the same hinge. The
current ladder is test-printed; 80-100mm rungs want a test print before they
are offered.

## Later

- **The lights over-drive the brightest faces _on the solid_.** The diagram is
  fixed: its faces are coplanar, so the lights could only ever apply one
  multiplier to all of them, and that multiplier was 1.65. The keys now fade
  out over the explosion while the ambient rises to the value that renders a
  surface at exactly its own colour, so the finished diagram is flat and
  measures the palette's red and teal to within one count. The solid still
  clips. Measured on the icosahedron:
  the red cap renders at `(255, 94, 86)` — the red channel pinned, so the
  measured saturation falls from 0.66 to 0.51 purely because the channel has
  no room left. It clips on the *solid*, not only in the diagram; the old
  half-opacity fill was pulling the diagram back under the ceiling and hiding
  it. The cube does not clip, because its faces sit at angles that never face
  both key lights at once. The fix is to scale each palette's `amb`/`k1`/`k2`
  so the worst case lands just under 255 — about 0.87× for standard — but it
  retunes a look that has been adjusted by eye for a long time, so it wants
  doing deliberately rather than in passing.
- **The plain palette reads brighter than "plain" should.** `0x9ba3b0` lit
  face-on comes out close to white, which is clean but not boring. One
  constant, whenever the mood is right for it.

- **Tutte pins the view face to a *regular* polygon, which is arbitrary.**
  Pinning it to the face's own shape is more correct in general. It changes
  nothing for the five solids with regular faces and would unbend the
  trapezohedron's lopsided diagram. Not special-casing: removing an
  assumption that only *shows* on one solid.
- **True perspective projection** as an alternative to Tutte. A Schlegel
  diagram really is a projection from a point just outside a face, and for a
  convex polytope a viewpoint close enough to the facet is always
  crossing-free. It would make the morph literally honest. The cost is that
  projection crowds the middle, where Tutte spreads things evenly — so this
  is a trade, not an upgrade.
- **Scrub bar for the trace**, to drag through the walk rather than watch it.
- **Filament-matching colours** — pick your two filament colours and see the
  print. The *only* version of colour-picking worth building (see below).
- **A shared JS module for geometry the page animates** (Tutte, the unfold),
  imported by node in a build check that asserts it agrees with Python.
  Turns "two sources of truth" into "one source, two consumers, verified".

## Decided against

- **A random "new cycle" button.** On four of six solids there is only one
  cycle up to symmetry, so the button would produce a rotation of what is
  already there — a control that appears to work while doing nothing. On the
  icosahedron 93% of cycles do not fold into strips, so it would routinely
  contradict the rest of the page. The multiplicity is worth *stating*, not
  shuffling.
- **General colour pickers for the two-colour palette.** Each palette carries
  its own lighting, tuned so the caps read against that background and
  against the orange seam. Free colour wells make bad results look like our
  bug. Filament matching is the version with a reason.
- **Migrating the generators to JavaScript.** The checks have to be able to
  fail a build (`check_fold` folds both halves and measures the overlap
  volume); every downloadable file is verified before it ships. The
  duplication that actually cost us time was the *data*, not the language.
- **Nesting camera-follow under "draw the cycle".** It works standalone —
  with the trace off it flies the cycle on its own — so nesting would
  advertise a dependency that does not exist and remove a mode.
- **Dropping the trapezohedron.** It is the only non-Platonic solid here,
  which is what makes the page's claim about *Hamiltonicity* rather than
  regularity; and it is the only solid with two genuinely different printable
  cuts. Its diagram is lopsided and its closest vertex pair is about half the
  next worst — but at 38px on an 800px canvas it is still a bigger target
  than most buttons on the page. Keep it; do not accommodate it.

## What we know

Computed with `hardware/scripts/`; the enumeration agrees with the
literature (30 Hamiltonian cycles on the dodecahedron — Hamilton's own
icosian game — and 2560 directed, 1280 undirected, on the icosahedron).

"Forms" counts cycles up to the solid's full symmetry group, rotations and
reflections, which for a 3-connected planar graph is exactly its graph
automorphism group (Whitney).

| solid | \|Aut\| | cycles | forms | fold into strips | forms |
|---|---:|---:|---:|---:|---:|
| tetrahedron | 24 | 3 | 1 | 3 | 1 |
| cube | 48 | 6 | 1 | 6 | 1 |
| octahedron | 48 | 16 | 2 | 12 | 1 |
| trapezohedron | 20 | 20 | 2 | 20 | 2 |
| dodecahedron | 120 | 30 | 1 | 30 | 1 |
| icosahedron | 120 | 1280 | 17 | 90 | 2 |

A cap's faces form a **path** — a strip — or a branching tree. That test is
what collapses the icosahedron's 17 forms to 2, and it is what the printed
part needs, because the part is a linear ladder of faces on living hinges.

It is **not** a test for lying flat, although this file and `census.py` both
used to say it was. Every cap of every Hamiltonian cycle lies flat. A cycle
puts two cut edges at every vertex, so the fan of faces around each vertex is
broken into an arc and no cap ever closes a wheel — no cap has an interior
vertex, and for a disc a dual cycle is exactly what encircles one, so the dual
is always a tree rather than merely connected. A tree rolls out isometrically
whatever its shape.

What can still go wrong is the net **colliding with itself** far from where it
branched, and that cuts across the path/branching line rather than along it:

| icosahedron caps | lay out clean | self-overlap |
|---|---:|---:|
| path-shaped | 338 | 22 |
| branching | 1985 | 215 |

15 of those 22 belong to cycles whose *both* caps are strips. So being a strip
is neither necessary for a flat net nor sufficient for one, and the only
honest test is to lay the net out and measure it — which is what
`hardware/scripts/check_unfold.py` does. On the icosahedron the collisions are
exact rather than marginal, because equilateral triangles roll out onto the
triangular lattice and two cells either coincide or miss entirely.

The cut we ship is, on every solid, one whose two halves are **congruent** —
which is what lets a single STL be printed twice. On the icosahedron that
condition is decisive: of the two strip forms, one has congruent halves (30
cuts) and one does not (60 cuts), and we ship the rarer. Only the
trapezohedron offers a real choice: both of its forms give congruent halves.

## Open questions

- How much help while drawing? Marking vertices that have become unreachable
  is the "trapped" lesson made live; warning that *no completion exists* is
  cheap at 12–20 vertices but may steal the discovery.
