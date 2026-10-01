# Roadmap

What is coming, what is parked, and — most usefully — what was considered and
turned down, so it does not get relitigated.

## Next

In this order. Each step is chosen so the one after it is not built on
something we already intend to change.

1. **One source for the polyhedra.** `polyhedra.py` is canonical and
   self-verifying; the page reads a generated `polyhedra.js`. *(in progress)*
2. **Tabs.** Solid · Diagram · Make it. Per-tab sidebar, which retires the
   greying-out. Downloads move from the 280px sidebar into the main area.
   Trace and paint become separate controls, camera-follow stays a sibling
   rather than a child. Reset view in a canvas corner. URL state
   (`#solid/tab/palette`) folded in here rather than retrofitted.
3. **Colour in the diagram.** The two caps filled, and the outer region
   tinted in its cap's colour — see *the outer face* below. A census line
   under the stats.
4. **Drawing the cycle.** Drag across vertices in the Diagram tab; the solid
   splits when the loop closes. Brings with it the "Make it" explanation of
   why a given cut is or is not printable, which cannot be tested before
   then because until now every cut has been the good one.
5. **The unfold.** The Make it tab's main area becomes a canvas, and the
   solid unfolds into its two flat strips.

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

A cap unfolds flat as one strip exactly when its faces form a **path**. If
they form a branching tree there is no single chain to roll out. That test
is what collapses the icosahedron's 17 forms to 2.

The cut we ship is, on every solid, one whose two halves are **congruent** —
which is what lets a single STL be printed twice. On the icosahedron that
condition is decisive: of the two strip forms, one has congruent halves (30
cuts) and one does not (60 cuts), and we ship the rarer. Only the
trapezohedron offers a real choice: both of its forms give congruent halves.

## Open questions

- Should the final diagram keep its faces filled, or fade to a line drawing?
  The blank version already exists as the printable Schlegel PDF, which
  argues for filling the one on screen.
- How much help while drawing? Marking vertices that have become unreachable
  is the "trapped" lesson made live; warning that *no completion exists* is
  cheap at 12–20 vertices but may steal the discovery.
