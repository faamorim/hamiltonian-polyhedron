"""The printed page itself: its size, and the ruler that proves it printed at
that size.

Both PDF generators used to carry their own copy of the page dimensions,
which is exactly the kind of constant that drifts apart unnoticed -- so they
share this one.
"""

# The overlap of landscape A4 (297 x 210) and landscape US Letter
# (279.4 x 215.9): Letter's width, A4's height. One file then prints at 100%
# on either paper with room on every side, which matters because whoever
# prints these may not be on the same continent as whoever generated them.
# The cost is 5.9mm of height against Letter, and the nets are sized to it.
PAGE_W_MM, PAGE_H_MM = 279.4, 210.0
MM_PER_INCH = 25.4
MARGIN_MM = 10    # safe margin on all sides (most printers can't print edge-to-edge)
HEADER_MM = 15    # vertical space reserved for the title at the top

# NO PRINTED RULER, deliberately, and it was here once. The argument for it
# was that "fit to page" silently rescales by about 5% between A4 and
# Letter, so measure the bar and reprint if it is short. That argument is
# wrong for paper: a net at 95% folds into a solid at 95%, every edge and
# every tab scaled together, and it works exactly as well. The one thing in
# this project that genuinely cannot be scaled is the STL, whose hinge is a
# fixed thickness in millimetres while its faces are not -- and that warning
# belongs on the STL, where it already is.
#
# So the ruler only ever bought a false constraint, and charged for it: a
# reader who measured 47mm reprinted a sheet that was going to be fine. The
# page is still sized to the A4/Letter overlap, so anyone who does print at
# 100% gets the edge length quoted on the web page; anyone who does not gets
# a slightly smaller polyhedron and no worse an afternoon.
