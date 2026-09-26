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

SCALE_BAR_MM = 50.0     # length of the printed ruler
SCALE_TICK_MM = 10.0    # spacing of its ticks


def draw_scale_bar(ax, x=MARGIN_MM, y=PAGE_H_MM - MARGIN_MM - 9.0):
    """A ruler printed on the page, because no page size can defend against
    the real hazard: a print dialog set to 'fit to page'. That silently
    rescales by about 5% between A4 and Letter, which is invisible on the
    drawing but wrong in every edge length -- and these templates are only
    worth anything at 1:1. Measure the bar; if it is not 50mm, reprint.
    """
    ax.plot([x, x + SCALE_BAR_MM], [y, y], color="black", lw=1.0,
            solid_capstyle="butt", zorder=5)
    n = int(round(SCALE_BAR_MM / SCALE_TICK_MM))
    for k in range(n + 1):
        tx = x + k * SCALE_TICK_MM
        ax.plot([tx, tx], [y, y + (2.5 if k in (0, n) else 1.5)],
                color="black", lw=1.0, zorder=5)
    ax.text(x, y - 3.2,
            f"{SCALE_BAR_MM:.0f}mm - measure it. If it is short, the printer "
            f"scaled the page; reprint at 100%, not 'fit to page'.",
            ha="left", va="top", fontsize=6, color="0.35")
