"""Figure: INT × PIE interaction taxonomy.

2×2 core grid (PIE > 0 / ≈ 0  vs  INT ≈ 0 / > 0) with greyed-out
margins for PIE < 0 and INT < 0 (TN / FN territory).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"

COL_CELL1 = "#d5ecd4"   # green — trustworthy
COL_CELL2 = "#fde8c8"   # amber — context-modulated
COL_CELL3 = "#f8d0d0"   # red-ish — dormancy
COL_CELL4 = "#e8e8e8"   # grey — inert
COL_MARGIN = "#f2f2f2"  # light grey — TN/FN


def _cell(ax, x, y, w, h, fc, ec="#999", lw=1.2, r=0.15, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.set_xlim(-2.5, 11)
    ax.set_ylim(-2.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    cw, ch = 4.8, 3.0  # cell width, height
    gap = 0.25
    x0 = 0.0            # left column start
    x1 = x0 + cw + gap  # right column start
    y1 = 4.0             # top row start
    y0 = y1 - ch - gap   # bottom row start

    # ── 2×2 core cells ──
    # Cell 1: PIE > 0, INT ≈ 0  (top-left)
    _cell(ax, x0, y1, cw, ch, COL_CELL1)
    ax.text(x0 + cw / 2, y1 + ch - 0.35, "Cell 1", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#2d6a2e")
    ax.text(x0 + cw / 2, y1 + ch / 2 - 0.1,
            "clean self-contained\nmechanism",
            ha="center", va="center", fontsize=9.5, color="#333")
    ax.text(x0 + cw / 2, y1 + 0.35, "(trustworthy)",
            ha="center", va="bottom", fontsize=8.5, color="#2d6a2e",
            style="italic")

    # Cell 2: PIE > 0, INT > 0  (top-right)
    _cell(ax, x1, y1, cw, ch, COL_CELL2)
    ax.text(x1 + cw / 2, y1 + ch - 0.35, "Cell 2", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#b8760a")
    ax.text(x1 + cw / 2, y1 + ch / 2 - 0.1,
            "real but\ncontext-modulated",
            ha="center", va="center", fontsize=9.5, color="#333")
    ax.text(x1 + cw / 2, y1 + 0.35, "(rank-unstable; backup / boost)",
            ha="center", va="bottom", fontsize=8, color="#b8760a",
            style="italic")

    # Cell 4: PIE ≈ 0, INT ≈ 0  (bottom-left)
    _cell(ax, x0, y0, cw, ch, COL_CELL4)
    ax.text(x0 + cw / 2, y0 + ch - 0.35, "Cell 4", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#555")
    ax.text(x0 + cw / 2, y0 + ch / 2 - 0.1,
            "inert / harmless\ndivergence",
            ha="center", va="center", fontsize=9.5, color="#333")
    ax.text(x0 + cw / 2, y0 + 0.35, "(Class 1 false positive)",
            ha="center", va="bottom", fontsize=8, color="#555",
            style="italic")

    # Cell 3: PIE ≈ 0, INT > 0  (bottom-right)
    _cell(ax, x1, y0, cw, ch, COL_CELL3)
    ax.text(x1 + cw / 2, y0 + ch - 0.35, "Cell 3", ha="center", va="top",
            fontsize=10, fontweight="bold", color="#a83232")
    ax.text(x1 + cw / 2, y0 + ch / 2 + 0.15,
            "dormancy",
            ha="center", va="center", fontsize=10.5, color="#a83232",
            fontweight="bold")
    ax.text(x1 + cw / 2, y0 + ch / 2 - 0.55,
            "3a: healthy interaction-\n      dependence\n"
            "3b: pernicious borrowing",
            ha="center", va="center", fontsize=8.5, color="#333",
            family="monospace")

    # ── margin cells: PIE < 0 row (below) ──
    margin_h = 1.5
    margin_y = y0 - gap - margin_h
    _cell(ax, x0, margin_y, 2 * cw + gap, margin_h, COL_MARGIN, ec="#ccc")
    ax.text(x0 + cw + gap / 2, margin_y + margin_h / 2,
            "PIE $< 0$  →  TN or FN",
            ha="center", va="center", fontsize=10, color="#888")

    # ── margin cells: INT < 0 column (left, but visually to the right as note) ──
    # Show as a note below the grid since INT < 0 is symmetric to INT > 0
    # but with suppression semantics → TN / FN
    # Actually place it as a side column
    margin_w = 1.6
    margin_x = x1 + cw + gap
    # spans both rows
    # not needed — the user's table shows INT < 0 only has TN/FN

    # ── axis labels ──
    # INT axis (horizontal, top)
    ax.annotate("", xy=(x1 + cw, y1 + ch + 1.0), xytext=(x0, y1 + ch + 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
    ax.text(x0 - 0.1, y1 + ch + 1.0, "INT $\\approx 0$",
            ha="right", va="center", fontsize=10, color="#444")
    ax.text(x1 + cw + 0.1, y1 + ch + 1.0, "INT $> 0$",
            ha="left", va="center", fontsize=10, color="#444", fontweight="bold")
    ax.text((x0 + x1 + cw) / 2, y1 + ch + 1.65,
            "interaction effect (INT)",
            ha="center", va="center", fontsize=11, fontweight="bold")

    # PIE axis (vertical, left)
    ax.annotate("", xy=(-1.3, y1 + ch), xytext=(-1.3, y0),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5))
    ax.text(-1.3, y0 - 0.15, "PIE $\\approx 0$",
            ha="center", va="top", fontsize=10, color="#444")
    ax.text(-1.3, y1 + ch + 0.15, "PIE $> 0$",
            ha="center", va="bottom", fontsize=10, color="#444", fontweight="bold")
    ax.text(-2.2, (y0 + y1 + ch) / 2,
            "path-specific\nindirect effect\n(PIE)",
            ha="center", va="center", fontsize=11, fontweight="bold",
            rotation=90)

    # ── title ──
    ax.text((x0 + x1 + cw) / 2, y1 + ch + 2.5,
            "INT $\\times$ PIE interaction taxonomy",
            ha="center", va="center", fontsize=13, fontweight="bold")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_int_pie_grid.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
