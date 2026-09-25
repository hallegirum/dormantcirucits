"""Schematic: two pathways to intervention-induced dormancy.

Self-induced vs coupled pathways with the three instrument families.
Curvature shown both locally (gate crossing) and propagating through
the residual stream.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, FancyBboxPatch, FancyArrowPatch,
                                Arc)
from matplotlib.lines import Line2D
from matplotlib.path import Path as MPath
import matplotlib.patches as mpatches

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"

COL_S = "#2a78d6"
COL_DIV = "#2a78d6"
COL_CURV = "#2ca02c"
COL_RIP = "#9467bd"


def _box(ax, xy, w, h, fc="white", ec="#444", lw=1.0, r=0.18, z=2):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def _arrow(ax, p0, p1, color="#444", lw=1.4, ls="-", z=5, rad=0.0, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color, lw=lw,
                                 linestyle=ls, mutation_scale=ms, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}"))


def _curved_gate(ax, x_center, y_bot, y_top, bulge=0.4, n=80):
    """Draw a smooth curved nonlinear boundary (gate)."""
    t = np.linspace(0, 1, n)
    y = y_bot + (y_top - y_bot) * t
    x = x_center + bulge * np.sin(np.pi * t) ** 1.5

    ax.plot(x, y, color=COL_CURV, ls="-", lw=2.0, zorder=4)

    shade_x = np.concatenate([x, [x_center - 0.8] * n])
    shade_y = np.concatenate([y, y[::-1]])
    ax.fill(shade_x, shade_y, color=COL_CURV, alpha=0.06, zorder=1)

    mid = n // 2
    dx = x[mid + 1] - x[mid - 1]
    dy = y[mid + 1] - y[mid - 1]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    nx, ny = -dy / norm, dx / norm
    tick_len = 0.15
    for i in range(5, n - 5, n // 6):
        dxi = x[i + 1] - x[i - 1]
        dyi = y[i + 1] - y[i - 1]
        ni = np.sqrt(dxi ** 2 + dyi ** 2)
        tnx, tny = -dyi / ni, dxi / ni
        ax.plot([x[i], x[i] - tick_len * tnx],
                [y[i], y[i] - tick_len * tny],
                color=COL_CURV, lw=0.8, zorder=4)

    return x, y


def main():
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.5, 12.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Two pathways to intervention-induced dormancy",
                 fontsize=12, fontweight="bold", pad=12)

    # ── residual stream (central vertical column) ──
    stream_x = 7.0
    stream_w = 1.0
    ax.add_patch(FancyBboxPatch((stream_x - stream_w / 2, 1.0), stream_w, 9.0,
                                boxstyle="round,pad=0,rounding_size=0.35",
                                fc="#ececec", ec="#c8c8c8", lw=0.8, zorder=1))
    ax.text(stream_x, 0.5, "residual stream", ha="center", va="center",
            fontsize=9, color="#555")

    # upward arrow to task output
    _arrow(ax, (stream_x, 10.0), (stream_x, 10.8), color="#444", lw=1.8, ms=14)
    ax.text(stream_x, 11.3, "task output  →  attributed effect",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # ── ripple: perturbation propagating up the stream (purple wavy) ──
    yy_rip = np.linspace(4.3, 9.8, 400)
    xx_rip = stream_x + 0.22 * np.sin((yy_rip - 4.3) * 2 * np.pi / 0.85)
    ax.plot(xx_rip, yy_rip, color=COL_RIP, lw=1.8, zorder=3)

    # ── curvature propagating up the stream (green dashed wavy) ──
    yy_curv = np.linspace(4.3, 9.8, 400)
    xx_curv = stream_x - 0.22 * np.sin((yy_curv - 4.3) * 2 * np.pi / 1.1 + 0.5)
    ax.plot(xx_curv, yy_curv, color=COL_CURV, lw=1.6, ls="--", zorder=3)

    # label the two propagation channels
    ax.text(stream_x + 0.55, 8.0, "ripple", fontsize=7, color=COL_RIP,
            ha="left", va="center", rotation=90)
    ax.text(stream_x - 0.55, 8.0, "curv.", fontsize=7, color=COL_CURV,
            ha="right", va="center", rotation=90)

    # ═════════════════════════════════════════════════════════
    # LEFT: self-induced pathway — measured component
    # ═════════════════════════════════════════════════════════
    _box(ax, (0.3, 1.6), 5.8, 5.0, fc="white", ec="#444", lw=1.2)
    ax.text(3.2, 6.35, "measured component $c$, layer $\\ell$",
            ha="center", va="center", fontsize=9.5, fontweight="bold")

    # natural-geometry ellipse
    ell_c = (2.3, 4.0)
    ell_w, ell_h = 2.2, 1.4
    ax.add_patch(Ellipse(ell_c, ell_w, ell_h, fc=COL_S, ec=COL_S,
                         alpha=0.12, lw=0.9, zorder=3))
    ax.add_patch(Ellipse(ell_c, ell_w, ell_h, fc="none", ec=COL_S,
                         lw=1.2, zorder=3))
    ax.text(2.3, 2.9, "natural activation\ngeometry", ha="center", va="top",
            fontsize=7.5, color=COL_S, style="italic")

    # natural point h (inside ellipse)
    p_nat = (2.0, 4.05)
    ax.scatter([p_nat[0]], [p_nat[1]], s=50, c=COL_S, zorder=7, edgecolors="white", lw=0.5)
    ax.text(p_nat[0] - 0.15, p_nat[1] - 0.35, "$h$", fontsize=9, color=COL_S,
            ha="center", va="top", fontweight="bold")

    # curved nonlinear gate boundary
    gate_x, gate_y = _curved_gate(ax, x_center=3.6, y_bot=2.4, y_top=5.8,
                                  bulge=0.55)
    ax.text(3.05, 2.15, "nonlinear\nboundary", ha="center", va="top",
            fontsize=7, color=COL_CURV, fontweight="bold")

    # displaced point h' (outside the gate)
    p_int = (4.8, 4.75)
    ax.scatter([p_int[0]], [p_int[1]], s=50, c="#222", zorder=7, edgecolors="white", lw=0.5)
    ax.text(p_int[0] + 0.2, p_int[1] + 0.1, "$h'$", fontsize=9, color="#222",
            ha="left", va="center", fontweight="bold")

    # δ arrow from h to h' (displacement)
    _arrow(ax, p_nat, p_int, color=COL_DIV, lw=2.0, z=6, ms=14)
    ax.text(3.1, 4.85, r"$\delta$", fontsize=11, color=COL_DIV,
            ha="center", va="center", fontweight="bold")

    # crossing indicator: small "X" or burst where δ crosses the gate
    cross_y = 4.35
    cross_idx = np.argmin(np.abs(np.array([y_bot + (5.8 - 2.4) * t
                                           for y_bot in [2.4]
                                           for t in np.linspace(0, 1, 80)]) - cross_y))
    gate_cross_x = 3.6 + 0.55 * np.sin(np.pi * (cross_y - 2.4) / (5.8 - 2.4)) ** 1.5
    ax.scatter([gate_cross_x], [cross_y], s=100, marker="*", c=COL_CURV,
              zorder=8, edgecolors="none")

    # connect measured component to residual stream
    _arrow(ax, (6.1, 3.8), (stream_x - stream_w / 2, 3.8), color=COL_DIV, lw=1.8)
    ax.text(6.35, 4.15, "writes $\\delta$\nto stream", ha="center", va="bottom",
            fontsize=7, color=COL_DIV)

    # self-induced label
    ax.text(-0.2, 3.9, "self-\ninduced", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")

    # ═════════════════════════════════════════════════════════
    # RIGHT: coupled pathway — downstream / interacting components
    # ═════════════════════════════════════════════════════════
    _box(ax, (8.5, 5.3), 5.5, 4.5, fc="white", ec="#444", lw=1.2)
    ax.text(11.25, 9.15, "downstream / interacting\ncomponents (layers $> \\ell$)",
            ha="center", va="center", fontsize=9, fontweight="bold")

    # natural-geometry ellipse for downstream component
    ell2_c = (10.2, 7.1)
    ell2_w, ell2_h = 1.8, 1.1
    ax.add_patch(Ellipse(ell2_c, ell2_w, ell2_h, fc=COL_S, ec=COL_S,
                         alpha=0.12, lw=0.9, zorder=3))
    ax.add_patch(Ellipse(ell2_c, ell2_w, ell2_h, fc="none", ec=COL_S,
                         lw=1.0, zorder=3))

    # natural point (downstream)
    p2_nat = (10.0, 7.15)
    ax.scatter([p2_nat[0]], [p2_nat[1]], s=35, c=COL_S, zorder=7, edgecolors="white", lw=0.5)
    ax.text(p2_nat[0] - 0.1, p2_nat[1] - 0.25, "$h_d$", fontsize=7.5,
            color=COL_S, ha="center", va="top")

    # curved gate for downstream component
    gate2_x, gate2_y = _curved_gate(ax, x_center=11.3, y_bot=6.0, y_top=8.5,
                                    bulge=0.4)
    ax.text(11.0, 5.7, "boundary", ha="center", va="top",
            fontsize=6.5, color=COL_CURV)

    # displaced point (downstream, outside gate)
    p2_int = (12.3, 7.65)
    ax.scatter([p2_int[0]], [p2_int[1]], s=35, c="#222", zorder=7, edgecolors="white", lw=0.5)
    ax.text(p2_int[0] + 0.15, p2_int[1] + 0.05, "$h'_d$", fontsize=7.5,
            color="#222", ha="left", va="center")

    # δ_d arrow
    _arrow(ax, p2_nat, p2_int, color=COL_DIV, lw=1.6, z=6, ms=11)
    ax.text(10.85, 7.75, r"$\delta_d$", fontsize=8.5, color=COL_DIV,
            ha="center", va="center", fontweight="bold")

    # crossing star for downstream gate
    cross2_y = 7.35
    gate2_cross_x = 11.3 + 0.4 * np.sin(np.pi * (cross2_y - 6.0) / (8.5 - 6.0)) ** 1.5
    ax.scatter([gate2_cross_x], [cross2_y], s=70, marker="*", c=COL_CURV,
              zorder=8, edgecolors="none")

    # ripple arrow from stream to downstream box
    _arrow(ax, (stream_x + stream_w / 2, 7.0), (8.5, 7.0), color=COL_RIP, lw=2.0, ms=14)
    ax.text(8.0, 7.3, "ripple", ha="center", va="bottom", fontsize=7.5, color=COL_RIP,
            fontweight="bold")

    # curvature also arrives from stream
    _arrow(ax, (stream_x + stream_w / 2, 6.2), (8.5, 6.2), color=COL_CURV, lw=1.6,
           ls="--", ms=11)
    ax.text(8.0, 6.45, "curvature\npropagates", ha="center", va="bottom",
            fontsize=6.5, color=COL_CURV)

    # coupled label
    ax.text(14.3, 7.55, "coupled", rotation=270, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333")

    # ── instrument families legend ──
    handles = [
        Line2D([0], [0], color=COL_DIV, lw=2.0,
               label=r"divergence ($\delta$): displacement off natural geometry"),
        Line2D([0], [0], color=COL_CURV, lw=2.0,
               label="curvature: crossing / propagation of nonlinear boundary"),
        Line2D([0], [0], color=COL_RIP, lw=2.0,
               label="interaction: coupling via residual stream"),
        Line2D([0], [0], color=COL_CURV, lw=0, marker="*", markersize=10,
               markerfacecolor=COL_CURV, markeredgecolor="none",
               label="gate crossing"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.07),
              fontsize=8, frameon=True, framealpha=0.95, edgecolor="#ddd",
              title="instrument families", title_fontsize=9, handlelength=1.6,
              borderpad=0.6, ncol=2)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_dormancy_schematic.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
