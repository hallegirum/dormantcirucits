"""Figure: TP d_base vs d_K by intervention strategy.

4 panels (2x2): swap cross | swap within | STR push cross | STR push within
Cross panels color points by whether K is a target task in the combo.
Within panels use a standard hexbin.
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from collections import defaultdict

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..")
TP_FILE = os.path.join(DATA_ROOT, "endpoint_density_tp_v2.jsonl")
if not os.path.isfile(TP_FILE):
    TP_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/endpoint_density_tp_v2.jsonl"

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def identify_base(r):
    dpt = r.get("dist_per_task") or {}
    db = r.get("dist_base")
    if db is None or not dpt:
        return None
    for tid, dv in dpt.items():
        if abs(dv - db) < 0.001:
            return str(tid)
    return None


def is_k_target(r):
    base_tid = identify_base(r)
    if base_tid is None:
        return None
    tasks_in_combo = r["combo"].replace("cases_", "").split("+")
    target_tasks = [t for t in tasks_in_combo if t != base_tid]
    K = str(r.get("K", ""))
    return K in target_tasks


tps = load(TP_FILE)

REGIME_LABELS = {
    "swap_cross": "STR swap cross",
    "swap_within": "STR swap within",
    "STR_push_cross": "STR push cross",
    "STR_push_within": "STR push within",
}
PANEL_ORDER = ["swap_cross", "swap_within", "STR_push_cross", "STR_push_within"]
PANEL_LETTERS = ["a", "b", "c", "d"]

COL_K_TARGET = "#eb6834"
COL_K_OTHER = "#2a78d6"

fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
axes_flat = axes.flatten()

for idx, regime in enumerate(PANEL_ORDER):
    ax = axes_flat[idx]
    recs = [r for r in tps if r["regime"] == regime]

    db_all = []
    dk_all = []
    is_cross = "cross" in regime

    if is_cross:
        db_target = []; dk_target = []
        db_other = []; dk_other = []
        for r in recs:
            db = r.get("dist_base")
            dk = r.get("dist_K")
            if db is None or dk is None:
                continue
            db_all.append(db)
            dk_all.append(dk)
            if is_k_target(r):
                db_target.append(db)
                dk_target.append(dk)
            else:
                db_other.append(db)
                dk_other.append(dk)

        n_target = len(db_target)
        n_other = len(db_other)
        n_total = n_target + n_other
        pct_target = 100 * n_target / n_total if n_total else 0

        ax.scatter(np.log10(np.array(db_other) + 0.1),
                   np.log10(np.array(dk_other) + 0.1),
                   c=COL_K_OTHER, s=12, alpha=0.3, edgecolors="none",
                   rasterized=True, zorder=2, label=f"$K$ $\\neq$ target ({n_other})")
        ax.scatter(np.log10(np.array(db_target) + 0.1),
                   np.log10(np.array(dk_target) + 0.1),
                   c=COL_K_TARGET, s=12, alpha=0.3, edgecolors="none",
                   rasterized=True, zorder=3, label=f"$K$ = target ({n_target}, {pct_target:.0f}%)")

        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_K_TARGET,
                   markersize=7, label=f"$K$ = target ({pct_target:.0f}%)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_K_OTHER,
                   markersize=7, label=f"$K$ $\\neq$ target ({100 - pct_target:.0f}%)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)

    else:
        for r in recs:
            db = r.get("dist_base")
            dk = r.get("dist_K")
            if db is None or dk is None:
                continue
            db_all.append(db)
            dk_all.append(dk)

        db_arr = np.array(db_all)
        dk_arr = np.array(dk_all)
        ax.hexbin(np.log10(db_arr + 0.1), np.log10(dk_arr + 0.1),
                  gridsize=30, cmap="BuPu", mincnt=1, norm=LogNorm(),
                  edgecolors="face", linewidths=0.2)

    ax.plot([-1, 3], [-1, 3], "k--", lw=0.8, alpha=0.5)
    ax.axvline(0, color="#bbb", ls=":", lw=0.6)
    ax.axhline(0, color="#bbb", ls=":", lw=0.6)
    ax.set_xlabel(r"$\log_{10}(d_\mathrm{base})$", fontsize=10)
    ax.set_ylabel(r"$\log_{10}(d_K)$", fontsize=10)

    label = REGIME_LABELS[regime]
    pl = PANEL_LETTERS[idx]
    ax.set_title(f"({pl})  {label}  ($n$={len(db_all)})", fontsize=10, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

    closer_base = sum(1 for a, b in zip(db_all, dk_all) if a < b)
    pct = 100 * closer_base / len(db_all) if db_all else 0
    ax.text(0.97, 0.03,
            f"closer to base: {pct:.0f}%",
            transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ccc", boxstyle="round,pad=0.3"))

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "fig_tp_dk_by_strategy.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved {out}")
