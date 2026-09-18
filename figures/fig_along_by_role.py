"""Figure: along_t distribution by task role.

(a) FP borrowed — box plots by role (base, K, nearest, other)
(b) FP epiotic  — box plots by role
(c) TP          — box plots by role

K = non-base task with highest alignment (argmax along_by_task excl base).
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..")
FP_FILE = os.path.join(DATA_ROOT, "endpoint_density_v2.jsonl")
TP_FILE = os.path.join(DATA_ROOT, "endpoint_density_tp_v2.jsonl")
if not os.path.isfile(FP_FILE):
    FP_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/endpoint_density_v2.jsonl"
    TP_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/endpoint_density_tp_v2.jsonl"

COL_BASE = "#2a78d6"
COL_K = "#eb6834"
COL_NEAR = "#1baf7a"
COL_OTHER = "#898781"
ROLES = ["base", "K", "nearest", "other"]
COLORS = {"base": COL_BASE, "K": COL_K, "nearest": COL_NEAR, "other": COL_OTHER}


def load(path, filter_fp=None):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if filter_fp is not None and r.get("is_fp") != filter_fp:
                continue
            out.append(r)
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


def collect_along(records, split_key):
    out = defaultdict(lambda: defaultdict(list))
    for r in records:
        abt = r.get("along_by_task") or {}
        if not abt:
            continue
        sv = r.get(split_key)
        if sv is None:
            continue
        base_tid = identify_base(r)
        K_tid = str(r["K"]) if r.get("K") is not None else None
        nearest_tid = str(r["nearest_task"]) if r.get("nearest_task") is not None else None
        for tid, val in abt.items():
            if tid == base_tid:
                out[sv]["base"].append(val)
            if tid == K_tid:
                out[sv]["K"].append(val)
            if tid == nearest_tid:
                out[sv]["nearest"].append(val)
            if tid != base_tid and tid != K_tid and tid != nearest_tid:
                out[sv]["other"].append(val)
    return out


def collect_along_all(records):
    out = defaultdict(list)
    for r in records:
        abt = r.get("along_by_task") or {}
        if not abt:
            continue
        base_tid = identify_base(r)
        K_tid = str(r["K"]) if r.get("K") is not None else None
        nearest_tid = str(r["nearest_task"]) if r.get("nearest_task") is not None else None
        for tid, val in abt.items():
            if tid == base_tid:
                out["base"].append(val)
            if tid == K_tid:
                out["K"].append(val)
            if tid == nearest_tid:
                out["nearest"].append(val)
            if tid != base_tid and tid != K_tid and tid != nearest_tid:
                out["other"].append(val)
    return out


def draw_boxpanel(ax, data_by_role, title, panel_label):
    positions = []
    box_data = []
    colors = []
    labels = []
    for i, role in enumerate(ROLES):
        vals = data_by_role.get(role, [])
        box_data.append(vals if vals else [0])
        positions.append(i)
        colors.append(COLORS[role])
        labels.append(f"{role}\n($n$={len(vals)})")

    bp = ax.boxplot(box_data, positions=positions, widths=0.55,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    for i, role in enumerate(ROLES):
        vals = data_by_role.get(role, [])
        if vals:
            med = np.median(vals)
            ax.text(i + 0.32, med, f"{med:.2f}", va="center", ha="left",
                    fontsize=7.5, color="black", fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_title(f"({panel_label})  {title}", fontsize=10, pad=6)


def main():
    fps = load(FP_FILE, filter_fp=1)
    tps = load(TP_FILE)

    fp_along = collect_along(fps, "subset")
    tp_along = collect_along_all(tps)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)

    draw_boxpanel(axes[0], fp_along.get("borrowed", {}), "FP borrowed", "a")
    draw_boxpanel(axes[1], fp_along.get("epotic", {}), "FP epiotic", "b")
    draw_boxpanel(axes[2], tp_along, "TP", "c")
    axes[0].set_ylabel(r"$\mathrm{along}_t(\delta)$", fontsize=10)

    legend_handles = [
        Patch(facecolor=COLORS[r], alpha=0.6, edgecolor="black",
              linewidth=0.6, label=r)
        for r in ROLES
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.05))

    fig.text(0.5, -0.02,
             r"$K$ = non-base task with max $\mathrm{along}$ (argmax excl base)",
             ha="center", fontsize=9, style="italic", color="#555")

    out_dir = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fig_along_by_role.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
