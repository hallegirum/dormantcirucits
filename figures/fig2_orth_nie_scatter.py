"""Fig 2 — Orth vs |NIE| scatter: divergence is deceptive.

2x2 layout: swap (within), swap (cross), push, Gaussian push.
TN and TP share blue (non-dormant) with different markers/median shapes;
FP is red (dormant).  Two arrows per panel:
  TN → FP  (non-dormant to dormant: orth increases)
  TP → FP  (non-dormant to dormant: orth DECREASES — the deception)

Usage:
    python figures/fig2_orth_nie_scatter.py [--csv PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base = row.get("base_task")
    target = row.get("target_task")
    within = (pd.notna(target) and base == target)
    if push == "input":
        return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self":
        return f"{strategy}_self"
    if strategy == "STR":
        return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def classify(row, thresholds):
    combo = row["combo"]
    thr = thresholds.get(combo)
    if thr is None or pd.isna(row["abs_NIE"]):
        return None
    above = row["abs_NIE"] >= thr
    in_circ = row["in_base_circuit"]
    if in_circ and above:
        return "TP"
    if in_circ and not above:
        return "FN"
    if not in_circ and above:
        return "FP"
    return "TN"


PANELS = [
    ("STR swap (within)", ["STR_swap_within"]),
    ("STR swap (cross)",  ["STR_swap_cross"]),
    ("STR push",          ["STR_push_within", "STR_push_cross"]),
    ("Gaussian push",     ["Gaussian_push"]),
]

# scatter styling
SCATTER_CFG = {
    "TN": dict(color="#4C72B0", marker="o", s=4,  alpha=0.08, zorder=1),
    "TP": dict(color="#4C72B0", marker="^", s=14, alpha=0.35, zorder=3),
    "FP": dict(color="#d62728", marker="x", s=14, alpha=0.40, zorder=2, linewidths=0.7),
}

# median marker shapes — distinct per class
MEDIAN_CFG = {
    "TN": dict(marker="o", color="#4C72B0", markersize=10),
    "TP": dict(marker="^", color="#4C72B0", markersize=11),
    "FP": dict(marker="X", color="#d62728", markersize=11),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out", default="figures/fig2_orth_nie_scatter.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)
    df = df[df["bucket"] != "Gaussian_self"].copy()

    in_circ = df[df["in_base_circuit"] == True]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    df["gt_class"] = df.apply(lambda r: classify(r, thresholds), axis=1)
    df = df.dropna(subset=["gt_class", "orth_base", "abs_NIE"])
    df = df[df["gt_class"] != "FN"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
    axes = axes.flatten()

    for ax, (title, buckets) in zip(axes, PANELS):
        sub = df[df["bucket"].isin(buckets)]

        orth_clip = sub["orth_base"].quantile(0.98)

        # scatter points
        for cls in ["TN", "FP", "TP"]:
            d = sub[sub["gt_class"] == cls]
            cfg = SCATTER_CFG[cls]
            ax.scatter(d["orth_base"], d["abs_NIE"],
                       c=cfg["color"], marker=cfg["marker"], s=cfg["s"],
                       alpha=cfg["alpha"], zorder=cfg["zorder"],
                       linewidths=cfg.get("linewidths", 0.5),
                       edgecolors="none" if cls != "FP" else cfg["color"],
                       rasterized=True)

        # median markers with labels
        medians = {}
        for cls in ["TN", "FP", "TP"]:
            d = sub[sub["gt_class"] == cls]
            if len(d) < 5:
                continue
            med_orth = d["orth_base"].median()
            med_nie = d["abs_NIE"].median()
            medians[cls] = (med_orth, med_nie)
            mcfg = MEDIAN_CFG[cls]
            ax.plot(med_orth, med_nie, marker=mcfg["marker"],
                    markersize=mcfg["markersize"], color=mcfg["color"],
                    markeredgecolor="black", markeredgewidth=1.0, zorder=10)
            dy = 10 if cls != "TN" else -14
            ax.annotate(cls, (med_orth, med_nie),
                       textcoords="offset points", xytext=(10, dy),
                       fontsize=9, fontweight="bold",
                       color=mcfg["color"], zorder=11)

        # arrows: non-dormant → dormant
        # TN → FP (orth increases toward dormancy in M3L direction)
        if "TN" in medians and "FP" in medians:
            tn_x, tn_y = medians["TN"]
            fp_x, fp_y = medians["FP"]
            ax.annotate("", xy=(fp_x, fp_y), xytext=(tn_x, tn_y),
                       arrowprops=dict(arrowstyle="->", color="0.45",
                                       lw=1.5, connectionstyle="arc3,rad=-0.2"),
                       zorder=9)

        # TP → FP (non-dormant to dormant: orth DECREASES — the deception)
        if "TP" in medians and "FP" in medians:
            tp_x, tp_y = medians["TP"]
            fp_x, fp_y = medians["FP"]
            ax.annotate("", xy=(fp_x, fp_y), xytext=(tp_x, tp_y),
                       arrowprops=dict(arrowstyle="->", color="0.45",
                                       lw=1.5, connectionstyle="arc3,rad=0.2"),
                       zorder=9)

        ax.set_xlabel("$\\mathrm{orth}_b(\\delta)$", fontsize=10)
        ax.set_title(title, fontsize=10.5, pad=8)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlim(0, orth_clip * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("|NIE|  (symlog)", fontsize=10)
    axes[2].set_ylabel("|NIE|  (symlog)", fontsize=10)

    # shared legend
    handles = [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#4C72B0",
                   markersize=7, label="TP (non-dormant)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C72B0",
                   markersize=6, label="TN (non-dormant)"),
        plt.Line2D([0], [0], marker="x", color="#d62728", markeredgewidth=1.5,
                   markersize=7, label="FP (dormant)", linestyle="None"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9.5,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(
        "Divergence is Deceptive: $\\mathrm{orth}_b(\\delta)$ correlates "
        "with dormancy in opposing directions",
        fontsize=12, y=1.04)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
