"""along_s vs curvature scatter — FP vs TN (off-circuit population).

2x2: swap(within), swap(cross), STR push, Gaussian push.
Raw feature values, Pearson r, regression line.
Per-strategy best curvature.

Usage:
    python figures/fig5_along_vs_curv_scatter_fptn.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PANELS = [
    ("STR_swap_within",  "STR swap (within)",  "gap_frac",       "downstream curv."),
    ("STR_swap_cross",   "STR swap (cross)",   "gap_frac",       "downstream curv."),
    ("STR_push",         "STR push",           "local_gap_frac", "local curvature"),
    ("Gaussian_push",    "Gaussian push",      "gap_frac",       "downstream curv."),
]


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base, target = row.get("base_task"), row.get("target_task")
    within = pd.notna(target) and base == target
    if push == "input": return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self": return f"{strategy}_self"
    if strategy == "STR": return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_fptn_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False].copy()
    thr = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        t = thr.get(row["combo"])
        return t is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= t
    oc["label"] = oc.apply(is_fp, axis=1).astype(float)
    return oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig5_along_vs_curv_scatter_fptn.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    for i, (bkey, blabel, curv_feat, curv_label) in enumerate(PANELS):
        ax = axes[i]
        if bkey == "STR_push":
            pool = pd.concat([build_fptn_population(df, "STR_push_within"),
                              build_fptn_population(df, "STR_push_cross")])
        else:
            pool = build_fptn_population(df, bkey)

        clean = pool.dropna(subset=["along_nonbase_max", curv_feat, "label"])
        clean = clean[np.isfinite(clean["along_nonbase_max"]) & np.isfinite(clean[curv_feat])]

        tn = clean[clean["label"] == 0]
        fp = clean[clean["label"] == 1]

        ax.scatter(tn["along_nonbase_max"], tn[curv_feat], s=12, alpha=0.3,
                   color="#6A9FCA", marker="o", label=f"TN ({len(tn)})",
                   zorder=2, rasterized=True)
        ax.scatter(fp["along_nonbase_max"], fp[curv_feat], s=18, alpha=0.5,
                   color="#C44E52", marker="x", linewidths=0.8,
                   label=f"FP ({len(fp)})", zorder=3, rasterized=True)

        x_all = clean["along_nonbase_max"].values
        y_all = clean[curv_feat].values
        r, _ = pearsonr(x_all, y_all)

        coeffs = np.polyfit(x_all, y_all, 1)
        x_fit = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), color="grey", ls="--", lw=1.2, zorder=4)

        ax.text(0.95, 0.95, f"r = {r:.2f}",
                transform=ax.transAxes, fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5))

        ax.set_title(blabel, fontsize=10)
        ax.set_xlabel(r"$\mathrm{along}_s(\delta)$", fontsize=9)
        ax.set_ylabel(curv_label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 3:
            ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)

    fig.suptitle(r"$\mathrm{along}_s(\delta)$ vs curvature — FP vs TN (off-circuit)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
