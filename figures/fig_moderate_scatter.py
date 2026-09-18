"""Scatter: orth vs |NIE| and best-ripple vs |NIE| in the moderate-|NIE| band.

Two figures:
  1. orth_base vs abs_NIE, colored TP/FP, one panel per strategy (6 panels)
  2. best ripple metric vs abs_NIE, same layout

Usage:
    python figures/fig_moderate_scatter.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BUCKETS = [
    ("STR_swap_within",  "STR swap (within)"),
    ("STR_swap_cross",   "STR swap (cross)"),
    ("STR_push_within",  "STR push (within)"),
    ("STR_push_cross",   "STR push (cross)"),
    ("Gaussian_push",    "Gaussian push"),
    ("Gaussian_self",    "Gaussian self"),
]

BUCKETS_RIPPLE = [
    ("STR_swap_within",  "STR swap (within)"),
    ("STR_swap_cross",   "STR swap (cross)"),
    ("STR_push_within",  "STR push (within)"),
    ("STR_push_cross",   "STR push (cross)"),
    ("Gaussian_push",    "Gaussian push"),
]

RIPPLE_FEATS = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
RIPPLE_SHORT = {
    "ripple_in_orth": "in",
    "ripple_off_orth": "off",
    "ripple_frac": "in_prop",
    "ripple_xint_in_orth": "xint",
}


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


def build_moderate_band(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    def above(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    pool = pd.concat([in_circ, off_circ])
    pool = pool[pool.apply(above, axis=1)].copy()
    pool_median = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def pick_best_ripple(pool):
    best_auc, best_feat = -1, RIPPLE_FEATS[0]
    for rf in RIPPLE_FEATS:
        clean = pool.dropna(subset=[rf, "label"])
        clean = clean[np.isfinite(clean[rf])]
        if len(clean) < 20 or clean["label"].nunique() < 2:
            continue
        X = clean[[rf]].values
        y = clean["label"].values
        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(X, y)
        a = roc_auc_score(y, lr.predict_proba(X)[:, 1])
        if a > best_auc:
            best_auc, best_feat = a, rf
    return best_feat, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out-orth", default="figures/fig_moderate_orth_scatter.pdf")
    ap.add_argument("--out-ripple", default="figures/fig_moderate_ripple_scatter.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    # --- Figure 1: orth vs |NIE| ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, (bkey, blabel) in enumerate(BUCKETS):
        ax = axes[i]
        pool = build_moderate_band(df, bkey)
        pool = pool.dropna(subset=["orth_base", "abs_NIE"])

        tp = pool[pool["label"] == 0]
        fp = pool[pool["label"] == 1]

        ax.scatter(tp["abs_NIE"], tp["orth_base"], s=8, alpha=0.4,
                   color="#4C72B0", label=f"TP (n={len(tp)})", zorder=2, rasterized=True)
        ax.scatter(fp["abs_NIE"], fp["orth_base"], s=8, alpha=0.4,
                   color="#C44E52", label=f"FP (n={len(fp)})", zorder=2, rasterized=True)

        ax.set_title(blabel, fontsize=10)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.set_xlabel("|NIE|", fontsize=8)
        if i % 3 == 0:
            ax.set_ylabel("orth_base", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Orth vs |NIE| in moderate-|NIE| band (FP vs TP)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out_orth, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out_orth}")

    # --- Figure 2: best ripple vs |NIE| (no Gaussian self) ---
    n_rip = len(BUCKETS_RIPPLE)
    nrows_r = 2
    ncols_r = 3
    fig2, axes2 = plt.subplots(nrows_r, ncols_r, figsize=(14, 9))
    axes2 = axes2.flatten()

    for i, (bkey, blabel) in enumerate(BUCKETS_RIPPLE):
        ax = axes2[i]
        pool = build_moderate_band(df, bkey)

        best_rip, best_auc = pick_best_ripple(pool)
        pool = pool.dropna(subset=[best_rip, "abs_NIE"])

        tp = pool[pool["label"] == 0]
        fp = pool[pool["label"] == 1]

        ax.scatter(tp["abs_NIE"], tp[best_rip], s=8, alpha=0.4,
                   color="#4C72B0", label=f"TP (n={len(tp)})", zorder=2, rasterized=True)
        ax.scatter(fp["abs_NIE"], fp[best_rip], s=8, alpha=0.4,
                   color="#C44E52", label=f"FP (n={len(fp)})", zorder=2, rasterized=True)

        # Add median lines
        tp_med = tp[best_rip].median()
        fp_med = fp[best_rip].median()
        ax.axhline(tp_med, color="#4C72B0", ls="--", lw=1.5, alpha=0.8, zorder=3)
        ax.axhline(fp_med, color="#C44E52", ls="--", lw=1.5, alpha=0.8, zorder=3)
        # Label medians
        xlim_r = ax.get_xlim() if ax.get_xlim() != (0, 1) else (pool["abs_NIE"].min(), pool["abs_NIE"].max())
        x_pos = pool["abs_NIE"].quantile(0.95)
        ax.text(x_pos, tp_med, f" TP med={tp_med:.3f}", fontsize=6.5, color="#4C72B0",
                va="bottom", ha="right", fontweight="bold")
        ax.text(x_pos, fp_med, f" FP med={fp_med:.3f}", fontsize=6.5, color="#C44E52",
                va="top", ha="right", fontweight="bold")

        ax.set_title(f"{blabel}  [{RIPPLE_SHORT[best_rip]}, AUC={best_auc:.2f}]", fontsize=10)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)
        ax.set_xlabel("|NIE|", fontsize=8)
        if i % ncols_r == 0:
            ax.set_ylabel(best_rip, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplot
    for j in range(n_rip, nrows_r * ncols_r):
        axes2[j].set_visible(False)

    fig2.suptitle("Best ripple metric vs |NIE| in moderate-|NIE| band (FP vs TP)",
                  fontsize=12, y=1.01)
    fig2.tight_layout()
    fig2.savefig(args.out_ripple, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out_ripple}")


if __name__ == "__main__":
    main()
