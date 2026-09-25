"""Fig — Suppression effect: orth vs local curvature in moderate-|NIE| band.

2D scatter colored TP/FP with multivariate logistic decision boundary.
Shows why orth beta increases when curvature is added: the boundary
cuts diagonally through the positively correlated cloud.

Usage:
    python figures/fig_suppression_scatter.py [--csv PATH] [--out PATH]
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
]


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


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_suppression_scatter.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    fig = plt.figure(figsize=(14, 9))
    gs_top = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.6)
    axes = [fig.add_subplot(gs_top[0, j*2:j*2+2]) for j in range(3)]
    axes += [fig.add_subplot(gs_top[1, 1:3]),
             fig.add_subplot(gs_top[1, 3:5])]

    for i, (bkey, blabel) in enumerate(BUCKETS):
        ax = axes[i]
        pool = build_moderate_band(df, bkey)
        curv_feat = "gap_frac" if bkey == "Gaussian_self" else "local_gap_frac"
        feats = ["orth_base", curv_feat]
        pool = zscore_within_combo(pool, feats)
        z_curv = f"z_{curv_feat}"
        clean = pool.dropna(subset=["z_orth_base", z_curv, "label"])
        clean = clean[np.isfinite(clean["z_orth_base"]) & np.isfinite(clean[z_curv])]

        X = clean[["z_orth_base", z_curv]].values
        y = clean["label"].values

        tp = clean[clean["label"] == 0]
        fp = clean[clean["label"] == 1]

        ax.scatter(tp["z_orth_base"], tp[z_curv], s=10, alpha=0.35,
                   color="#4C72B0", label=f"TP ({len(tp)})", zorder=2, rasterized=True)
        ax.scatter(fp["z_orth_base"], fp[z_curv], s=10, alpha=0.35,
                   color="#C44E52", label=f"FP ({len(fp)})", zorder=2, rasterized=True)

        # Fit multivariate LR and draw decision boundary
        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(X, y)
        auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

        # Also fit univariate orth for comparison
        lr_orth = LogisticRegression(max_iter=2000, C=10.0)
        lr_orth.fit(X[:, :1], y)
        auc_orth = roc_auc_score(y, lr_orth.predict_proba(X[:, :1])[:, 1])

        # Decision boundary: w0*x + w1*y + b = 0  =>  y = -(w0*x + b) / w1
        w0, w1 = lr.coef_[0]
        b = lr.intercept_[0]

        xlims = ax.get_xlim() if ax.get_xlim() != (0, 1) else (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
        # Set axis limits from data first
        pad = 0.3
        x_lo, x_hi = np.percentile(X[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(X[:, 1], [1, 99])
        x_lo, x_hi = x_lo - pad, x_hi + pad
        y_lo, y_hi = y_lo - pad, y_hi + pad

        if abs(w1) > 1e-6:
            x_line = np.linspace(x_lo, x_hi, 100)
            y_line = -(w0 * x_line + b) / w1
            mask = (y_line >= y_lo) & (y_line <= y_hi)
            ax.plot(x_line[mask], y_line[mask], "k-", lw=1.8, label=f"joint ({auc:.2f})", zorder=4)

        # Univariate orth boundary: vertical line at x = -b/w0
        w0u = lr_orth.coef_[0][0]
        bu = lr_orth.intercept_[0]
        x_thresh = -bu / w0u if abs(w0u) > 1e-6 else 0
        ax.axvline(x_thresh, color="grey", ls="--", lw=1.2,
                   label=f"orth only ({auc_orth:.2f})", zorder=3)

        # Univariate curv boundary: horizontal line at y = -b/w1
        lr_curv = LogisticRegression(max_iter=2000, C=10.0)
        lr_curv.fit(X[:, 1:], y)
        auc_curv = roc_auc_score(y, lr_curv.predict_proba(X[:, 1:])[:, 1])
        w1c = lr_curv.coef_[0][0]
        bc = lr_curv.intercept_[0]
        y_thresh = -bc / w1c if abs(w1c) > 1e-6 else 0
        ax.axhline(y_thresh, color="#2ca02c", ls=":", lw=1.4,
                   label=f"curv only ({auc_curv:.2f})", zorder=3)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(blabel, fontsize=10)
        ax.set_xlabel("z(orth)", fontsize=9)
        curv_label = "z(downstream curv)" if bkey == "Gaussian_self" else "z(local curv)"
        if i in (0, 3):
            ax.set_ylabel(curv_label, fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate betas
        ax.text(0.98, 0.02,
                f"$\\beta_{{orth}}$={lr.coef_[0][0]:+.2f}\n"
                f"$\\beta_{{curv}}$={lr.coef_[0][1]:+.2f}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Complentarity of orth + local curvature in FP-vs-TP (moderate |NIE|)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
