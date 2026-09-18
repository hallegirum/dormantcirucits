"""along_s vs local curvature scatter — FP vs TP.

2D scatter colored TP/FP with multivariate logistic decision boundary.
Uses local_gap_frac consistently across all strategies.

Usage:
    python figures/fig_along_vs_curv_scatter_fptp.py [--csv PATH]
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
CURV_FEAT = "local_gap_frac"


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base, target = row.get("base_task"), row.get("target_task")
    within = pd.notna(target) and base == target
    if push == "input": return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self": return f"{strategy}_self"
    if strategy == "STR": return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_moderate_band(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False]
    thr = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def above(row):
        t = thr.get(row["combo"])
        return t is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= t
    pool = pd.concat([ic, oc])
    pool = pool[pool.apply(above, axis=1)].copy()
    med = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= med].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_along_vs_curv_scatter_fptp.pdf")
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
        feats = ["along_nonbase_max", CURV_FEAT]
        pool = zscore(pool, feats)
        z_along = "z_along_nonbase_max"
        z_curv = f"z_{CURV_FEAT}"
        clean = pool.dropna(subset=[z_along, z_curv, "label"])
        clean = clean[np.isfinite(clean[z_along]) & np.isfinite(clean[z_curv])]

        X = clean[[z_along, z_curv]].values
        y = clean["label"].values

        tp = clean[clean["label"] == 0]
        fp = clean[clean["label"] == 1]

        ax.scatter(tp[z_along], tp[z_curv], s=10, alpha=0.35,
                   color="#4C72B0", label=f"TP ({len(tp)})", zorder=2, rasterized=True)
        ax.scatter(fp[z_along], fp[z_curv], s=10, alpha=0.35,
                   color="#C44E52", label=f"FP ({len(fp)})", zorder=2, rasterized=True)

        lr = LogisticRegression(max_iter=2000, C=10.0)
        lr.fit(X, y)
        auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

        lr_along = LogisticRegression(max_iter=2000, C=10.0)
        lr_along.fit(X[:, :1], y)
        auc_along = roc_auc_score(y, lr_along.predict_proba(X[:, :1])[:, 1])

        w0, w1 = lr.coef_[0]
        b = lr.intercept_[0]

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

        w0u = lr_along.coef_[0][0]
        bu = lr_along.intercept_[0]
        x_thresh = -bu / w0u if abs(w0u) > 1e-6 else 0
        ax.axvline(x_thresh, color="grey", ls="--", lw=1.2,
                   label=f"along only ({auc_along:.2f})", zorder=3)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(blabel, fontsize=10)
        ax.set_xlabel(r"z($\mathrm{along}_s$)", fontsize=9)
        if i in (0, 3):
            ax.set_ylabel("z(local curv)", fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(0.98, 0.02,
                f"$\\beta_{{along}}$={lr.coef_[0][0]:+.2f}\n"
                f"$\\beta_{{curv}}$={lr.coef_[0][1]:+.2f}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle(r"$\mathrm{along}_s$ vs local curvature — FP vs TP (moderate |NIE|)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
