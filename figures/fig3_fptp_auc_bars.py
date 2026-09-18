"""Fig 3 — FP-vs-TP AUC bar chart: univariate AUC per instrument per strategy
on the moderate-|NIE| band.

Population: moderate-|NIE| band (two-stage filter):
  1. Per-combo median threshold on in-circuit |NIE| -> label TP/FP
  2. Lower-half split: keep only below-pool-median |NIE| rows

Label: FP=1 (off-circuit above threshold), TP=0 (in-circuit above threshold).
Univariate logistic regression AUC per instrument (z-scored within combo),
cluster-bootstrapped over combos.

Usage:
    python figures/fig3_fptp_auc_bars.py [--csv PATH] [--out PATH]
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
    """Two-stage filter: per-combo threshold then lower-half split."""
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def above(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr

    pool = pd.concat([in_circ, off_circ])
    pool = pool[pool.apply(above, axis=1)].copy()
    pool_median_nie = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median_nie].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)  # FP=1, TP=0
    return pool


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def univariate_auc_boot(pool, feat, B=200):
    """Univariate logistic regression AUC (z-scored within combo) with cluster-bootstrap CI."""
    z_feat = f"z_{feat}"
    clean = pool.dropna(subset=[z_feat, "label", "combo"]).copy()
    clean = clean[np.isfinite(clean[z_feat])]
    if len(clean) < 20 or clean["label"].nunique() < 2:
        return None, None, None

    X = clean[[z_feat]].values.astype(float)
    y = clean["label"].values
    mask = np.isfinite(X).ravel()
    X, y = X[mask], y[mask]

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

    combos = clean["combo"].values[mask]
    unique_combos = np.unique(combos)
    rng = np.random.RandomState(0)
    auc_boots = []
    for _ in range(B):
        idx_c = rng.choice(unique_combos, size=len(unique_combos), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
        yb, Xb = y[idx], X[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
            lrb.fit(Xb, yb)
            auc_boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
        except Exception:
            continue
    lo = np.percentile(auc_boots, 2.5) if auc_boots else auc
    hi = np.percentile(auc_boots, 97.5) if auc_boots else auc
    return auc, lo, hi


# ── config ──────────────────────────────────────────────────────────────────
BUCKETS = [
    ("STR_swap_within",     "STR swap\n(within)"),
    ("STR_swap_cross",      "STR swap\n(cross)"),
    ("STR_push_within",     "STR push\n(within)"),
    ("STR_push_cross",      "STR push\n(cross)"),
    ("Gaussian_push",       "Gaussian\npush"),
]

INSTRUMENTS = [
    ("$\\mathrm{orth}_b(\\delta)$",  "orth_base"),
    ("downstream curv.",             "gap_frac"),
    ("local curvature",              "local_gap_frac"),
    ("excess curv.",                 "excess_local_gap"),
    ("$\\mathrm{along}_s(\\delta)$", "along_nonbase_max"),
]

COLORS = [
    "#4C72B0",   # orth
    "#DD8452",   # downstream curv.
    "#55A868",   # local curvature
    "#8172B3",   # excess curv.
    "#C44E52",   # along_s
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig3_fptp_auc_bars.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    n_buckets = len(BUCKETS)
    n_inst = len(INSTRUMENTS)
    bar_width = 0.15
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    all_feats = [feat for _, feat in INSTRUMENTS]

    for i, (inst_label, feat) in enumerate(INSTRUMENTS):
        aucs, los, his = [], [], []
        for bkey, _ in BUCKETS:
            pool = build_moderate_band(df, bkey)
            pool = zscore_within_combo(pool, all_feats)
            auc, lo, hi = univariate_auc_boot(pool, feat)
            if auc is None:
                auc, lo, hi = 0.5, 0.5, 0.5
            aucs.append(auc)
            los.append(auc - lo)
            his.append(hi - auc)
        offset = (i - n_inst / 2 + 0.5) * bar_width
        ax.bar(x + offset, aucs, bar_width, yerr=[los, his],
               label=inst_label, color=COLORS[i],
               capsize=2, error_kw={"linewidth": 0.8}, edgecolor="white",
               linewidth=0.4, zorder=3)

    ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=9)
    ax.set_ylabel("Univariate AUC  (FP vs TP)", fontsize=10)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Which instrument separates False Positive from True Positive?\n"
                 "(moderate-$|$NIE$|$ band)",
                 fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
