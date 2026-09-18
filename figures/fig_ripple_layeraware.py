"""Fig — Layer-aware ripple analysis.

Perturbations propagate only downstream in feedforward transformers.
When in-circuit partners are upstream (unreachable), ripple_in is
structurally zero. This figure tests whether using ripple_in, ripple_off,
and ripple_frac as *separate features* in a multivariate model recovers
the lost signal — compared to any single metric alone.

Two bar-chart panels:
  1. FP-vs-TP (moderate |NIE| band)
  2. FN-vs-TN (below-threshold band)

Models compared per strategy:
  - ripple_in (univariate)
  - ripple_off (univariate)
  - ripple_in + ripple_off (bivariate)
  - ripple_in + ripple_off + ripple_frac + ripple_xint (full block)

Usage:
    python figures/fig_ripple_layeraware.py [--csv PATH]
"""
import argparse, re, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BUCKETS = [
    ("STR_swap_within",  "STR swap\n(within)"),
    ("STR_swap_cross",   "STR swap\n(cross)"),
    ("STR_push_within",  "STR push\n(within)"),
    ("STR_push_cross",   "STR push\n(cross)"),
    ("Gaussian_push",    "Gaussian\npush"),
]

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]

MODELS = [
    (["ripple_in_orth"],                                "ripple$_{\\mathrm{in}}$ only",  "#AAAAAA"),
    (["ripple_off_orth"],                               "ripple$_{\\mathrm{off}}$ only", "#4C72B0"),
    (["ripple_in_orth", "ripple_off_orth"],              "in + off",                     "#DD8452"),
    (RIPPLE_ALL,                                        "full ripple block",             "#8172B3"),
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


def build_fntn_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    def below(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] < thr
    pool = pd.concat([in_circ, off_circ])
    pool = pool[pool.apply(below, axis=1)].copy()
    pool["label"] = pool["in_base_circuit"].astype(float)
    return pool


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_auc_boot(pool, feats, B=200):
    z_feats = [f"z_{f}" for f in feats]
    clean = pool.dropna(subset=z_feats + ["label", "combo"]).copy()
    X = clean[z_feats].values.astype(float)
    y = clean["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    combos = clean["combo"].values[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None, None
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_ripple_layeraware.pdf")
    ap.add_argument("--txt", default="figures/fig_ripple_layeraware.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    n_buckets = len(BUCKETS)
    n_models = len(MODELS)
    bar_width = 0.18
    x = np.arange(n_buckets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    txt_lines = [f"{'panel':>8s}  {'strategy':>18s}  {'model':>25s}  {'AUC':>6s}  {'95% CI':>14s}",
                 "-" * 80]

    for panel_idx, (ax, panel_label, builder) in enumerate([
            (ax1, "FP-TP", build_moderate_band),
            (ax2, "FN-TN", build_fntn_population)]):

        for mi, (feats, label, color) in enumerate(MODELS):
            aucs, los, his = [], [], []
            for bkey, _ in BUCKETS:
                pool = builder(df, bkey)
                pool = zscore_within_combo(pool, RIPPLE_ALL)
                auc, lo, hi = fit_auc_boot(pool, feats)
                if auc is None:
                    auc, lo, hi = 0.5, 0.5, 0.5
                aucs.append(auc)
                los.append(auc - lo)
                his.append(hi - auc)
            offset = (mi - n_models / 2 + 0.5) * bar_width
            ax.bar(x + offset, aucs, bar_width, yerr=[los, his],
                   label=label, color=color,
                   capsize=2, error_kw={"linewidth": 0.8}, edgecolor="white",
                   linewidth=0.4, zorder=3)
            for bi in range(n_buckets):
                blbl = BUCKETS[bi][1].replace('\n', ' ')
                feat_str = "+".join(f.replace("ripple_","r_").replace("_orth","") for f in feats)
                txt_lines.append(
                    f"{panel_label:>8s}  {blbl:>18s}  {feat_str:>25s}  "
                    f"{aucs[bi]:>6.3f}  [{aucs[bi]-los[bi]:.3f}, {aucs[bi]+his[bi]:.3f}]")

        ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_ylim(0.35, 1.0)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{panel_label}", fontsize=11, pad=8)

    fig.suptitle("Ripple: univariate vs multivariate (recovering signal lost to layer structure)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
