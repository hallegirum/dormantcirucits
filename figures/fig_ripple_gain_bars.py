"""Fig — Base model vs base + ripple: AUC bars.

Three bars per strategy:
  1. base = curv (best) + orth + along  (grey)
  2. base + best ripple individual      (orange, annotated)
  3. base + in_prop                     (purple)

Two panels: FP-vs-TN (left), FP-vs-TP (right).
Best curvature selected per strategy per decomposition.
Best ripple selected per strategy per decomposition.

Usage:
    python figures/fig_ripple_gain_bars.py [--csv PATH] [--out PATH]
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

RIPPLE_FEATS = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
RIPPLE_INDIVIDUAL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
CURV_CANDIDATES = ["gap_frac", "local_gap_frac", "excess_local_gap"]

METRIC_SHORT = {
    "ripple_in_orth": "in",
    "ripple_off_orth": "off",
    "ripple_frac": "in_prop",
    "ripple_xint_in_orth": "xint",
}

CURV_SHORT = {
    "gap_frac": "dwnstrm",
    "local_gap_frac": "local",
    "excess_local_gap": "excess",
}

BUCKETS = [
    ("STR_swap_within",  "swap\n(within)"),
    ("STR_swap_cross",   "swap\n(cross)"),
    ("STR_push_within",  "STR push\n(within)"),
    ("STR_push_cross",   "STR push\n(cross)"),
    ("Gaussian_push",    "Gaussian\npush"),
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


def build_fptn_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False].copy()
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    off_circ["label"] = off_circ.apply(is_fp, axis=1).astype(float)
    return off_circ


def build_fptp_moderate(df, bucket):
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


def fit_auc_boot(df, feats, B=300):
    z_feats = [f"z_{f}" for f in feats]
    sub = df.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    combos = sub["combo"].values[mask]
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


def auc_univar(clean, feat):
    z = f"z_{feat}"
    sub = clean.dropna(subset=[z, "label"]).copy()
    X = sub[[z]].values.astype(float)
    y = sub["label"].values.astype(float)
    m = np.isfinite(X).ravel()
    X, y = X[m], y[m]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, y)
    return roc_auc_score(y, lr.predict_proba(X)[:, 1])


def pick_best_curv(clean):
    best_auc, best_feat = -1, CURV_CANDIDATES[0]
    for c in CURV_CANDIDATES:
        a = auc_univar(clean, c)
        if a is not None and a > best_auc:
            best_auc, best_feat = a, c
    return best_feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_ripple_gain_bars.pdf")
    ap.add_argument("--txt", default="figures/fig_ripple_gain_bars.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    n_buckets = len(BUCKETS)
    n_models = 3
    bar_width = 0.22
    x = np.arange(n_buckets)

    colors = ["#AAAAAA", "#DD8452", "#8172B3"]
    model_labels = ["base (curv+orth+along)", "+ best ripple", "+ in_prop"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    txt_lines = [f"{'strategy':>20s}  {'decomp':>10s}  {'model':>24s}  "
                 f"{'AUC':>6s}  {'95% CI':>16s}  {'note':>12s}",
                 "-" * 100]

    for panel_idx, (decomp_name, build_fn, panel_title) in enumerate([
        ("FP vs TN", build_fptn_population, "FP vs TN"),
        ("FP vs TP", build_fptp_moderate, "FP vs TP"),
    ]):
        ax = axes[panel_idx]

        for bi, (bkey, blabel_nl) in enumerate(BUCKETS):
            blabel = blabel_nl.replace('\n', ' ')
            is_cross = "cross" in bkey

            all_feats = (["orth_base", "along_nonbase_max"] + CURV_CANDIDATES
                         + RIPPLE_FEATS)
            if is_cross:
                all_feats.append("along_target")

            pool = build_fn(df, bkey)
            clean = zscore_within_combo(pool, all_feats)

            best_curv = pick_best_curv(clean)

            base_feats = [best_curv, "orth_base", "along_nonbase_max"]
            if is_cross:
                base_feats.append("along_target")

            # find best individual ripple
            best_rip_auc, best_rip_feat = -1, RIPPLE_FEATS[0]
            for rf in RIPPLE_INDIVIDUAL:
                a = auc_univar(clean, rf)
                if a is not None and a > best_rip_auc:
                    best_rip_auc, best_rip_feat = a, rf

            models = [
                (base_feats, ""),
                (base_feats + [best_rip_feat], METRIC_SHORT[best_rip_feat]),
                (base_feats + ["ripple_frac"], ""),
            ]

            for mi, (feats, note) in enumerate(models):
                auc, lo, hi = fit_auc_boot(clean, feats)
                if auc is None:
                    auc, lo, hi = 0.5, 0.5, 0.5

                offset = (mi - n_models / 2 + 0.5) * bar_width
                label = model_labels[mi] if bi == 0 else None
                ax.bar(x[bi] + offset, auc, bar_width,
                       yerr=[[auc - lo], [hi - auc]],
                       color=colors[mi], label=label,
                       capsize=2.5, error_kw={"linewidth": 0.8},
                       edgecolor="white", linewidth=0.4, zorder=3)

                if note:
                    ax.text(x[bi] + offset, hi + 0.005, note,
                            ha="center", va="bottom", fontsize=8.5,
                            fontweight="bold", color="black")

                curv_note = CURV_SHORT[best_curv] if mi == 0 else ""
                txt_lines.append(
                    f"{blabel:>20s}  {decomp_name:>10s}  {model_labels[mi]:>24s}  "
                    f"{auc:>6.3f}  [{lo:.3f}, {hi:.3f}]  "
                    f"{note or curv_note:>12s}")

        ax.axhline(0.5, color="grey", ls="--", lw=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8)
        ax.set_title(panel_title, fontsize=12, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx == 0:
            ax.set_ylabel("AUC", fontsize=10)

    fig.suptitle("Base model (curv + orth + along) vs adding ripple",
                 fontsize=13, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,

            loc='upper right',

            bbox_to_anchor=(0.98, 0.98),

            fontsize=9,

            framealpha=0.9,

            ncol=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for legend
    fig.savefig(args.out, dpi=200, bbox_inches="tight")

    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
