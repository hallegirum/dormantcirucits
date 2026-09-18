"""Fig — Standalone discriminative power of each feature family.

1x2 panels (FP-vs-TN, FP-vs-TP). Five grouped bars per strategy:
curv (best), orth, along, best ripple individual, ripple in_prop.
Each bar = univariate AUC with cluster-bootstrap CI.

Best curvature and best ripple selected per strategy per decomposition.

Usage:
    python figures/fig_standalone_auc_bars.py [--csv PATH] [--out PATH]
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
    ("STR_swap_within",  "swap (within)"),
    ("STR_swap_cross",   "swap (cross)"),
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


def fit_auc_boot(clean, feats, B=300):
    z_feats = [f"z_{f}" for f in feats]
    sub = clean.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    combos = sub["combo"].values[mask]

    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None, None

    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

    uc = np.unique(combos)
    rng = np.random.RandomState(0)
    auc_boots = []
    for _ in range(B):
        ic = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in ic])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, C=10.0)
            lrb.fit(X[idx], yb)
            auc_boots.append(roc_auc_score(yb, lrb.predict_proba(X[idx])[:, 1]))
        except Exception:
            continue
    lo = np.percentile(auc_boots, 2.5) if auc_boots else auc
    hi = np.percentile(auc_boots, 97.5) if auc_boots else auc
    return auc, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_standalone_auc_bars.pdf")
    ap.add_argument("--txt", default="figures/fig_standalone_auc_bars.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    decomps = [
        ("FP vs TN", build_fptn_population),
        ("FP vs TP", build_fptp_moderate),
    ]

    bar_defs = [
        ("curv (best)", "#C44E52"),
        ("orth",        "#4C72B0"),
        ("along",       "#55A868"),
        ("best ripple", "#DD8452"),
        ("in_prop",     "#8172B3"),
    ]

    n_bars = len(bar_defs)
    bar_width = 0.14
    x = np.arange(len(BUCKETS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    txt_lines = [f"{'strategy':>20s}  {'decomp':>10s}  {'feature':>14s}  "
                 f"{'AUC':>6s}  {'95% CI':>16s}  {'note':>12s}",
                 "-" * 90]

    for pi, (decomp_name, build_fn) in enumerate(decomps):
        ax = axes[pi]

        for bi, (bkey, blabel) in enumerate(BUCKETS):
            is_cross = "cross" in bkey
            all_feats = (["orth_base", "along_nonbase_max"] + CURV_CANDIDATES
                         + RIPPLE_FEATS)
            if is_cross:
                all_feats.append("along_target")

            pool = build_fn(df, bkey)
            clean = zscore_within_combo(pool, all_feats)

            # 1. curv (best)
            best_curv_auc, best_curv_feat = -1, CURV_CANDIDATES[0]
            for c in CURV_CANDIDATES:
                a, _, _ = fit_auc_boot(clean, [c])
                if a is not None and a > best_curv_auc:
                    best_curv_auc, best_curv_feat = a, c
            curv_auc, curv_lo, curv_hi = fit_auc_boot(clean, [best_curv_feat])

            # 2. orth
            orth_auc, orth_lo, orth_hi = fit_auc_boot(clean, ["orth_base"])

            # 3. along
            along_feats = ["along_nonbase_max"] + (["along_target"] if is_cross else [])
            along_auc, along_lo, along_hi = fit_auc_boot(clean, along_feats)

            # 4. best ripple individual
            best_rip_auc, best_rip_feat = -1, RIPPLE_FEATS[0]
            best_rip_lo, best_rip_hi = 0.5, 0.5
            for rf in RIPPLE_FEATS:
                a, lo, hi = fit_auc_boot(clean, [rf])
                if a is not None and a > best_rip_auc:
                    best_rip_auc, best_rip_feat = a, rf
                    best_rip_lo, best_rip_hi = lo, hi

            # 5. in_prop (ripple_frac)
            frac_auc, frac_lo, frac_hi = fit_auc_boot(clean, ["ripple_frac"])

            results = [
                (curv_auc, curv_lo, curv_hi, CURV_SHORT[best_curv_feat]),
                (orth_auc, orth_lo, orth_hi, ""),
                (along_auc, along_lo, along_hi, ""),
                (best_rip_auc, best_rip_lo, best_rip_hi, METRIC_SHORT[best_rip_feat]),
                (frac_auc, frac_lo, frac_hi, ""),
            ]

            for mi, (auc, lo, hi, note) in enumerate(results):
                if auc is None:
                    continue
                offset = (mi - n_bars / 2 + 0.5) * bar_width
                color = bar_defs[mi][1]
                label = bar_defs[mi][0] if bi == 0 else None
                ax.bar(x[bi] + offset, auc, bar_width,
                       yerr=[[auc - lo], [hi - auc]],
                       color=color, label=label,
                       capsize=2, error_kw={"linewidth": 0.7},
                       edgecolor="white", linewidth=0.3, zorder=3)

                if note:
                    ax.text(x[bi] + offset, hi + 0.008, note,
                            ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="black")

                txt_lines.append(
                    f"{blabel:>20s}  {decomp_name:>10s}  {bar_defs[mi][0]:>14s}  "
                    f"{auc:>6.3f}  [{lo:.3f}, {hi:.3f}]  {note:>12s}")

        ax.axhline(0.5, color="grey", ls="--", lw=0.7, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8)
        ax.set_title(decomp_name, fontsize=12, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("Univariate AUC", fontsize=10)
        ax.set_ylim(0.40, 1.02)

    fig.suptitle("Standalone discriminative power by feature family",
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
