"""Standalone discriminative power — FP vs TP (moderate |NIE| band).

Bars: local curvature, orth, along, best individual ripple (labeled), ripple block.
Ripple_frac only shown if it's the best individual.

Usage:
    python figures/fig_standalone_auc_fptp.py [--csv PATH]
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
CURV_FEAT = "local_gap_frac"

RIPPLE_SHORT = {
    "ripple_in_orth": "in", "ripple_off_orth": "off",
    "ripple_frac": "frac", "ripple_xint_in_orth": "xint",
}

BUCKETS = [
    ("STR_swap_within",  "swap\n(within)"),
    ("STR_swap_cross",   "swap\n(cross)"),
    ("STR_push_within",  "push\n(within)"),
    ("STR_push_cross",   "push\n(cross)"),
    ("Gaussian_push",    "Gaussian\npush"),
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


def build_fptp_moderate(df, bucket):
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


def fit_auc_boot(clean, feats, B=300):
    z = [f"z_{f}" for f in feats]
    sub = clean.dropna(subset=z + ["label"]).copy()
    X = sub[z].values.astype(float)
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
    boots = []
    for _ in range(B):
        idx = np.concatenate([np.where(combos == c)[0] for c in rng.choice(uc, len(uc), replace=True)])
        yb = y[idx]
        if len(np.unique(yb)) < 2: continue
        try:
            lrb = LogisticRegression(max_iter=2000, C=10.0)
            lrb.fit(X[idx], yb)
            boots.append(roc_auc_score(yb, lrb.predict_proba(X[idx])[:, 1]))
        except: continue
    return auc, np.percentile(boots, 2.5) if boots else auc, np.percentile(boots, 97.5) if boots else auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_standalone_auc_fptp.pdf")
    ap.add_argument("--txt", default="figures/fig_standalone_auc_fptp.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    x = np.arange(len(BUCKETS))
    txt_lines = []

    # Dynamic bar count: curv, orth, along, best_ripple, ripple_block = 5
    bar_colors = {
        "local curv": "#C44E52",
        "orth":       "#4C72B0",
        "along":      "#55A868",
        "best ripple":"#DD8452",
        "ripple block":"#8172B3",
    }
    bar_names = list(bar_colors.keys())
    n_bars = len(bar_names)
    bar_w = 0.14

    fig, ax = plt.subplots(figsize=(12, 5))

    for bi, (bkey, blabel) in enumerate(BUCKETS):
        is_cross = "cross" in bkey
        pool = build_fptp_moderate(df, bkey)
        all_feats = ["orth_base", "along_nonbase_max", CURV_FEAT] + RIPPLE_FEATS
        if is_cross:
            all_feats.append("along_target")
        clean = zscore(pool, all_feats)

        # Find best individual ripple
        best_rip_auc, best_rip_feat = -1, RIPPLE_FEATS[0]
        for rf in RIPPLE_FEATS:
            a, _, _ = fit_auc_boot(clean, [rf])
            if a is not None and a > best_rip_auc:
                best_rip_auc, best_rip_feat = a, rf

        results = []
        # 1. local curv
        a, lo, hi = fit_auc_boot(clean, [CURV_FEAT])
        results.append((a, lo, hi, ""))
        # 2. orth
        a, lo, hi = fit_auc_boot(clean, ["orth_base"])
        results.append((a, lo, hi, ""))
        # 3. along
        along_f = ["along_nonbase_max"] + (["along_target"] if is_cross else [])
        a, lo, hi = fit_auc_boot(clean, along_f)
        results.append((a, lo, hi, ""))
        # 4. best ripple individual
        a, lo, hi = fit_auc_boot(clean, [best_rip_feat])
        results.append((a, lo, hi, RIPPLE_SHORT[best_rip_feat]))
        # 5. ripple block
        a, lo, hi = fit_auc_boot(clean, RIPPLE_FEATS)
        results.append((a, lo, hi, ""))

        for mi, (auc, lo, hi, note) in enumerate(results):
            if auc is None: continue
            offset = (mi - n_bars / 2 + 0.5) * bar_w
            color = list(bar_colors.values())[mi]
            label = bar_names[mi] if bi == 0 else None
            ax.bar(x[bi] + offset, auc, bar_w,
                   yerr=[[auc - lo], [hi - auc]],
                   color=color, label=label,
                   capsize=2, error_kw={"linewidth": 0.7},
                   edgecolor="white", linewidth=0.3, zorder=3)
            if note:
                ax.text(x[bi] + offset, hi + 0.008, note,
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color="black")

            txt_lines.append(f"{blabel.replace(chr(10),' '):>18s}  {bar_names[mi]:>14s}  "
                             f"{auc:.3f}  [{lo:.3f}, {hi:.3f}]  {note}")

    ax.axhline(0.5, color="grey", ls="--", lw=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8)
    ax.set_ylabel("Univariate AUC", fontsize=10)
    ax.set_ylim(0.40, 1.02)
    ax.set_title("Standalone discriminative power — FP vs TP (moderate |NIE| band, matched sample)",
                 fontsize=11, pad=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")
    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
