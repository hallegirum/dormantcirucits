"""Fig R (FP vs TN) — Feature complementarity bars, off-circuit population.

Same 7 model combinations as the FP-vs-TP version, but on the FP-vs-TN contrast:
FP = off-circuit with |NIE| >= per-combo in-circuit median;
TN = off-circuit below threshold.

Usage:
    python figures/figR_fptn_complementarity_bars.py [--csv PATH]
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
    ("STR_swap_within",  "STR swap\n(within)"),
    ("STR_swap_cross",   "STR swap\n(cross)"),
    ("STR_push_within",  "STR push\n(within)"),
    ("STR_push_cross",   "STR push\n(cross)"),
    ("Gaussian_push",    "Gaussian\npush"),
]

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]

MODELS = [
    (["orth_base"],                          "orth",           "#7f7f7f"),
    (["gap_frac"],                             "curv",           "#2ca02c"),
    (RIPPLE_ALL,                             "ripple",         "#9467bd"),
    (["orth_base", "gap_frac"],               "orth+curv",      "#1f77b4"),
    (RIPPLE_ALL + ["gap_frac"],              "ripple+curv",    "#ff7f0e"),
    (["orth_base"] + RIPPLE_ALL,             "orth+ripple",    "#d62728"),
    (["orth_base", "gap_frac"] + RIPPLE_ALL,  "full",           "#17becf"),
]


def _strat_bucket(row):
    push = str(row["push_site"]) if pd.notna(row.get("push_site")) else ""
    strategy = str(row["strategy"]) if pd.notna(row.get("strategy")) else ""
    base, target = row.get("base_task"), row.get("target_task")
    within = pd.notna(target) and base == target
    if push == "input":
        return f"STR_swap_{'within' if within else 'cross'}"
    if push == "self":
        return f"{strategy}_self"
    if strategy == "STR":
        return f"STR_push_{'within' if within else 'cross'}"
    return f"{strategy}_push"


def build_fptn_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False].copy()
    thresholds = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    oc["label"] = oc.apply(is_fp, axis=1).astype(float)
    return oc


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_auc_boot(clean, y, combos, feats, B=300):
    z = [f"z_{f}" for f in feats]
    X = clean[z].values
    mask = np.all(np.isfinite(X), axis=1)
    X, yy, cc = X[mask], y[mask], combos[mask]
    if len(np.unique(yy)) < 2 or len(yy) < 20:
        return 0.5, 0.5, 0.5
    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, yy)
    auc = roc_auc_score(yy, lr.predict_proba(X)[:, 1])
    rng = np.random.RandomState(0)
    uc = np.unique(cc)
    boots = []
    for _ in range(B):
        idx = np.concatenate([np.where(cc == c)[0] for c in rng.choice(uc, len(uc), replace=True)])
        Xb, yb = X[idx], yy[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, C=10.0)
            lrb.fit(Xb, yb)
            boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
        except Exception:
            continue
    lo = np.percentile(boots, 2.5) if boots else auc
    hi = np.percentile(boots, 97.5) if boots else auc
    return auc, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/figR_fptn_complementarity_bars.pdf")
    ap.add_argument("--txt", default="figures/figR_fptn_complementarity_bars.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    n_buckets = len(BUCKETS)
    n_models = len(MODELS)
    bar_w = 0.11
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    txt_lines = [f"{'strategy':>18s}  {'model':>15s}  {'AUC':>6s}  {'95% CI':>14s}  {'n':>5s}",
                 "-" * 65]

    for mi, (feats, label, color) in enumerate(MODELS):
        aucs, los, his = [], [], []
        for bkey, _ in BUCKETS:
            pool = build_fptn_population(df, bkey)
            all_feats_needed = list(set(["orth_base", "gap_frac"] + RIPPLE_ALL))
            pool = zscore(pool, all_feats_needed)
            z_all = [f"z_{f}" for f in all_feats_needed]
            clean = pool.dropna(subset=z_all + ["label", "combo"])
            Xt = clean[z_all].values
            mask = np.all(np.isfinite(Xt), axis=1)
            clean = clean[mask]
            y = clean["label"].values
            combos = clean["combo"].values

            auc, lo, hi = fit_auc_boot(clean, y, combos, feats)
            aucs.append(auc)
            los.append(auc - lo)
            his.append(hi - auc)

            blbl = BUCKETS[[b[0] for b in BUCKETS].index(bkey)][1].replace("\n", " ")
            txt_lines.append(f"{blbl:>18s}  {label:>15s}  {auc:>6.3f}  "
                             f"[{auc - los[-1]:.3f}, {auc + his[-1]:.3f}]  {len(y):>5d}")

        offset = (mi - n_models / 2 + 0.5) * bar_w
        ax.bar(x + offset, aucs, bar_w, yerr=[los, his],
               label=label, color=color,
               capsize=2, error_kw={"linewidth": 0.7}, edgecolor="white",
               linewidth=0.4, zorder=3)

    ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8)
    ax.set_ylabel("AUC", fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7.5, ncol=4, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("FP-vs-TN: feature complementarity (off-circuit population)",
                 fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
