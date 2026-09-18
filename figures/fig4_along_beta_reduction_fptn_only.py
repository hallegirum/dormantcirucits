"""along_s beta reduction — FP vs TN only.

Shows along_s beta in base model vs after adding downstream curvature.
Uses gap_frac consistently across all strategies.

Usage:
    python figures/fig4_along_beta_reduction_fptn_only.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BUCKETS = ["STR_swap_within", "STR_swap_cross",
           "STR_push_within", "STR_push_cross", "Gaussian_push"]
BUCKET_LABELS = {
    "STR_swap_within": "swap\n(within)", "STR_swap_cross": "swap\n(cross)",
    "STR_push_within": "push\n(within)", "STR_push_cross": "push\n(cross)",
    "Gaussian_push": "Gaussian\npush",
}
CURV_FEAT = "gap_frac"


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


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_model(df, feats):
    z = [f"z_{f}" for f in feats]
    X = df[z].values.astype(float)
    y = df["label"].values
    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, y)
    return {f: lr.coef_[0][i] for i, f in enumerate(feats)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig4_along_beta_reduction_fptn_only.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    x = np.arange(len(BUCKETS))
    along_base, along_curv, curv_beta = [], [], []

    for bk in BUCKETS:
        pool = build_fptn_population(df, bk)
        is_cross = "cross" in bk
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")
        required = base_feats + [CURV_FEAT]
        pool = pool.dropna(subset=required + ["label", "combo"])
        pool = zscore(pool, required)
        for f in required:
            pool = pool[np.isfinite(pool[f])]

        betas_b = fit_model(pool, base_feats)
        along_base.append(betas_b["along_nonbase_max"])

        betas_c = fit_model(pool, base_feats + [CURV_FEAT])
        along_curv.append(betas_c["along_nonbase_max"])
        curv_beta.append(betas_c[CURV_FEAT])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, along_base, color="#C44E52", marker="o", ls="-", lw=2.5,
            markersize=9, zorder=4, label=r"$\mathrm{along}_s$ (base)")
    ax.plot(x, along_curv, color="#9467BD", marker="s", ls="--", lw=2.0,
            markersize=8, zorder=4, label=r"$\mathrm{along}_s$ (+dwnstrm curv)")
    ax.plot(x, curv_beta, color="#55A868", marker="D", ls="-", lw=2.0,
            markersize=7, zorder=3, label=r"downstream curvature $\beta$")
    ax.fill_between(x, along_base, along_curv, color="#C44E52", alpha=0.12, zorder=1)

    ax.axhline(0, color="grey", ls="-", lw=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[bk] for bk in BUCKETS], fontsize=9)
    ax.set_ylabel(r"$\beta$ (standardised)", fontsize=10)
    ax.set_title(r"$\mathrm{along}_s$ suppression by downstream curvature — FP vs TN", fontsize=11, pad=8)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
