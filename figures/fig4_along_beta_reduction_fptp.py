"""Fig 4 — along_s beta reduction: M3L vs FP-vs-TP across strategies.

Two-panel dot+line plot. X = strategy bucket. Three lines:
  - along_s (base model) — solid
  - along_s (+best curvature) — dashed, different color to show reduction
  - curvature beta — the curvature metric's own beta when added

Curvature metric per panel:
  - FP-vs-TP: local curvature for all strategies
  - FP-vs-TN: downstream curvature for swap + Gaussian push,
               local curvature for STR push

No CIs — clean comparison of point estimates across strategies.

Usage:
    python figures/fig4_along_beta_reduction_fptp.py [--csv PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def build_m3l_population(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    in_circ = sub[sub["in_base_circuit"] == True]
    off_circ = sub[sub["in_base_circuit"] == False].copy()
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr

    off_circ["label"] = off_circ.apply(is_fp, axis=1).astype(float)
    return off_circ


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
    pool_median_nie = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median_nie].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def zscore_within_combo(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def fit_model(df, feats):
    z_feats = [f"z_{f}" for f in feats]
    X = df[z_feats].values.astype(float)
    y = df["label"].values
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0,
                            class_weight="balanced")
    lr.fit(X, y)
    return {f: lr.coef_[0][i] for i, f in enumerate(feats)}


BUCKETS = ["STR_swap_within", "STR_swap_cross",
           "STR_push_within", "STR_push_cross", "Gaussian_push"]
BUCKET_LABELS = {
    "STR_swap_within": "swap\n(within)",
    "STR_swap_cross": "swap\n(cross)",
    "STR_push_within": "push\n(within)",
    "STR_push_cross": "push\n(cross)",
    "Gaussian_push": "Gaussian\npush",
}

# best curvature metric per bucket for M3L (FP vs TN)
M3L_CURV = {
    "STR_swap_within": "gap_frac",
    "STR_swap_cross":  "gap_frac",
    "STR_push_within": "local_gap_frac",
    "STR_push_cross":  "local_gap_frac",
    "Gaussian_push":   "gap_frac",
}

# FP-vs-TP always uses local curvature
FPTP_CURV = {bk: "local_gap_frac" for bk in BUCKETS}


def compute_betas(df, builder, curv_map):
    results = {bk: {} for bk in BUCKETS}
    for bucket in BUCKETS:
        pool = builder(df, bucket)
        is_cross = "cross" in bucket
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")

        curv_feat = curv_map[bucket]
        required = base_feats + ["local_gap_frac", "gap_frac", "excess_local_gap"]
        pool_clean = pool.dropna(subset=required + ["label", "combo"])
        for f in required:
            pool_clean = pool_clean[np.isfinite(pool_clean[f])]
        pool_clean = zscore_within_combo(pool_clean, required)

        betas_base = fit_model(pool_clean, base_feats)
        results[bucket]["along_s_base"] = betas_base["along_nonbase_max"]

        betas_curv = fit_model(pool_clean, base_feats + [curv_feat])
        results[bucket]["along_s_plus_curv"] = betas_curv["along_nonbase_max"]
        results[bucket]["curv_beta"] = betas_curv[curv_feat]

    return results


def plot_panel(ax, results, title, curv_map):
    x = np.arange(len(BUCKETS))

    along_base = [results[bk]["along_s_base"] for bk in BUCKETS]
    along_curv = [results[bk]["along_s_plus_curv"] for bk in BUCKETS]
    curv_beta  = [results[bk]["curv_beta"] for bk in BUCKETS]

    ax.plot(x, along_base, color="#C44E52", marker="o", ls="-", lw=2.5,
            markersize=9, zorder=4, label="$\\mathrm{along}_s$ (base)")
    ax.plot(x, along_curv, color="#9467BD", marker="s", ls="--", lw=2.0,
            markersize=8, zorder=4, label="$\\mathrm{along}_s$ (+curv.)")
    ax.plot(x, curv_beta, color="#55A868", marker="D", ls="-", lw=2.0,
            markersize=7, zorder=3, label="curvature $\\beta$")

    # shade the reduction
    ax.fill_between(x, along_base, along_curv, color="#C44E52", alpha=0.12, zorder=1)


    ax.axhline(0, color="grey", ls="-", lw=0.5, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[bk] for bk in BUCKETS], fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel("$\\beta$ (standardised)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out", default="figures/fig4_along_beta_reduction_fptp.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    m3l_results = compute_betas(df, build_m3l_population, M3L_CURV)
    fptp_results = compute_betas(df, build_moderate_band, FPTP_CURV)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    plot_panel(axes[0], m3l_results, "FP vs TN (M3L)", M3L_CURV)
    plot_panel(axes[1], fptp_results, "FP vs TP (moderate $|$NIE$|$ band)", FPTP_CURV)

    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.suptitle("$\\mathrm{along}_s(\\delta)$ beta: base model vs after adding curvature",
                 fontsize=12, y=1.04)
    fig.text(0.5, 0.99,
             "Dashed purple and green lines are from the same joint model "
             "(base + curvature).",
             ha="center", fontsize=9, color="0.4")
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
