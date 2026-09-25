"""Bar chart: local vs downstream vs excess curvature betas (FP vs TP).

Each curvature variant is added to the base model (orth_b + along_s) one at
a time.  Bars show standardised beta with 95% cluster-bootstrap CI.

Usage:
    python figures/fig_fptp_curv_comparison_bars.py [--csv PATH] [--out PATH]
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
    ("STR_swap_within",  "swap\n(within)"),
    ("STR_swap_cross",   "swap\n(cross)"),
    ("STR_push_within",  "push\n(within)"),
    ("STR_push_cross",   "push\n(cross)"),
    ("Gaussian_push",    "Gaussian\npush"),
]

CURV_FEATS = [
    ("local_gap_frac",     "local curv.",     "#2166ac"),
    ("gap_frac",           "downstream curv.", "#92c5de"),
    ("excess_local_gap",   "excess curv.",     "#f4a582"),
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


def build_fptp_population(df, bucket):
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


def fit_and_extract_curv_beta(pool, base_feats, curv_feat, B=500):
    feats = list(base_feats) + [curv_feat]
    z_feats = [f"z_{f}" for f in feats]
    X = pool[z_feats].values.astype(float)
    y = pool["label"].values

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0,
                            class_weight="balanced")
    lr.fit(X, y)
    curv_idx = len(base_feats)
    beta = lr.coef_[0][curv_idx]

    combos = pool["combo"].values
    unique_combos = np.unique(combos)
    rng = np.random.RandomState(42)
    boots = []
    for _ in range(B):
        idx_c = rng.choice(unique_combos, size=len(unique_combos), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
        yb, Xb = y[idx], X[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0,
                                     class_weight="balanced")
            lrb.fit(Xb, yb)
            boots.append(lrb.coef_[0][curv_idx])
        except Exception:
            continue

    boots = np.array(boots)
    lo = np.percentile(boots, 2.5) if len(boots) else beta
    hi = np.percentile(boots, 97.5) if len(boots) else beta
    sig = (lo > 0) or (hi < 0)
    return beta, lo, hi, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out",
                    default="figures/fig_fptp_curv_comparison_bars.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    all_feats = ["orth_base", "along_nonbase_max", "along_target",
                 "local_gap_frac", "gap_frac", "excess_local_gap"]

    results = {}
    for bucket_key, bucket_label in BUCKETS:
        pool = build_fptp_population(df, bucket_key)
        is_cross = "cross" in bucket_key
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")

        required = base_feats + [c[0] for c in CURV_FEATS]
        pool = pool.dropna(subset=required + ["label", "combo"])
        for f in required:
            pool = pool[np.isfinite(pool[f])]
        pool = zscore_within_combo(pool, required)

        bucket_res = []
        for curv_feat, curv_label, color in CURV_FEATS:
            beta, lo, hi, sig = fit_and_extract_curv_beta(
                pool, base_feats, curv_feat, B=500)
            bucket_res.append({
                "curv_label": curv_label,
                "color": color,
                "beta": beta,
                "lo": lo,
                "hi": hi,
                "sig": sig,
            })
            print(f"  {bucket_label.replace(chr(10),' '):20s}  "
                  f"{curv_label:20s}  beta={beta:+.2f}  "
                  f"CI=[{lo:.2f}, {hi:.2f}]  {'*' if sig else 'ns'}")
        results[bucket_key] = bucket_res

      # ── plot ──
    fig, ax = plt.subplots(figsize=(8, 4.0))

    n_strats = len(BUCKETS)
    n_curv = len(CURV_FEATS)
    bar_width = 0.22
    x_base = np.arange(n_strats)

    # Track CI extrema so the axis can accommodate all error bars + stars.
    all_ci_lows = []
    all_ci_highs = []

    for j, (curv_feat, curv_label, color) in enumerate(CURV_FEATS):
        offsets = (j - 1) * bar_width
        betas, errs_lo, errs_hi, sigs = [], [], [], []

        for i, (bucket_key, bucket_label) in enumerate(BUCKETS):
            r = results[bucket_key][j]
            betas.append(r["beta"])
            errs_lo.append(r["beta"] - r["lo"])
            errs_hi.append(r["hi"] - r["beta"])
            sigs.append(r["sig"])

            all_ci_lows.append(r["lo"])
            all_ci_highs.append(r["hi"])

        betas = np.array(betas)
        errs_lo = np.array(errs_lo)
        errs_hi = np.array(errs_hi)
        errs = np.array([errs_lo, errs_hi])

        bars = ax.bar(
            x_base + offsets,
            betas,
            bar_width,
            yerr=errs,
            capsize=3,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=curv_label,
            zorder=3,
            error_kw=dict(
                lw=1.0,
                capthick=1.0,
                color="#333333",
                zorder=4,
            ),
        )

        for i, (b, s) in enumerate(zip(betas, sigs)):
            if s:
                # Position star beyond the CI, rather than beyond the bar.
                if b >= 0:
                    y_pos = b + errs_hi[i] + 0.06
                    va = "bottom"
                else:
                    y_pos = b - errs_lo[i] - 0.06
                    va = "top"

                ax.text(
                    x_base[i] + offsets,
                    y_pos,
                    "*",
                    ha="center",
                    va=va,
                    fontsize=12,
                    fontweight="bold",
                    color="#333333",
                    zorder=5,
                    clip_on=False,
                )

    ax.set_xticks(x_base)
    ax.set_xticklabels([bl for _, bl in BUCKETS], fontsize=9)
    ax.set_ylabel("Standardised β (curvature)", fontsize=10)

    ax.axhline(
        0,
        color="#999999",
        lw=0.8,
        ls="--",
        zorder=1,
    )

    # Legend in top-left corner.
    ax.legend(
        fontsize=8.5,
        loc="upper left",
        framealpha=0.9,
    )

    # Dynamically set limits so CIs and significance stars are not clipped.
    ci_min = min(all_ci_lows)
    ci_max = max(all_ci_highs)
    ci_range = max(ci_max - ci_min, 1.0)

    bottom_padding = 0.10 * ci_range
    top_padding = 0.18 * ci_range  # extra room for stars

    ax.set_ylim(
        min(-0.8, ci_min - bottom_padding),
        max(2.4, ci_max + top_padding),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, zorder=0)

    fig.tight_layout()
    fig.savefig(
        args.out,
        dpi=300,
        bbox_inches="tight",
    )
    print(f"\nSaved {args.out}")

if __name__ == "__main__":
    main()
