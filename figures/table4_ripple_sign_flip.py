"""Table 4 — Ripple directional decomposition: where does disruption land?

Shows univariate betas and AUC for all 5 ripple signals (in, off, frac,
xint, block) side by side in FP-vs-TP and FP-vs-TN. The key asymmetry:
ripple_in is consistently negative in FP-vs-TP (TPs propagate more
in-circuit ripple) but positive in FP-vs-TN. ripple_off is largely
insignificant in FP-vs-TP — the signal is one-sided.

Usage:
    python figures/table4_ripple_sign_flip.py [--csv PATH]
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

BUCKETS = [
    ("STR_swap_within",  "swap (within)"),
    ("STR_swap_cross",   "swap (cross)"),
    ("STR_push_within",  "STR push (within)"),
    ("STR_push_cross",   "STR push (cross)"),
    ("Gaussian_push",    "Gaussian push"),
]

METRICS = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth", "block"]
METRIC_SHORT = {
    "ripple_in_orth": "in-circuit",
    "ripple_off_orth": "off-circuit",
    "ripple_frac": "in_prop",
    "ripple_xint_in_orth": "xint",
    "block": "block (all 4)",
}
BAR_COLORS = {
    "ripple_in_orth": "#4C72B0",
    "ripple_off_orth": "#C44E52",
    "ripple_frac": "#55A868",
    "ripple_xint_in_orth": "#DD8452",
    "block": "#8172B3",
}


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


def fit_univar(df, feat_or_list, B=200):
    feats = feat_or_list if isinstance(feat_or_list, list) else [feat_or_list]
    z_feats = [f"z_{f}" for f in feats]
    sub = df.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    beta = lr.coef_[0][0] if len(feats) == 1 else None

    combos = sub["combo"].values[mask]
    unique_combos = np.unique(combos)
    rng = np.random.RandomState(0)
    beta_boots, auc_boots = [], []
    for _ in range(B):
        idx_c = rng.choice(unique_combos, size=len(unique_combos), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in idx_c])
        yb, Xb = y[idx], X[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lrb = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
            lrb.fit(Xb, yb)
            if beta is not None:
                beta_boots.append(lrb.coef_[0][0])
            auc_boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
        except Exception:
            continue

    result = {"auc": auc,
              "auc_lo": np.percentile(auc_boots, 2.5) if auc_boots else auc,
              "auc_hi": np.percentile(auc_boots, 97.5) if auc_boots else auc}
    if beta is not None:
        result["beta"] = beta
        result["beta_lo"] = np.percentile(beta_boots, 2.5) if beta_boots else beta
        result["beta_hi"] = np.percentile(beta_boots, 97.5) if beta_boots else beta
    return result


def fmt_beta(est, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"${est:+.2f}${sig}"


def fmt_beta_txt(est, lo, hi):
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{est:+.2f}{sig}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/table4_ripple_sign_flip.tex")
    ap.add_argument("--txt", default="figures/table4_ripple_sign_flip.txt")
    ap.add_argument("--fig", default="figures/fig_ripple_direction.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    all_results = {}

    for bkey, blabel in BUCKETS:
        for decomp_name, build_fn in [("FP vs TP", build_fptp_moderate),
                                       ("FP vs TN", build_fptn_population)]:
            pool = build_fn(df, bkey)
            clean = pool.dropna(subset=RIPPLE_FEATS + ["label", "combo"])
            for f in RIPPLE_FEATS:
                clean = clean[np.isfinite(clean[f])]
            clean = zscore_within_combo(clean, RIPPLE_FEATS)

            n_pos = int(clean["label"].sum())
            n_neg = len(clean) - n_pos

            univar = {}
            for rf in METRICS:
                feat_input = RIPPLE_FEATS if rf == "block" else rf
                res = fit_univar(clean, feat_input)
                univar[rf] = res

            key = (bkey, decomp_name)
            all_results[key] = {"n": len(clean), "n_pos": n_pos, "n_neg": n_neg,
                                "univar": univar}

            print(f"\n{blabel} — {decomp_name}  (n={len(clean)}, +={n_pos}, -={n_neg})")
            for rf in METRICS:
                u = univar[rf]
                if u:
                    beta_str = fmt_beta_txt(u["beta"], u["beta_lo"], u["beta_hi"]) if "beta" in u else "multi"
                    print(f"  {METRIC_SHORT[rf]:>14s}  beta={beta_str:>8s}  AUC={u['auc']:.3f} [{u['auc_lo']:.3f}, {u['auc_hi']:.3f}]")

    # ── Bar chart: betas (individual) + AUC (block) across strategies ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    n_buckets = len(BUCKETS)
    n_metrics = len(METRICS)
    bar_width = 0.14
    x = np.arange(n_buckets)

    for panel_idx, (decomp_name, panel_title) in enumerate([
        ("FP vs TP", "FP vs TP (moderate $|$NIE$|$ band)"),
        ("FP vs TN", "FP vs TN (off-circuit)"),
    ]):
        ax = axes[panel_idx]
        for mi, rf in enumerate(METRICS):
            vals, lo_errs, hi_errs = [], [], []
            for bkey, _ in BUCKETS:
                u = all_results[(bkey, decomp_name)]["univar"][rf]
                if u and "beta" in u:
                    vals.append(u["beta"])
                    lo_errs.append(u["beta"] - u["beta_lo"])
                    hi_errs.append(u["beta_hi"] - u["beta"])
                elif u and rf == "block":
                    vals.append(u["auc"] - 0.5)
                    lo_errs.append(u["auc"] - u["auc_lo"])
                    hi_errs.append(u["auc_hi"] - u["auc"])
                else:
                    vals.append(0)
                    lo_errs.append(0)
                    hi_errs.append(0)

            offset = (mi - n_metrics / 2 + 0.5) * bar_width
            label = METRIC_SHORT[rf]
            if rf == "block":
                label += " (AUC$-$0.5)"
            ax.bar(x + offset, vals, bar_width, yerr=[lo_errs, hi_errs],
                   label=label, color=BAR_COLORS[rf],
                   capsize=2, error_kw={"linewidth": 0.7}, edgecolor="white",
                   linewidth=0.4, zorder=3)

        ax.axhline(0, color="grey", ls="-", lw=0.6, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=7.5)
        ax.set_title(panel_title, fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx == 0:
            ax.set_ylabel("Univariate $\\beta$ (z-scored)\nblock shown as AUC$-$0.5", fontsize=9)
            ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, ncol=2)

    fig.suptitle("Ripple directional decomposition across strategies", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.fig, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.fig}")

    # ── LaTeX table ──
    with open(args.out, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Ripple directional decomposition: univariate $\\beta$ and AUC "
                "for all five ripple signals. "
                "In FP-vs-TP, ripple\\_in is consistently \\emph{negative} "
                "(TPs propagate more in-circuit ripple) while ripple\\_off is insignificant "
                "--- the signal is one-sided. "
                "In FP-vs-TN, both are positive (disruption magnitude). "
                "Block AUC (multivariate, all four features) shown in last column. "
                "* $=$ 95\\% cluster-bootstrap CI excludes 0.}\n")
        f.write("\\label{tab:ripple-direction}\n\\small\n")

        f.write("\\begin{tabular}{l rrrr r rrrr r}\n\\toprule\n")
        f.write(" & \\multicolumn{5}{c}{FP vs.\\ TP} "
                "& \\multicolumn{5}{c}{FP vs.\\ TN} \\\\\n")
        f.write("\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}\n")
        f.write(" & in & off & frac & xint & \\scriptsize{block} "
                "& in & off & frac & xint & \\scriptsize{block} \\\\\n\\midrule\n")

        for bkey, blabel in BUCKETS:
            parts_beta = [f"\\textbf{{{blabel}}}"]
            parts_auc = [""]
            for decomp in ["FP vs TP", "FP vs TN"]:
                for rf in METRICS:
                    u = all_results[(bkey, decomp)]["univar"][rf]
                    if u and "beta" in u:
                        parts_beta.append(fmt_beta(u["beta"], u["beta_lo"], u["beta_hi"]))
                        parts_auc.append(f"\\scriptsize{{{u['auc']:.3f}}}")
                    elif u and rf == "block":
                        parts_beta.append("")
                        parts_auc.append(f"\\scriptsize{{{u['auc']:.3f}}}")
                    else:
                        parts_beta.append("--")
                        parts_auc.append("")
            f.write(" & ".join(parts_beta) + " \\\\\n")
            f.write(" & ".join(parts_auc) + " \\\\[2pt]\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"Saved {args.out}")

    # ── plain text ──
    with open(args.txt, "w") as f:
        f.write(f"{'':>20s}  {'--- FP vs TP ---':^50s}  {'--- FP vs TN ---':^50s}\n")
        cols = "in-circ  off-circ     frac     xint    block"
        f.write(f"{'':>20s}  {cols}  {cols}\n")
        f.write("-" * 124 + "\n")
        for bkey, blabel in BUCKETS:
            row_b = f"{blabel:>20s}"
            row_a = f"{'':>20s}"
            for decomp in ["FP vs TP", "FP vs TN"]:
                for rf in METRICS:
                    u = all_results[(bkey, decomp)]["univar"][rf]
                    if u and "beta" in u:
                        row_b += f"  {fmt_beta_txt(u['beta'],u['beta_lo'],u['beta_hi']):>8s}"
                        row_a += f"  {u['auc']:>8.3f}"
                    elif u and rf == "block":
                        row_b += f"  {'':>8s}"
                        row_a += f"  {u['auc']:>8.3f}"
                    else:
                        row_b += f"  {'--':>8s}"
                        row_a += f"  {'--':>8s}"
            f.write(row_b + "\n")
            f.write(row_a + "\n\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
