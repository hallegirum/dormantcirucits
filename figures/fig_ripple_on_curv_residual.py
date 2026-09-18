"""Fig — Ripple on curvature residual: per-strategy best metric + block.

Fit curvature model (base + gap_frac), identify misclassified rows,
then show ripple discriminates on the subset curvature gets wrong.

Per strategy: selects the best individual ripple metric by AUC on the
misclassified subset, annotates which metric was selected.

Usage:
    python figures/fig_ripple_on_curv_residual.py [--csv PATH] [--out PATH]
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
BEST_CURV = "gap_frac"

METRIC_SHORT = {
    "ripple_in_orth": "in",
    "ripple_off_orth": "off",
    "ripple_frac": "in_prop",
    "ripple_xint_in_orth": "xint",
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


def get_misclassified(df, curv_feats, threshold=0.5):
    z_feats = [f"z_{f}" for f in curv_feats]
    sub = df.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    sub_clean = sub.iloc[np.where(mask)[0]].copy()

    if len(np.unique(y)) < 2 or len(y) < 20:
        return pd.DataFrame(), 0.0

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
    lr.fit(X, y)
    probs = lr.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(float)

    sub_clean = sub_clean.copy()
    sub_clean["curv_wrong"] = (preds != y)
    return sub_clean, roc_auc_score(y, probs)


def fit_auc_boot(df, feats, B=200):
    z_feats = [f"z_{f}" for f in feats]
    sub = df.dropna(subset=z_feats + ["label"]).copy()
    X = sub[z_feats].values.astype(float)
    y = sub["label"].values
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]

    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None, None

    lr = LogisticRegression(max_iter=2000, solver="lbfgs", C=10.0)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

    combos = sub["combo"].values[mask] if "combo" in sub.columns else np.arange(len(y))
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
    ap.add_argument("--out", default="figures/fig_ripple_on_curv_residual.pdf")
    ap.add_argument("--txt", default="figures/fig_ripple_on_curv_residual.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    n_buckets = len(BUCKETS)
    bar_width = 0.3
    x = np.arange(n_buckets)
    colors = ["#DD8452", "#8172B3"]

    txt_lines = []

    for panel_idx, (decomp_name, build_fn, panel_title) in enumerate([
        ("FP vs TN", build_fptn_population, "FP vs TN — curvature mistakes"),
        ("FP vs TP", build_fptp_moderate, "FP vs TP — curvature mistakes"),
    ]):
        ax = axes[panel_idx]
        txt_lines.append(f"\n{'='*75}")
        txt_lines.append(f"  {decomp_name} — ripple on curvature residual")
        txt_lines.append(f"{'='*75}\n")
        txt_lines.append(f"{'strategy':>20s}  {'n_wrong':>7s}  {'model':>18s}  {'AUC':>6s}  {'95% CI':>14s}")
        txt_lines.append("-" * 72)

        # precompute misclassified subsets and find best metric per strategy
        strategy_results = {}
        for bi, (bkey, blabel_nl) in enumerate(BUCKETS):
            blabel = blabel_nl.replace('\n', ' ')
            is_cross = "cross" in bkey
            base_feats = ["orth_base", "along_nonbase_max"]
            if is_cross:
                base_feats.append("along_target")
            curv_feats = base_feats + [BEST_CURV]
            all_needed = curv_feats + RIPPLE_FEATS

            pool = build_fn(df, bkey)
            clean = pool.dropna(subset=all_needed + ["label", "combo"])
            for f in all_needed:
                clean = clean[np.isfinite(clean[f])]
            clean = zscore_within_combo(clean, all_needed)

            scored, curv_auc = get_misclassified(clean, curv_feats)

            if len(scored) == 0 or scored["curv_wrong"].sum() < 20:
                strategy_results[bkey] = None
                txt_lines.append(f"{blabel:>20s}  {'<20':>7s}  {'--':>18s}  {'--':>6s}  {'--':>14s}")
                continue

            wrong_subset = scored[scored["curv_wrong"]].copy()
            wrong_subset = zscore_within_combo(wrong_subset, RIPPLE_FEATS)
            n_wrong = int(scored["curv_wrong"].sum())

            # evaluate all individual metrics
            best_auc, best_metric, best_width = -1, None, 999
            metric_results = {}
            for rf in RIPPLE_INDIVIDUAL:
                auc, lo, hi = fit_auc_boot(wrong_subset, [rf])
                if auc is not None:
                    metric_results[rf] = (auc, lo, hi)
                    width = hi - lo
                    if auc > best_auc or (auc == best_auc and width < best_width):
                        best_auc, best_metric, best_width = auc, rf, width

            # block
            auc_bl, lo_bl, hi_bl = fit_auc_boot(wrong_subset, RIPPLE_FEATS)

            strategy_results[bkey] = {
                "n_wrong": n_wrong,
                "best_metric": best_metric,
                "best": metric_results.get(best_metric, (0.5, 0.5, 0.5)),
                "block": (auc_bl, lo_bl, hi_bl) if auc_bl else (0.5, 0.5, 0.5),
            }

        # plot: best individual, block
        for mi, model_key in enumerate(["best", "block"]):
            aucs, lo_errs, hi_errs = [], [], []
            for bi, (bkey, _) in enumerate(BUCKETS):
                sr = strategy_results[bkey]
                if sr is None:
                    aucs.append(0.5); lo_errs.append(0); hi_errs.append(0)
                    continue
                auc, lo, hi = sr[model_key]
                aucs.append(auc)
                lo_errs.append(auc - lo)
                hi_errs.append(hi - auc)

            if model_key == "best":
                bar_label = "best individual ripple"
            else:
                bar_label = "ripple block"

            offset = (mi - 0.5) * bar_width
            bars = ax.bar(x + offset, aucs, bar_width, yerr=[lo_errs, hi_errs],
                          label=bar_label, color=colors[mi],
                          capsize=3, error_kw={"linewidth": 0.8}, edgecolor="white",
                          linewidth=0.4, zorder=3)

            if model_key == "best":
                for bi, bar in enumerate(bars):
                    sr = strategy_results[BUCKETS[bi][0]]
                    if sr and sr["best_metric"]:
                        mname = METRIC_SHORT[sr["best_metric"]]
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + hi_errs[bi] + 0.01,
                                mname, ha="center", va="bottom", fontsize=7,
                                fontstyle="italic", color=colors[mi])

            blabel = [b[1].replace('\n', ' ') for b in BUCKETS]
            for bi in range(n_buckets):
                sr = strategy_results[BUCKETS[bi][0]]
                if sr:
                    auc, lo, hi = sr[model_key]
                    mname = METRIC_SHORT[sr["best_metric"]] if model_key == "best" else "block"
                    txt_lines.append(f"{blabel[bi]:>20s}  {sr['n_wrong']:>7d}  "
                                     f"{bar_label + ' (' + mname + ')':>18s}  {auc:>6.3f}  [{lo:.3f}, {hi:.3f}]")

        ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8.5)
        ax.set_title(panel_title, fontsize=11, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx == 0:
            ax.set_ylabel("AUC on curvature-misclassified subset", fontsize=10)
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
        ax.set_ylim(0.35, 1.05)

    fig.suptitle("Ripple discriminates where curvature fails\n"
                 f"(base curvature model: downstream gap\\_frac)", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
