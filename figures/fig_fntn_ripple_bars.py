"""Fig — FN-vs-TN AUC bar chart: base + per-strategy best metric + block.

For each strategy, selects the best individual ripple metric by AUC
(tiebreak: narrower CI), shows it alongside base and ripple block.
Annotates which metric was selected per strategy.

Usage:
    python figures/fig_fntn_ripple_bars.py [--csv PATH] [--out PATH]
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
METRIC_NAMES = {
    "ripple_in_orth": "ripple$_{\\mathrm{in}}$",
    "ripple_off_orth": "ripple$_{\\mathrm{off}}$",
    "ripple_frac": "ripple$_{\\mathrm{in\\_prop}}$",
    "ripple_xint_in_orth": "ripple$_{\\mathrm{xint}}$",
}
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


def fit_auc_boot(df, feats, B=200):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_fntn_ripple_bars.pdf")
    ap.add_argument("--txt", default="figures/fig_fntn_ripple_bars.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    n_buckets = len(BUCKETS)
    bar_width = 0.25
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(10, 5))

    txt_lines = [f"{'strategy':>20s}  {'model':>25s}  {'AUC':>6s}  {'95% CI':>14s}", "-" * 70]

    # precompute all individual metric AUCs per strategy to find the best
    all_data = {}
    for bi, (bkey, blabel_nl) in enumerate(BUCKETS):
        is_cross = "cross" in bkey
        base_feats = ["orth_base", "along_nonbase_max"]
        if is_cross:
            base_feats.append("along_target")
        all_needed = base_feats + RIPPLE_FEATS
        pool = build_fntn_population(df, bkey)
        clean = pool.dropna(subset=all_needed + ["label", "combo"])
        for f in all_needed:
            clean = clean[np.isfinite(clean[f])]
        clean = zscore_within_combo(clean, all_needed)
        all_data[bkey] = {"clean": clean, "base_feats": base_feats}

    # compute AUCs for base, each individual, and block per strategy
    results = {}
    for bi, (bkey, _) in enumerate(BUCKETS):
        clean = all_data[bkey]["clean"]
        base_feats = all_data[bkey]["base_feats"]

        auc_b, lo_b, hi_b = fit_auc_boot(clean, base_feats)
        results[(bkey, "base")] = (auc_b, lo_b, hi_b, "base")

        best_auc, best_metric = -1, None
        best_width = 999
        for rf in RIPPLE_INDIVIDUAL:
            feats = list(base_feats) + [rf]
            auc, lo, hi = fit_auc_boot(clean, feats)
            if auc is None:
                continue
            results[(bkey, rf)] = (auc, lo, hi, METRIC_SHORT[rf])
            width = hi - lo
            if auc > best_auc or (auc == best_auc and width < best_width):
                best_auc = auc
                best_metric = rf
                best_width = width
        results[(bkey, "best")] = results[(bkey, best_metric)] if best_metric else (0.5, 0.5, 0.5, "?")

        auc_bl, lo_bl, hi_bl = fit_auc_boot(clean, list(base_feats) + RIPPLE_FEATS)
        results[(bkey, "block")] = (auc_bl, lo_bl, hi_bl, "block")

    # plot: base, best individual, block
    colors = ["#AAAAAA", "#DD8452", "#8172B3"]
    for mi, model_key in enumerate(["base", "best", "block"]):
        aucs, lo_errs, hi_errs = [], [], []
        labels_used = []
        for bi, (bkey, _) in enumerate(BUCKETS):
            auc, lo, hi, label = results[(bkey, model_key)]
            if auc is None:
                auc, lo, hi = 0.5, 0.5, 0.5
            aucs.append(auc)
            lo_errs.append(auc - lo)
            hi_errs.append(hi - auc)
            labels_used.append(label)

        if model_key == "base":
            bar_label = "base (orth + along)"
        elif model_key == "best":
            bar_label = "+ best individual ripple"
        else:
            bar_label = "+ ripple block"

        offset = (mi - 1) * bar_width
        bars = ax.bar(x + offset, aucs, bar_width, yerr=[lo_errs, hi_errs],
                      label=bar_label, color=colors[mi],
                      capsize=3, error_kw={"linewidth": 0.8}, edgecolor="white",
                      linewidth=0.4, zorder=3)

        # annotate best individual metric name on each bar
        if model_key == "best":
            for bi, bar in enumerate(bars):
                metric_name = labels_used[bi]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi_errs[bi] + 0.008,
                        metric_name, ha="center", va="bottom", fontsize=7, fontstyle="italic",
                        color=colors[mi])

        blabel = [b[1].replace('\n', ' ') for b in BUCKETS]
        for bi in range(n_buckets):
            txt_lines.append(f"{blabel[bi]:>20s}  {bar_label + ' (' + labels_used[bi] + ')':>25s}  "
                             f"{aucs[bi]:>6.3f}  [{aucs[bi]-lo_errs[bi]:.3f}, {aucs[bi]+hi_errs[bi]:.3f}]")

    ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=9)
    ax.set_ylabel("AUC  (FN vs TN)", fontsize=10)
    ax.set_ylim(0.55, 1.0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Can ripple identify dormant circuit members?\n"
                 "(FN = in-circuit below threshold, TN = off-circuit below threshold)",
                 fontsize=11, pad=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
