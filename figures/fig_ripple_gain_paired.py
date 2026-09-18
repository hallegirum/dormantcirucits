"""Fig — Which base feature absorbs ripple? 1x2 panel comparing gains.

Left panel: FP-vs-TN. Right panel: FP-vs-TP.
Three series per panel: orth-only, along-only, best-curv-only as base.
Y-axis = AUC(base+best_ripple) - AUC(base), paired cluster-bootstrap CIs.

Best ripple metric selected per strategy per decomposition per base model.
Best curvature metric selected per strategy per decomposition.

Usage:
    python figures/fig_ripple_gain_paired.py [--csv PATH] [--out PATH]
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
    "gap_frac": "downstream",
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


def compute_gain_with_ci(clean, base_feats, ripple_feat, B=500):
    """Bootstrap the AUC gain = AUC(base+ripple) - AUC(base), paired."""
    full_feats = list(base_feats) + [ripple_feat]
    z_base = [f"z_{f}" for f in base_feats]
    z_full = [f"z_{f}" for f in full_feats]
    sub = clean.dropna(subset=z_full + ["label"]).copy()
    X_b = sub[z_base].values.astype(float)
    X_f = sub[z_full].values.astype(float)
    y = sub["label"].values
    combos = sub["combo"].values
    mask = np.all(np.isfinite(X_b), axis=1) & np.all(np.isfinite(X_f), axis=1)
    X_b, X_f, y, combos = X_b[mask], X_f[mask], y[mask], combos[mask]

    if len(np.unique(y)) < 2 or len(y) < 20:
        return None, None, None

    lr_b = LogisticRegression(max_iter=2000, C=10.0)
    lr_b.fit(X_b, y)
    auc_b = roc_auc_score(y, lr_b.predict_proba(X_b)[:, 1])

    lr_f = LogisticRegression(max_iter=2000, C=10.0)
    lr_f.fit(X_f, y)
    auc_f = roc_auc_score(y, lr_f.predict_proba(X_f)[:, 1])

    gain = auc_f - auc_b

    uc = np.unique(combos)
    rng = np.random.RandomState(0)
    gboots = []
    for _ in range(B):
        ic = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([np.where(combos == c)[0] for c in ic])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            lb = LogisticRegression(max_iter=2000, C=10.0)
            lb.fit(X_b[idx], yb)
            ab = roc_auc_score(yb, lb.predict_proba(X_b[idx])[:, 1])
            lf = LogisticRegression(max_iter=2000, C=10.0)
            lf.fit(X_f[idx], yb)
            af = roc_auc_score(yb, lf.predict_proba(X_f[idx])[:, 1])
            gboots.append(af - ab)
        except Exception:
            continue

    lo = np.percentile(gboots, 2.5) if gboots else gain
    hi = np.percentile(gboots, 97.5) if gboots else gain
    return gain, lo, hi


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
    ap.add_argument("--out", default="figures/fig_ripple_gain_paired.pdf")
    ap.add_argument("--txt", default="figures/fig_ripple_gain_paired.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    base_configs = [
        ("orth", "#4C72B0", "D",  lambda cross: ["orth_base"]),
        ("along", "#55A868", "s", lambda cross: ["along_nonbase_max"] + (["along_target"] if cross else [])),
        ("curv (best)", "#C44E52", "o", None),
    ]

    decomps = [
        ("FP vs TN", build_fptn_population),
        ("FP vs TP", build_fptp_moderate),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    n_bases = len(base_configs)
    offsets = np.linspace(-0.25, 0.25, n_bases)
    x = np.arange(len(BUCKETS))

    txt_lines = [f"{'strategy':>20s}  {'decomp':>10s}  {'base':>14s}  {'best_rip':>8s}  "
                 f"{'gain':>7s}  {'95% CI':>18s}  {'curv_used':>12s}",
                 "-" * 100]

    for pi, (decomp_name, build_fn) in enumerate(decomps):
        ax = axes[pi]

        for ci, (cfg_name, color, marker, cfg_fn) in enumerate(base_configs):
            gains, lo_errs, hi_errs = [], [], []
            best_names = []

            for bi, (bkey, blabel) in enumerate(BUCKETS):
                is_cross = "cross" in bkey
                all_feats = (["orth_base", "along_nonbase_max"] + CURV_CANDIDATES
                             + RIPPLE_FEATS)
                if is_cross:
                    all_feats.append("along_target")

                pool = build_fn(df, bkey)
                clean = pool.copy()
                clean = zscore_within_combo(clean, all_feats)

                curv_used = ""
                if cfg_fn is not None:
                    base_feats = cfg_fn(is_cross)
                else:
                    best_curv = pick_best_curv(clean)
                    base_feats = [best_curv]
                    curv_used = CURV_SHORT[best_curv]

                best_gain, best_metric = -999, None
                best_lo, best_hi = 0, 0
                for rf in RIPPLE_FEATS:
                    g, lo, hi = compute_gain_with_ci(clean, base_feats, rf)
                    if g is not None and g > best_gain:
                        best_gain, best_metric = g, rf
                        best_lo, best_hi = lo, hi

                if best_metric:
                    gains.append(best_gain)
                    lo_errs.append(best_gain - best_lo)
                    hi_errs.append(best_hi - best_gain)
                    best_names.append(METRIC_SHORT[best_metric])
                else:
                    gains.append(0)
                    lo_errs.append(0)
                    hi_errs.append(0)
                    best_names.append("?")

                sig = "*" if best_lo > 0 else ""
                txt_lines.append(
                    f"{blabel:>20s}  {decomp_name:>10s}  {cfg_name:>14s}  "
                    f"{METRIC_SHORT.get(best_metric, '?'):>8s}  "
                    f"{best_gain:>+7.3f}{sig}  [{best_lo:+.3f}, {best_hi:+.3f}]"
                    f"  {curv_used:>12s}")

            ax.errorbar(x + offsets[ci], gains, yerr=[lo_errs, hi_errs],
                        fmt=marker, color=color, markersize=7, capsize=3.5,
                        capthick=1.0, linewidth=0, elinewidth=1.0,
                        label=cfg_name, zorder=3)

            for bi in range(len(BUCKETS)):
                ax.annotate(best_names[bi],
                            xy=(x[bi] + offsets[ci],
                                gains[bi] + hi_errs[bi] + 0.005),
                            ha="center", va="bottom", fontsize=8.5,
                            fontweight="bold", color="black")

        ax.axhline(0, color="grey", ls="-", lw=0.6, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=8.5)
        ax.set_title(decomp_name, fontsize=12, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("AUC gain from adding best ripple\nto single-feature base", fontsize=10)
            ax.legend(loc="upper center", fontsize=9, framealpha=0.9,
                      bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Which base feature absorbs ripple's signal?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
