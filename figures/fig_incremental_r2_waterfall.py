"""Incremental pseudo-R² waterfall: feature addition order reveals curvature as centerpiece.

Two stacked bars per strategy show the same total R² (full model) decomposed by
addition order:
  A: orth → +curv → +ripple
  B: orth → +ripple → +curv

The height difference of each color between A and B shows whether a feature's
contribution depends on what came before it. Curvature (green) stays large in
both orderings; ripple (purple) shrinks without curvature for push/Gauss —
curvature is the unlock.

FP-vs-TP (moderate |NIE| band), matched sample. C=10.0, seed=0.

Usage:
    python figures/fig_incremental_r2_waterfall.py [--csv PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

BUCKETS = [
    ("STR_swap_within",  "swap(w)"),
    ("STR_swap_cross",   "swap(x)"),
    ("STR_push_within",  "push(w)"),
    ("STR_push_cross",   "push(x)"),
    ("Gaussian_push",    "Gauss"),
]

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
CURV = "local_gap_frac"

COLORS = {
    "orth": "#7f7f7f",
    "curv": "#2ca02c",
    "ripple": "#9467bd",
}


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


def build_moderate_band(df, bucket):
    sub = df[df["bucket"] == bucket].copy()
    ic = sub[sub["in_base_circuit"] == True]
    oc = sub[sub["in_base_circuit"] == False]
    thresholds = ic.groupby("combo")["abs_NIE"].median().to_dict()
    def above(row):
        thr = thresholds.get(row["combo"])
        return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr
    pool = pd.concat([ic, oc])
    pool = pool[pool.apply(above, axis=1)].copy()
    pool_median = pool["abs_NIE"].median()
    pool = pool[pool["abs_NIE"] <= pool_median].copy()
    pool["label"] = (~pool["in_base_circuit"]).astype(float)
    return pool


def zscore(df, feats):
    out = df.copy()
    for f in feats:
        out[f"z_{f}"] = out.groupby("combo")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-12))
    return out


def mcfadden_r2(X, y):
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return 0.0
    lr = LogisticRegression(max_iter=2000, C=10.0)
    lr.fit(X, y)
    ll_model = -log_loss(y, lr.predict_proba(X), normalize=False)
    p1 = y.mean()
    ll_null = -log_loss(y, np.full((len(y), 2), [1 - p1, p1]), normalize=False)
    if ll_null == 0:
        return 0.0
    return 1 - ll_model / ll_null


def get_r2(clean, y, feats):
    z = [f"z_{f}" for f in feats]
    return mcfadden_r2(clean[z].values, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_ripplefull/rows.csv")
    ap.add_argument("--out", default="figures/fig_incremental_r2_waterfall.pdf")
    ap.add_argument("--txt", default="figures/fig_incremental_r2_waterfall.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    all_data = {}
    txt_lines = [f"{'strategy':>10s}  {'seq':>5s}  {'orth':>6s}  {'curv':>6s}  "
                 f"{'ripple':>6s}  {'total':>6s}  {'n':>5s}",
                 "-" * 55]

    for bkey, blabel in BUCKETS:
        pool = build_moderate_band(df, bkey)
        all_feats = list(set(["orth_base", CURV] + RIPPLE_ALL))
        pool = zscore(pool, all_feats)
        z_all = [f"z_{f}" for f in all_feats]
        clean = pool.dropna(subset=z_all + ["label"]).copy()
        X_full = clean[z_all].values
        mask = np.all(np.isfinite(X_full), axis=1)
        clean = clean[mask]
        y = clean["label"].values

        r2_orth = get_r2(clean, y, ["orth_base"])
        r2_orth_curv = get_r2(clean, y, ["orth_base", CURV])
        r2_orth_rip = get_r2(clean, y, ["orth_base"] + RIPPLE_ALL)
        r2_full = get_r2(clean, y, ["orth_base", CURV] + RIPPLE_ALL)

        seqA = [r2_orth, r2_orth_curv - r2_orth, r2_full - r2_orth_curv]
        seqB = [r2_orth, r2_orth_rip - r2_orth, r2_full - r2_orth_rip]

        all_data[blabel] = {"seqA": seqA, "seqB": seqB, "total": r2_full, "n": len(y)}

        txt_lines.append(
            f"{blabel:>10s}  {'A':>5s}  {seqA[0]:>6.3f}  {seqA[1]:>6.3f}  "
            f"{seqA[2]:>6.3f}  {r2_full:>6.3f}  {len(y):>5d}")
        txt_lines.append(
            f"{'':>10s}  {'B':>5s}  {seqB[0]:>6.3f}  {seqB[2]:>6.3f}  "
            f"{seqB[1]:>6.3f}  {r2_full:>6.3f}  {'':>5s}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(11, 5))
    bar_w = 0.32
    gap = 0.9

    for i, (bkey, blabel) in enumerate(BUCKETS):
        d = all_data[blabel]
        cx = i * (2 * bar_w + gap)

        for j, (seq, order) in enumerate([
            (d["seqA"], ["orth", "curv", "ripple"]),
            (d["seqB"], ["orth", "ripple", "curv"]),
        ]):
            xp = cx + j * bar_w
            base = 0
            for val, col in zip(seq, order):
                ax.bar(xp, val, bar_w, bottom=base,
                       color=COLORS[col], edgecolor="white", lw=0.5, zorder=3)
                if val > 0.03:
                    ax.text(xp, base + val / 2, f"{val:.2f}",
                            ha="center", va="center", fontsize=6.5,
                            color="white", fontweight="bold")
                base += val

            ax.text(xp, -0.022, "A" if j == 0 else "B",
                    ha="center", va="top", fontsize=7, fontstyle="italic", color="#555")

        mid = cx + bar_w / 2
        ax.text(mid, -0.055, f"{blabel}\nn={d['n']}",
                ha="center", va="top", fontsize=8.5)

    legend_elements = [
        Patch(facecolor=COLORS["orth"], edgecolor="white", label="orth"),
        Patch(facecolor=COLORS["curv"], edgecolor="white", label="curvature"),
        Patch(facecolor=COLORS["ripple"], edgecolor="white", label="ripple block"),
    ]
    ax.text(0.98, 0.98,
            "A: orth → +curv → +ripple\nB: orth → +ripple → +curv\n(no along in either path)",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc", alpha=0.9))

    ax.legend(handles=legend_elements, fontsize=8.5, loc="center right",
              framealpha=0.9, bbox_to_anchor=(0.72, 0.85))

    ax.set_ylabel("McFadden’s pseudo-R²", fontsize=10)
    ax.set_ylim(-0.08, max(d["total"] for d in all_data.values()) + 0.06)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Incremental variance: regime-dependent complementarity (FP vs TP, matched sample)",
                 fontsize=10.5, pad=14)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
