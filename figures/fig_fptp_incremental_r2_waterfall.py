"""Incremental pseudo-R² waterfall (FP vs TP): moderate-|NIE| band.

Four pathways:
  A: curv → +ripple → +orth → +along
  B: ripple → +curv → +orth → +along
  C: orth → +curv → +ripple → +along
  D: orth → +ripple → +curv → +along

A/B compare curv-vs-ripple complementarity.
C/D show standard entry orders.
Local curvature throughout. C=10.0.

Usage:
    python figures/fig_fptp_incremental_r2_waterfall.py [--csv PATH]
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
    ("STR_swap_within",  "swap(within)"),
    ("STR_swap_cross",   "swap(cross)"),
    ("STR_push_within",  "push(within)"),
    ("STR_push_cross",   "push(cross)"),
    ("Gaussian_push",    "Gauss"),
]

RIPPLE_ALL = ["ripple_in_orth", "ripple_off_orth", "ripple_frac", "ripple_xint_in_orth"]
ALONG = "along_nonbase_max"
CURV = "local_gap_frac"

COLORS = {
    "orth": "#7f7f7f",
    "along": "#4C72B0",
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
    ap.add_argument("--out", default="figures/fig_fptp_incremental_r2_waterfall.pdf")
    ap.add_argument("--txt", default="figures/fig_fptp_incremental_r2_waterfall.txt")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    all_data = {}
    txt_lines = [f"{'strategy':>14s}  {'seq':>3s}  {'orth':>6s}  {'along':>6s}  "
                 f"{'curv':>6s}  {'ripple':>6s}  {'total':>6s}  {'n':>5s}",
                 "-" * 67]

    for bkey, blabel in BUCKETS:
        pool = build_moderate_band(df, bkey)
        all_feats = list(set(["orth_base", ALONG, CURV] + RIPPLE_ALL))
        pool = zscore(pool, all_feats)
        z_all = [f"z_{f}" for f in all_feats]
        clean = pool.dropna(subset=z_all + ["label"]).copy()
        X_full = clean[z_all].values
        mask = np.all(np.isfinite(X_full), axis=1)
        clean = clean[mask].reset_index(drop=True)
        y = clean["label"].values

        # Component R²s
        r2_orth = get_r2(clean, y, ["orth_base"])
        r2_curv = get_r2(clean, y, [CURV])
        r2_rip = get_r2(clean, y, RIPPLE_ALL)

        r2_curv_rip = get_r2(clean, y, [CURV] + RIPPLE_ALL)
        r2_curv_rip_orth = get_r2(clean, y, [CURV, "orth_base"] + RIPPLE_ALL)

        r2_orth_curv = get_r2(clean, y, ["orth_base", CURV])
        r2_orth_rip = get_r2(clean, y, ["orth_base"] + RIPPLE_ALL)
        r2_orth_curv_rip = get_r2(clean, y, ["orth_base", CURV] + RIPPLE_ALL)

        r2_full = get_r2(clean, y, ["orth_base", ALONG, CURV] + RIPPLE_ALL)

        # A: curv → +ripple → +orth → +along
        seqA = [r2_curv,
                r2_curv_rip - r2_curv,
                r2_curv_rip_orth - r2_curv_rip,
                r2_full - r2_curv_rip_orth]

        # B: ripple → +curv → +orth → +along
        seqB = [r2_rip,
                r2_curv_rip - r2_rip,
                r2_curv_rip_orth - r2_curv_rip,
                r2_full - r2_curv_rip_orth]

        # C: orth → +curv → +ripple → +along
        seqC = [r2_orth,
                r2_orth_curv - r2_orth,
                r2_orth_curv_rip - r2_orth_curv,
                r2_full - r2_orth_curv_rip]

        # D: orth → +ripple → +curv → +along
        seqD = [r2_orth,
                r2_orth_rip - r2_orth,
                r2_orth_curv_rip - r2_orth_rip,
                r2_full - r2_orth_curv_rip]

        all_data[blabel] = {
            "seqA": seqA, "seqB": seqB, "seqC": seqC, "seqD": seqD,
            "total": r2_full, "n": len(y),
        }

        # txt: instrument-aligned columns
        # A: [curv, ripple, orth, along]
        txt_lines.append(
            f"{blabel:>14s}  {'A':>3s}  {seqA[2]:>6.3f}  {seqA[3]:>6.3f}  "
            f"{seqA[0]:>6.3f}  {seqA[1]:>6.3f}  {r2_full:>6.3f}  {len(y):>5d}")
        # B: [ripple, curv, orth, along]
        txt_lines.append(
            f"{'':>14s}  {'B':>3s}  {seqB[2]:>6.3f}  {seqB[3]:>6.3f}  "
            f"{seqB[1]:>6.3f}  {seqB[0]:>6.3f}  {r2_full:>6.3f}  {'':>5s}")
        # C: [orth, curv, ripple, along]
        txt_lines.append(
            f"{'':>14s}  {'C':>3s}  {seqC[0]:>6.3f}  {seqC[3]:>6.3f}  "
            f"{seqC[1]:>6.3f}  {seqC[2]:>6.3f}  {r2_full:>6.3f}  {'':>5s}")
        # D: [orth, ripple, curv, along]
        txt_lines.append(
            f"{'':>14s}  {'D':>3s}  {seqD[0]:>6.3f}  {seqD[3]:>6.3f}  "
            f"{seqD[2]:>6.3f}  {seqD[1]:>6.3f}  {r2_full:>6.3f}  {'':>5s}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 5.5))
    n_bars = 4
    bar_w = 0.20
    gap = 0.65

    for i, (bkey, blabel) in enumerate(BUCKETS):
        d = all_data[blabel]
        cx = i * (n_bars * bar_w + gap)

        sequences = [
            (d["seqA"], ["curv", "ripple", "orth", "along"], "A"),
            (d["seqB"], ["ripple", "curv", "orth", "along"], "B"),
            (d["seqC"], ["orth", "curv", "ripple", "along"], "C"),
            (d["seqD"], ["orth", "ripple", "curv", "along"], "D"),
        ]

        for j, (seq, order, label) in enumerate(sequences):
            xp = cx + j * bar_w
            base = 0
            for val, col in zip(seq, order):
                ax.bar(xp, val, bar_w, bottom=base,
                       color=COLORS[col], edgecolor="white", lw=0.5, zorder=3)
                if val > 0.025:
                    ax.text(xp, base + val / 2, f"{val:.2f}",
                            ha="center", va="center", fontsize=5,
                            color="white", fontweight="bold")
                base += val

            ax.text(xp, -0.020, label,
                    ha="center", va="top", fontsize=6, fontstyle="italic", color="#555")

        mid = cx + (n_bars - 1) * bar_w / 2
        ax.text(mid, -0.050, f"{blabel}\nn={d['n']}",
                ha="center", va="top", fontsize=7.5)

    legend_elements = [
        Patch(facecolor=COLORS["orth"], edgecolor="white", label="orth"),
        Patch(facecolor=COLORS["along"], edgecolor="white", label=r"along$_s$"),
        Patch(facecolor=COLORS["curv"], edgecolor="white", label="local curvature"),
        Patch(facecolor=COLORS["ripple"], edgecolor="white", label="ripple block"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, ncol=4,
              loc="upper left", bbox_to_anchor=(0.0, 1.15), framealpha=0.9)

    ax.text(0.99, 1.15,
            u"A: curv → +ripple → +orth → +along\n"
            u"B: ripple → +curv → +orth → +along\n"
            u"C: orth → +curv → +ripple → +along\n"
            u"D: orth → +ripple → +curv → +along",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f0f0", ec="#cccccc", alpha=0.9))

    ax.set_ylabel(u"McFadden's pseudo-R²", fontsize=10)
    max_total = max(d["total"] for d in all_data.values())
    ax.set_ylim(-0.07, max_total + 0.06)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(u"Incremental variance: instrument complementarity (FP vs TP, moderate-|NIE| band)",
                 fontsize=10.5, pad=22)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")

    with open(args.txt, "w") as f:
        f.write("\n".join(txt_lines) + "\n")
    print(f"Saved {args.txt}")


if __name__ == "__main__":
    main()
