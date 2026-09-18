"""Fig 1 — M3L AUC bar chart: univariate AUC per instrument per strategy.

Reads the trio results JSON produced by collect_interaction.py and draws a
grouped bar chart (one group per strategy bucket, one bar per instrument)
with bootstrap 95 % CI error bars.

Usage:
    python figures/fig1_m3l_auc_bars.py [--json PATH] [--out PATH]

Defaults:
    --json  interaction_analysis_allcombos_fullgap_trio/path_model.json
    --out   figures/fig1_m3l_auc_bars.pdf
"""
import argparse, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── config ──────────────────────────────────────────────────────────────────
BUCKETS = [
    ("swap_within",     "STR swap\n(within)"),
    ("swap_cross",      "STR swap\n(cross)"),
    ("STR_push_within", "STR push\n(within)"),
    ("STR_push_cross",  "STR push\n(cross)"),
    ("Gaussian_push",   "Gaussian\npush"),
]

INSTRUMENTS = [
    # (display name,             json AUC key,              trio suffix)
    ("$\\mathrm{orth}_b(\\delta)$",  "AUC_orth",            "gf"),
    ("downstream curv.",             "AUC_gap_frac",         "gf"),
    ("local curvature",              "AUC_local_gap_frac",   "lgf"),
    ("$\\mathrm{along}_s(\\delta)$", "AUC_along_nonbase_max","gf"),
]

COLORS = [
    "#4C72B0",   # orth
    "#DD8452",   # downstream curv.
    "#55A868",   # local curvature
    "#C44E52",   # along_s
]

# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="interaction_analysis_allcombos_fullgap_trio/path_model.json")
    ap.add_argument("--out", default="figures/fig1_m3l_auc_bars.pdf")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    n_buckets = len(BUCKETS)
    n_inst = len(INSTRUMENTS)
    bar_width = 0.17
    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for i, (inst_label, auc_key, suffix) in enumerate(INSTRUMENTS):
        aucs, los, his = [], [], []
        for bkey, _ in BUCKETS:
            m3l = data.get(f"M3L_{bkey}_{suffix}", {})
            coef = m3l.get("coef", {})
            v = coef.get(auc_key, {})
            est = v.get("est", 0.5)
            lo = v.get("lo", est)
            hi = v.get("hi", est)
            aucs.append(est)
            los.append(est - lo)
            his.append(hi - est)
        offset = (i - n_inst / 2 + 0.5) * bar_width
        ax.bar(x + offset, aucs, bar_width, yerr=[los, his],
               label=inst_label, color=COLORS[i],
               capsize=2, error_kw={"linewidth": 0.8}, edgecolor="white",
               linewidth=0.4, zorder=3)

    ax.axhline(0.5, color="grey", ls="--", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in BUCKETS], fontsize=9)
    ax.set_ylabel("Univariate AUC  (FP vs TN)", fontsize=10)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="", fontsize=8.5, framealpha=0.9, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Which instrument separates False Positive vs. True Negative?",
                 fontsize=12, pad=10)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
