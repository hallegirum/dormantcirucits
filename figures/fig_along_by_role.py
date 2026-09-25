"""along_nonbase_max by component role (borrowed vs epotic) across regimes.

Splits FPs into borrowed (home_tasks non-empty) and epotic (home_tasks empty),
shows along_nb distributions per regime, with TP and FN medians as reference.

Usage:
    python figures/fig_along_by_role.py [--csv PATH] [--out PATH]
"""
import argparse, ast, warnings
import numpy as np
import pandas as pd
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

FEATS = [
    ("along_nonbase_max", r"$\mathrm{along}_{s}(\delta)$"),
    ("orth_base",         r"$\mathrm{orth}_{b}(\delta)$"),
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


def classify_role(row):
    """Classify a component into TP, FN, FP-borrowed, FP-epotic, TN-borrowed, TN-epotic."""
    in_base = row["in_base_circuit"]
    above = row["above_threshold"]
    if in_base:
        return "TP" if above else "FN"
    has_home = len(row["home_list"]) > 0
    if above:
        return "FP-borrowed" if has_home else "FP-epotic"
    else:
        return "TN-borrowed" if has_home else "TN-epotic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out",
                    default="figures/fig_along_by_role.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)
    df["home_list"] = df["home_tasks"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # Per-combo in-circuit median threshold (same as FP-vs-TP population)
    # First pass: classify components per bucket
    bucket_frames = {}
    for bucket_key, bucket_label in BUCKETS:
        sub = df[df["bucket"] == bucket_key].copy()
        in_circ = sub[sub["in_base_circuit"] == True]
        thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

        def above(row):
            thr = thresholds.get(row["combo"])
            return thr is not None and pd.notna(row["abs_NIE"]) and row["abs_NIE"] >= thr

        sub["above_threshold"] = sub.apply(above, axis=1)
        sub["role"] = sub.apply(classify_role, axis=1)
        bucket_frames[bucket_key] = sub

    # Collect stats per feature
    all_results = {}
    for feat_col, feat_label in FEATS:
        results = {}
        print(f"\n{'='*60}")
        print(f"  {feat_label}")
        print(f"{'='*60}")
        for bucket_key, bucket_label in BUCKETS:
            sub = bucket_frames[bucket_key]
            role_stats = {}
            for role in ["TP", "FN", "FP-borrowed", "FP-epotic"]:
                vals = sub.loc[sub["role"] == role, feat_col].dropna()
                vals = vals[np.isfinite(vals)]
                role_stats[role] = {
                    "median": vals.median() if len(vals) > 0 else np.nan,
                    "mean": vals.mean() if len(vals) > 0 else np.nan,
                    "q25": vals.quantile(0.25) if len(vals) > 0 else np.nan,
                    "q75": vals.quantile(0.75) if len(vals) > 0 else np.nan,
                    "n": len(vals),
                    "values": vals.values,
                }
            results[bucket_key] = role_stats

            label = bucket_label.replace("\n", " ")
            print(f"\n{label}:")
            for role in ["TP", "FN", "FP-borrowed", "FP-epotic"]:
                s = role_stats[role]
                print(f"  {role:15s}  n={s['n']:5d}  "
                      f"median={s['median']:+.4f}  mean={s['mean']:+.4f}  "
                      f"IQR=[{s['q25']:.4f}, {s['q75']:.4f}]")
        all_results[feat_col] = results

    # ── plot: two rows (along, orth) × 5 regimes ──
    fig, all_axes = plt.subplots(len(FEATS), len(BUCKETS),
                                  figsize=(14, 4.0 * len(FEATS)), sharey="row")

    colors = {
        "FP-borrowed": "#d73027",
        "FP-epotic": "#fc8d59",
    }
    ref_styles = {
        "TP": {"color": "#1a9850", "ls": "-", "lw": 1.8},
        "FN": {"color": "#91bfdb", "ls": "--", "lw": 1.5},
    }

    for row, (feat_col, feat_label) in enumerate(FEATS):
        axes = all_axes[row] if len(FEATS) > 1 else all_axes
        for i, (bucket_key, bucket_label) in enumerate(BUCKETS):
            ax = axes[i]
            stats = all_results[feat_col][bucket_key]

            data_to_plot = []
            labels_to_plot = []
            box_colors = []
            for role, label in [("FP-borrowed", "Borrowed"), ("FP-epotic", "Epotic")]:
                vals = stats[role]["values"]
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    labels_to_plot.append(f"{label}\n(n={stats[role]['n']})")
                    box_colors.append(colors[role])

            if data_to_plot:
                bp = ax.boxplot(data_to_plot, positions=range(len(data_to_plot)),
                               widths=0.5, patch_artist=True,
                               showfliers=False, medianprops=dict(color="black", lw=1.5))
                for patch, c in zip(bp["boxes"], box_colors):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.7)
                ax.set_xticks(range(len(labels_to_plot)))
                ax.set_xticklabels(labels_to_plot, fontsize=8)

            for role, style in ref_styles.items():
                med = stats[role]["median"]
                if not np.isnan(med):
                    ax.axhline(med, **style, zorder=1,
                              label=f"{role} median" if i == 0 and row == 0 else None)

            if row == 0:
                ax.set_title(bucket_label.replace("\n", " "), fontsize=9.5)
            if i == 0:
                ax.set_ylabel(feat_label, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2)

    all_axes[0][0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle("Divergence metrics by component role (FP subpopulations)",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nSaved {args.out}")

    # ── summary table ──
    txt_out = args.out.rsplit(".", 1)[0] + ".txt"
    with open(txt_out, "w") as f:
        for feat_col, feat_label in FEATS:
            f.write(f"\n{feat_label}\n")
            f.write(f"{'strategy':>20s}  {'role':>15s}  {'n':>6s}  "
                    f"{'median':>10s}  {'mean':>10s}  {'q25':>10s}  {'q75':>10s}\n")
            f.write("-" * 95 + "\n")
            for bucket_key, bucket_label in BUCKETS:
                label = bucket_label.replace("\n", " ")
                for role in ["TP", "FN", "FP-borrowed", "FP-epotic"]:
                    s = all_results[feat_col][bucket_key][role]
                    f.write(f"{label:>20s}  {role:>15s}  {s['n']:>6d}  "
                            f"{s['median']:>+10.4f}  {s['mean']:>+10.4f}  "
                            f"{s['q25']:>10.4f}  {s['q75']:>10.4f}\n")
            f.write("\n")
    print(f"Saved {txt_out}")


if __name__ == "__main__":
    main()
