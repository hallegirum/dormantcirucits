"""Fig 8 — FP anatomy: borrowed vs epiotic false positives.

2x2 panels: rows = swap / push, cols = along_t / along_nonbase_max.
Y = curvature (local for push, downstream for swap).
Color = borrowed (home_tasks non-empty) vs epiotic (home_tasks empty).

Usage:
    python figures/fig8_fp_anatomy_scatter.py [--csv PATH] [--out PATH]
"""
import argparse, ast, warnings
import numpy as np
import pandas as pd
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


def parse_home(ht):
    if pd.isna(ht):
        return []
    try:
        return ast.literal_eval(ht)
    except Exception:
        return []


SWAP_BUCKETS = ["STR_swap_within", "STR_swap_cross"]
PUSH_BUCKETS = ["STR_push_within", "STR_push_cross", "Gaussian_push"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--out", default="figures/fig8_fp_anatomy_scatter.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)

    in_circ = df[df["in_base_circuit"] == True]
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def is_fp(row):
        thr = thresholds.get(row["combo"])
        return (thr is not None and pd.notna(row["abs_NIE"])
                and row["abs_NIE"] >= thr
                and row["in_base_circuit"] == False)

    df["is_fp"] = df.apply(is_fp, axis=1)
    fps = df[df["is_fp"]].copy()
    fps["home"] = fps["home_tasks"].apply(parse_home)
    fps["subset"] = fps["home"].apply(lambda h: "borrowed" if len(h) > 0 else "epiotic")

    fps["kind"] = fps["bucket"].apply(
        lambda b: "swap" if "swap" in b else "push")

    curv_col = {"swap": "gap_frac", "push": "local_gap_frac"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    along_cols = ["along_target", "along_nonbase_max"]
    along_labels = ["$\\mathrm{along}_t(\\delta)$",
                    "$\\mathrm{along}_s(\\delta)$"]

    colors = {"borrowed": "#E8790C", "epiotic": "#1B7F8E"}
    markers = {"borrowed": "o", "epiotic": "D"}

    for row_i, kind in enumerate(["swap", "push"]):
        sub = fps[fps["kind"] == kind].copy()
        curv = curv_col[kind]
        curv_label = ("downstream curv." if kind == "swap"
                      else "local curvature")

        for col_i, (acol, alab) in enumerate(zip(along_cols, along_labels)):
            ax = axes[row_i, col_i]
            clean = sub.dropna(subset=[acol, curv])
            clean = clean[np.isfinite(clean[acol]) & np.isfinite(clean[curv])]

            for subset in ["epiotic", "borrowed"]:
                d = clean[clean["subset"] == subset]
                ax.scatter(d[acol], d[curv],
                           c=colors[subset], marker=markers[subset],
                           s=10, alpha=0.25, edgecolors="none",
                           rasterized=True, label=subset, zorder=2)
                med_x = d[acol].median()
                med_y = d[curv].median()
                ax.plot(med_x, med_y, marker=markers[subset],
                        color=colors[subset], markersize=12,
                        markeredgecolor="black", markeredgewidth=1.2,
                        zorder=10)

            ax.axhline(0, color="grey", ls="-", lw=0.4, zorder=0)
            ax.axvline(0, color="grey", ls="-", lw=0.4, zorder=0)
            if row_i == 1:
                ax.set_xlabel(alab, fontsize=10)
            if row_i == 0:
                ax.set_title(alab, fontsize=11, pad=6)
            if col_i == 0:
                ax.set_ylabel(curv_label, fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[0, 0].annotate("swap", xy=(0, 0.5), xytext=(-0.22, 0.5),
                         xycoords="axes fraction", textcoords="axes fraction",
                         fontsize=12, fontweight="bold", rotation=90,
                         va="center", ha="center")
    axes[1, 0].annotate("push", xy=(0, 0.5), xytext=(-0.22, 0.5),
                         xycoords="axes fraction", textcoords="axes fraction",
                         fontsize=12, fontweight="bold", rotation=90,
                         va="center", ha="center")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors["borrowed"], markersize=8,
                   label="borrowed"),
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=colors["epiotic"], markersize=7,
                   label="epiotic"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("FP anatomy: borrowed vs epiotic dormancy",
                 fontsize=12, y=1.03)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
