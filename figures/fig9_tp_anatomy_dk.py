"""Fig 9 — TP anatomy: in-circuit components with d_k coloring.

2x2 panels: rows = swap / push, cols = along_t / along_nonbase_max.
Y = curvature (local for push, downstream for swap).
Color = d_k (whitened endpoint distance to the most-aligned task K).
  d_k ~ 1 → endpoint in K's data cloud
  d_k >> 1 → no-man's land (pointing toward K but not reaching it)

d_k is aggregated to per-component medians (across interventions within each
regime) from endpoint_density_tp.jsonl, then merged with rows.csv.

Usage:
    python figures/fig9_tp_anatomy_dk.py [--csv PATH] [--tp-jsonl PATH] [--out PATH]
"""
import argparse, warnings
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


def bucket_to_regime(bucket):
    if "swap_within" in bucket:
        return "swap_within"
    if "swap_cross" in bucket:
        return "swap_cross"
    if bucket == "Gaussian_push":
        return "Gaussian_push"
    if "push_within" in bucket:
        return "STR_push_within"
    if "push_cross" in bucket:
        return "STR_push_cross"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="interaction_analysis_allcombos_fullgap_trio/rows.csv")
    ap.add_argument("--tp-jsonl", default="endpoint_density_tp.jsonl")
    ap.add_argument("--out", default="figures/fig9_tp_anatomy_dk.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["bucket"] = df.apply(_strat_bucket, axis=1)
    df["regime"] = df["bucket"].apply(bucket_to_regime)

    # median d_k per (combo, seed, regime, site)
    tp_recs = [json.loads(l) for l in open(args.tp_jsonl)]
    tp_df = pd.DataFrame(tp_recs)
    dk_agg = (tp_df.groupby(["combo", "seed", "regime", "site"])["dist_K"]
              .median().reset_index().rename(columns={"dist_K": "dk_med"}))

    # filter rows.csv to in-circuit, above threshold (TPs)
    in_circ = df[df["in_base_circuit"] == True].copy()
    thresholds = in_circ.groupby("combo")["abs_NIE"].median().to_dict()

    def is_tp(row):
        thr = thresholds.get(row["combo"])
        return (thr is not None and pd.notna(row["abs_NIE"])
                and row["abs_NIE"] >= thr)

    tps = in_circ[in_circ.apply(is_tp, axis=1)].copy()

    # aggregate rows.csv TPs to per-component medians
    along_curv_cols = ["along_target", "along_nonbase_max",
                       "local_gap_frac", "gap_frac"]
    tp_agg = (tps.groupby(["combo", "seed", "regime", "site"])[along_curv_cols]
              .median().reset_index())

    merged = tp_agg.merge(dk_agg, on=["combo", "seed", "regime", "site"],
                          how="inner")
    merged["kind"] = merged["regime"].apply(
        lambda r: "swap" if "swap" in r else "push")

    print(f"Merged TP+d_k records: {len(merged)}")
    print(f"  swap: {(merged['kind']=='swap').sum()}, push: {(merged['kind']=='push').sum()}")
    print(f"  dk_med: median={merged['dk_med'].median():.2f}, "
          f"min={merged['dk_med'].min():.2f}, max={merged['dk_med'].max():.2f}")

    curv_col = {"swap": "gap_frac", "push": "local_gap_frac"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    along_cols = ["along_target", "along_nonbase_max"]
    along_labels = ["$\\mathrm{along}_t(\\delta)$",
                    "$\\mathrm{along}_s(\\delta)$"]

    dk_clip = merged["dk_med"].quantile(0.98)
    vmin, vmax = 0.5, max(dk_clip, 5)

    for row_i, kind in enumerate(["swap", "push"]):
        sub = merged[merged["kind"] == kind].copy()
        curv = curv_col[kind]
        curv_label = ("downstream curv." if kind == "swap"
                      else "local curvature")

        for col_i, (acol, alab) in enumerate(zip(along_cols, along_labels)):
            ax = axes[row_i, col_i]
            clean = sub.dropna(subset=[acol, curv, "dk_med"])
            clean = clean[np.isfinite(clean[acol]) & np.isfinite(clean[curv])]

            sc = ax.scatter(clean[acol], clean[curv],
                            c=clean["dk_med"].clip(upper=dk_clip),
                            cmap="RdYlGn_r",
                            norm=LogNorm(vmin=vmin, vmax=vmax),
                            s=18, alpha=0.6, edgecolors="0.3",
                            linewidths=0.3, rasterized=True, zorder=2)

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

    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("$d_K$ (whitened endpoint distance)", fontsize=10)

    fig.suptitle("TP anatomy: in-circuit components colored by endpoint distance $d_K$",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
