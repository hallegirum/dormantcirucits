"""Figure: FP borrowed vs TP instrument comparison.

Scatter panels comparing instrument metrics between FP borrowed and TP.
For TPs, points are colored by d_base/d_K (geometric loyalty).

Metrics: along_nonbase_max, orth_base, local_gap_frac, ripple_frac
"""
import csv, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from collections import defaultdict

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..")
INT_FILE = os.path.join(DATA_ROOT, "interaction_analysis_ripplefull", "rows.csv")
TP_EP_FILE = os.path.join(DATA_ROOT, "endpoint_density_tp_v2.jsonl")
if not os.path.isfile(INT_FILE):
    INT_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/interaction_analysis_ripplefull/rows.csv"
    TP_EP_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/endpoint_density_tp_v2.jsonl"

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"

METRICS = [
    ("along_nonbase_max", r"$\mathrm{along}_\mathrm{nb}$"),
    ("orth_base", r"$\mathrm{orth}_\mathrm{base}$"),
    ("local_gap_frac", r"local curvature ($\mathrm{local\_gap\_frac}$)"),
    ("ripple_frac", r"$\mathrm{ripple\_frac}$"),
]


def load_interaction():
    rows = list(csv.DictReader(open(INT_FILE)))
    return [r for r in rows if r["strategy"] == "STR" and r["alpha"] == "1.0"
            and r["push_site"] != "self"]


def load_tp_endpoint():
    out = {}
    with open(TP_EP_FILE) as f:
        for line in f:
            r = json.loads(line)
            key = (r["combo"], str(r["seed"]), r.get("site", ""))
            if key not in out and r.get("dist_base") is not None and r.get("dist_K") is not None:
                out[key] = {"d_base": r["dist_base"], "d_K": r["dist_K"]}
    return out


def classify_sites(int_rows):
    """Classify each (seed, site) as borrowed, epiotic, or circuit across combos."""
    site_combos = defaultdict(lambda: defaultdict(set))
    for r in int_rows:
        key = (r["seed"], r["site"])
        site_combos[key][r["combo"]].add(r["gt_label"])

    borrowed = set()
    epiotic = set()
    for key, combo_labels in site_combos.items():
        has_circuit = any("circuit" in labels for labels in combo_labels.values())
        has_none = any("none" in labels for labels in combo_labels.values())
        if has_circuit and has_none:
            borrowed.add(key)
        elif has_none:
            epiotic.add(key)
    return borrowed, epiotic


def aggregate_metrics(int_rows, borrowed_set, tp_ep):
    """Aggregate per (combo, seed, site), return FP-borrowed and TP rows."""
    site_data = defaultdict(lambda: defaultdict(list))
    site_label = {}
    for r in int_rows:
        key = (r["combo"], r["seed"], r["site"])
        site_label[key] = r["gt_label"]
        for m, _ in METRICS:
            v = r.get(m, "")
            if v and v != "nan":
                site_data[key][m].append(float(v))

    fp_borrowed = []
    tp_rows = []
    for key, metrics in site_data.items():
        gt = site_label[key]
        agg = {}
        for m, _ in METRICS:
            vals = metrics.get(m, [])
            agg[m] = float(np.median(vals)) if vals else None

        seed_site = (key[1], key[2])
        if gt == "none" and seed_site in borrowed_set:
            fp_borrowed.append(agg)
        elif gt == "circuit":
            ep_key = (key[0], key[1], key[2])
            ep = tp_ep.get(ep_key, {})
            agg["d_base"] = ep.get("d_base")
            agg["d_K"] = ep.get("d_K")
            tp_rows.append(agg)

    return fp_borrowed, tp_rows


def main():
    int_rows = load_interaction()
    tp_ep = load_tp_endpoint()
    borrowed_set, epiotic_set = classify_sites(int_rows)

    print(f"Borrowed (seed,site): {len(borrowed_set)}")
    print(f"Epiotic (seed,site): {len(epiotic_set)}")

    fp_borrowed, tp_rows = aggregate_metrics(int_rows, borrowed_set, tp_ep)
    print(f"FP borrowed aggregated rows: {len(fp_borrowed)}")
    print(f"TP aggregated rows: {len(tp_rows)}")

    tp_with_ratio = [r for r in tp_rows
                     if r.get("d_base") is not None and r.get("d_K") is not None
                     and r["d_K"] > 1e-9]
    tp_no_ratio = [r for r in tp_rows
                   if r.get("d_base") is None or r.get("d_K") is None
                   or (r.get("d_K") or 0) <= 1e-9]
    print(f"TPs with d_base/d_K: {len(tp_with_ratio)}, without: {len(tp_no_ratio)}")

    # 2×2 panels: each metric pair as x vs y
    # Better: 4 panels, one per metric, showing distributions as strip/jitter
    # Actually: pairwise scatter of all 4 metrics colored by class

    # Simpler approach: 2×2, each panel = one metric
    # x-axis: d_base/d_K (for TPs), jittered position (for FPs)
    # y-axis: metric value

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=-2, vmax=2)

    for idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes_flat[idx]

        # FP borrowed: place at x = -3.5 with jitter
        fp_vals = [r[metric_key] for r in fp_borrowed if r[metric_key] is not None]
        if fp_vals:
            jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(fp_vals))
            ax.scatter(-3.5 + jitter, fp_vals, c="#555", s=12, alpha=0.5,
                       edgecolors="none", zorder=3, label=f"FP borrowed ({len(fp_vals)})")

        # TP: x = log10(d_base/d_K), colored by ratio
        tp_x = []
        tp_y = []
        tp_c = []
        for r in tp_with_ratio:
            v = r[metric_key]
            if v is None:
                continue
            log_ratio = np.log10(r["d_base"] / r["d_K"])
            tp_x.append(log_ratio)
            tp_y.append(v)
            tp_c.append(log_ratio)

        if tp_x:
            sc = ax.scatter(tp_x, tp_y, c=tp_c, cmap=cmap, norm=norm,
                           s=12, alpha=0.5, edgecolors="none", zorder=2)

        # TPs without ratio at x=-3.0
        tp_nr_vals = [r[metric_key] for r in tp_no_ratio if r[metric_key] is not None]
        if tp_nr_vals:
            jitter = np.random.default_rng(43).uniform(-0.15, 0.15, len(tp_nr_vals))
            ax.scatter(-3.0 + jitter, tp_nr_vals, c="#aaa", s=8, alpha=0.3,
                       edgecolors="none", zorder=1)

        ax.axvline(0, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(-2.8, color="#ccc", ls=":", lw=0.6)

        ax.set_ylabel(metric_label, fontsize=10)
        if idx >= 2:
            ax.set_xlabel(r"$\log_{10}(d_\mathrm{base}/d_K)$"
                          "   (FP at left)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        panel_letter = "abcd"[idx]
        ax.set_title(f"({panel_letter})  {metric_label}", fontsize=10, pad=6)

        # Median lines
        if fp_vals:
            med_fp = np.median(fp_vals)
            ax.axhline(med_fp, xmin=0, xmax=0.1, color="#555", ls="-", lw=2, alpha=0.8)
        if tp_y:
            med_tp = np.median(tp_y)
            ax.axhline(med_tp, xmin=0.15, xmax=1, color="#c44", ls="-", lw=1.5, alpha=0.6)

    # Colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label(r"$\log_{10}(d_\mathrm{base}/d_K)$"
                 "\ncloser to base ← → closer to K", fontsize=9)

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
               markersize=7, label="FP borrowed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44",
               markersize=7, label="TP (color = geometric loyalty)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, 1.04))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_fp_tp_instruments.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
