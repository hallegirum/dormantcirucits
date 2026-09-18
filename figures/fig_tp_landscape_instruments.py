"""Figure: TP d_base vs d_K landscape colored by instrument intensity.

Two panels: (a) colored by orth_base, (b) colored by ripple_frac.
Each is the same d_base vs d_K scatter from fig_tp_landscape but with
instrument intensity replacing density.
"""
import csv, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def load_tp_endpoint():
    out = {}
    with open(TP_EP_FILE) as f:
        for line in f:
            r = json.loads(line)
            key = (r["combo"], str(r["seed"]), r.get("site", ""))
            if key not in out and r.get("dist_base") is not None and r.get("dist_K") is not None:
                out[key] = {"d_base": r["dist_base"], "d_K": r["dist_K"]}
    return out


def load_interaction_metrics():
    """Load STR push alpha=1 rows, aggregate per (combo, seed, site)."""
    rows = list(csv.DictReader(open(INT_FILE)))
    str_a1_circuit = [r for r in rows
                      if r["strategy"] == "STR"
                      and r["alpha"] == "1.0"
                      and r["push_site"] != "self"
                      and r["gt_label"] == "circuit"]

    site_data = defaultdict(lambda: defaultdict(list))
    for r in str_a1_circuit:
        key = (r["combo"], r["seed"], r["site"])
        for m in ["orth_base", "ripple_frac"]:
            v = r.get(m, "")
            if v and v != "nan":
                site_data[key][m].append(float(v))

    agg = {}
    for key, metrics in site_data.items():
        agg[key] = {
            m: float(np.median(vals)) if vals else None
            for m, vals in metrics.items()
        }
    return agg


def main():
    tp_ep = load_tp_endpoint()
    int_agg = load_interaction_metrics()

    # Join
    joined = []
    for key in tp_ep:
        ep = tp_ep[key]
        inst = int_agg.get(key, {})
        if inst:
            joined.append({
                "d_base": ep["d_base"],
                "d_K": ep["d_K"],
                "orth_base": inst.get("orth_base"),
                "ripple_frac": inst.get("ripple_frac"),
            })

    print(f"Joined TP records: {len(joined)} / {len(tp_ep)} endpoint records")

    panels = [
        ("orth_base", r"$\mathrm{orth}_\mathrm{base}$", "YlOrRd"),
        ("ripple_frac", r"$\mathrm{ripple\_frac}$", "YlGnBu"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for idx, (metric, label, cmap_name) in enumerate(panels):
        ax = axes[idx]
        vals = [r for r in joined
                if r[metric] is not None and r["d_base"] > 0 and r["d_K"] > 0]

        x = np.log10(np.array([r["d_base"] for r in vals]) + 0.1)
        y = np.log10(np.array([r["d_K"] for r in vals]) + 0.1)
        c = np.array([r[metric] for r in vals])

        vmin, vmax = np.percentile(c, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)

        sc = ax.scatter(x, y, c=c, cmap=cmap_name, norm=norm,
                       s=8, alpha=0.5, edgecolors="none", rasterized=True)
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(label, fontsize=9)

        ax.plot([-1, 3], [-1, 3], "k--", lw=0.8, alpha=0.5)
        ax.axvline(0, color="#bbb", ls=":", lw=0.6)
        ax.axhline(0, color="#bbb", ls=":", lw=0.6)
        ax.set_xlabel(r"$\log_{10}(d_\mathrm{base})$", fontsize=10)
        ax.set_ylabel(r"$\log_{10}(d_K)$", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        pl = "ab"[idx]
        ax.set_title(f"({pl})  TP endpoints colored by {label}\n"
                     f"($n$={len(vals)})", fontsize=10, pad=6)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_tp_landscape_instruments.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
