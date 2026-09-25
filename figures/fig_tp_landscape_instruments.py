"""Figure: FP + TP endpoint landscape colored by instrument intensity.

Two panels: (a) colored by local_gap_frac, (b) colored by ripple_xint_in_orth.
TPs joined by (combo, seed, site); FPs joined by (combo, seed) median across sites.
"""
import csv, json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from collections import defaultdict

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..")
INT_FILE = os.path.join(DATA_ROOT, "interaction_analysis_ripplefull", "rows.csv")
TP_EP_FILE = os.path.join(DATA_ROOT, "endpoint_density_tp_v2.jsonl")
FP_EP_FILE = os.path.join(DATA_ROOT, "endpoint_density_v2.jsonl")
if not os.path.isfile(INT_FILE):
    BASE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis"
    INT_FILE = os.path.join(BASE, "interaction_analysis_ripplefull", "rows.csv")
    TP_EP_FILE = os.path.join(BASE, "endpoint_density_tp_v2.jsonl")
    FP_EP_FILE = os.path.join(BASE, "endpoint_density_v2.jsonl")

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"

METRICS = ["local_gap_frac", "ripple_xint_in_orth"]


def load_tp_endpoint():
    out = {}
    with open(TP_EP_FILE) as f:
        for line in f:
            r = json.loads(line)
            key = (r["combo"], str(r["seed"]), r.get("site", ""))
            if key not in out and r.get("dist_base") is not None and r.get("dist_K") is not None:
                out[key] = {"d_base": r["dist_base"], "d_K": r["dist_K"]}
    return out


def load_fp_endpoint():
    out = defaultdict(list)
    with open(FP_EP_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_fp") and r.get("dist_base") is not None and r.get("dist_K") is not None:
                out[(r["combo"], str(r["seed"]))].append(
                    {"d_base": r["dist_base"], "d_K": r["dist_K"]})
    return out


def load_interaction_metrics():
    rows = list(csv.DictReader(open(INT_FILE)))
    str_a1 = [r for r in rows
              if r["strategy"] == "STR"
              and r["alpha"] == "1.0"
              and r["push_site"] != "self"]

    tp_site = defaultdict(lambda: defaultdict(list))
    fp_site = defaultdict(lambda: defaultdict(list))
    for r in str_a1:
        is_circuit = r["gt_label"] == "circuit"
        key3 = (r["combo"], r["seed"], r["site"])
        key2 = (r["combo"], r["seed"])
        for m in METRICS:
            v = r.get(m, "")
            if v and v != "nan":
                if is_circuit:
                    tp_site[key3][m].append(float(v))
                else:
                    fp_site[key2][m].append(float(v))

    tp_agg = {}
    for key, metrics in tp_site.items():
        tp_agg[key] = {m: float(np.median(vals)) for m, vals in metrics.items() if vals}

    fp_agg = {}
    for key, metrics in fp_site.items():
        fp_agg[key] = {m: float(np.median(vals)) for m, vals in metrics.items() if vals}

    return tp_agg, fp_agg


def main():
    tp_ep = load_tp_endpoint()
    fp_ep = load_fp_endpoint()
    tp_int, fp_int = load_interaction_metrics()

    tp_joined = []
    for key in tp_ep:
        ep = tp_ep[key]
        inst = tp_int.get(key, {})
        if inst:
            tp_joined.append({
                "d_base": ep["d_base"], "d_K": ep["d_K"],
                **{m: inst.get(m) for m in METRICS},
            })

    fp_joined = []
    for key2, eps in fp_ep.items():
        inst = fp_int.get(key2, {})
        if inst:
            for ep in eps:
                fp_joined.append({
                    "d_base": ep["d_base"], "d_K": ep["d_K"],
                    **{m: inst.get(m) for m in METRICS},
                })

    print(f"TP joined: {len(tp_joined)}, FP joined: {len(fp_joined)}")

    panels = [
        ("local_gap_frac", "local curvature", "BuPu"),
        ("ripple_xint_in_orth", r"$\mathrm{ripple}_\mathrm{xint}$", "RdPu"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for idx, (metric, label, cmap_name) in enumerate(panels):
        ax = axes[idx]

        # Gather valid records for shared color scale
        tp_valid = [r for r in tp_joined
                    if r[metric] is not None and r["d_base"] > 0 and r["d_K"] > 0]
        fp_valid = [r for r in fp_joined
                    if r[metric] is not None and r["d_base"] > 0 and r["d_K"] > 0]

        all_c = ([r[metric] for r in tp_valid] + [r[metric] for r in fp_valid])
        vmin, vmax = np.percentile(all_c, [5, 95])
        norm = Normalize(vmin=vmin, vmax=vmax)

        # FPs first (behind)
        if fp_valid:
            fx = np.log10(np.array([r["d_base"] for r in fp_valid]) + 0.1)
            fy = np.log10(np.array([r["d_K"] for r in fp_valid]) + 0.1)
            fc = np.array([r[metric] for r in fp_valid])
            ax.scatter(fx, fy, c=fc, cmap=cmap_name, norm=norm,
                       s=6, alpha=0.35, edgecolors="none", marker="s",
                       rasterized=True)

        # TPs on top
        if tp_valid:
            tx = np.log10(np.array([r["d_base"] for r in tp_valid]) + 0.1)
            ty = np.log10(np.array([r["d_K"] for r in tp_valid]) + 0.1)
            tc = np.array([r[metric] for r in tp_valid])
            sc = ax.scatter(tx, ty, c=tc, cmap=cmap_name, norm=norm,
                            s=12, alpha=0.6, edgecolors="none", marker="o",
                            rasterized=True)
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
        ax.set_title(f"({pl})  colored by {label}\n"
                     f"(TP $n$={len(tp_valid)}, FP $n$={len(fp_valid)})",
                     fontsize=10, pad=6)

    # Shared legend for marker shapes
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
               markersize=6, label="TP"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#888",
               markersize=5, label="FP"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig_tp_landscape_instruments.pdf")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
