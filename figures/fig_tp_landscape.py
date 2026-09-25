"""Figure: TP geometric landscape.

(a) d_base vs d_K hexbin — where does the endpoint sit between base and K?
(b) d_base/d_K ratio histogram
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..")
TP_FILE = os.path.join(DATA_ROOT, "endpoint_density_tp_v2.jsonl")
if not os.path.isfile(TP_FILE):
    TP_FILE = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/endpoint_density_tp_v2.jsonl"

OUT_DIR = "/Users/halleluiahgirum/Documents/Thesis/causalab/dormant-circuit-analysis/figures"


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            out.append(json.loads(line))
    return out


def identify_base(r):
    dpt = r.get("dist_per_task") or {}
    db = r.get("dist_base")
    if db is None or not dpt:
        return None
    for tid, dv in dpt.items():
        if abs(dv - db) < 0.001:
            return str(tid)
    return None


tps = load(TP_FILE)

d_base = []; d_K = []
for r in tps:
    db = r.get("dist_base")
    dk = r.get("dist_K")
    if db is None or dk is None:
        continue
    d_base.append(db)
    d_K.append(dk)

d_base = np.array(d_base)
d_K = np.array(d_K)
ratio = d_base / (d_K + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

# (a) d_base vs d_K
ax = axes[0]
hb = ax.hexbin(np.log10(d_base + 0.1), np.log10(d_K + 0.1), gridsize=35,
               cmap="BuPu", mincnt=1, norm=LogNorm(), edgecolors="face",
               linewidths=0.2)
fig.colorbar(hb, ax=ax, shrink=0.75, label="count")
ax.plot([-1, 3], [-1, 3], 'k--', lw=1.0, alpha=0.6, label=r"$d_\mathrm{base} = d_K$")
ax.axvline(0, color="#bbb", ls=":", lw=0.6)
ax.axhline(0, color="#bbb", ls=":", lw=0.6)
ax.set_xlabel(r"$\log_{10}(d_\mathrm{base})$", fontsize=10)
ax.set_ylabel(r"$\log_{10}(d_K)$", fontsize=10)
ax.set_title(f"(a)  TP endpoints: base vs $K$ distance ($n$={len(d_base)})",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.text(0.25, 1.7, "closer to\nbase task", fontsize=8, color="#555",
        ha="center", style="italic")
ax.text(1.7, 0.2, "closer to\ntask $K$", fontsize=8, color="#555",
        ha="center", style="italic")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# (b) ratio distribution
ax = axes[1]
log_ratio = np.clip(np.log10(ratio + 1e-9), -2, 3)
ax.hist(log_ratio, bins=60, color="#7b4ea3", alpha=0.75, edgecolor="white",
        linewidth=0.3)
ax.axvline(0, color="black", ls="--", lw=1, label=r"$d_\mathrm{base} = d_K$")
ax.set_xlabel(r"$\log_{10}(d_\mathrm{base} / d_K)$", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.set_title(r"(b)  Ratio $d_\mathrm{base}/d_K$", fontsize=10)
ax.legend(fontsize=8)

lt = (ratio < 1).sum()
gt = (ratio >= 1).sum()
ax.text(-1.0, ax.get_ylim()[1] * 0.85,
        f"closer to base\n{lt} ({100 * lt / len(ratio):.0f}%)",
        fontsize=9, ha="center", color="#7b4ea3", fontweight="bold")
ax.text(1.5, ax.get_ylim()[1] * 0.85,
        f"closer to $K$\n{gt} ({100 * gt / len(ratio):.0f}%)",
        fontsize=9, ha="center", color="#eb6834", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "fig_tp_landscape.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved {out}")
print(f"TPs plotted: {len(d_base)}")
