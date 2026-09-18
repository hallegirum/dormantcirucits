"""Figure: TP geometric landscape.

(a) d_base vs d_K hexbin — where does the endpoint sit between base and K?
(b) d_base vs along_base hexbin — distance from base vs alignment to base
(c) d_base/d_K ratio histogram
"""
import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

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

d_base = []; d_K = []; along_base = []
for r in tps:
    db = r.get("dist_base")
    dk = r.get("dist_K")
    if db is None or dk is None:
        continue
    abt = r.get("along_by_task") or {}
    base_tid = identify_base(r)
    if base_tid is None:
        continue
    ab = abt.get(base_tid, 0)
    d_base.append(db)
    d_K.append(dk)
    along_base.append(ab)

d_base = np.array(d_base)
d_K = np.array(d_K)
along_base = np.array(along_base)
ratio = d_base / (d_K + 1e-9)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

# (a) d_base vs d_K
ax = axes[0]
ax.hexbin(np.log10(d_base + 0.1), np.log10(d_K + 0.1), gridsize=40,
          cmap="YlOrRd", mincnt=1, norm=LogNorm())
ax.plot([-1, 3], [-1, 3], 'k--', lw=0.8, alpha=0.5, label=r"$d_\mathrm{base} = d_K$")
ax.axvline(0, color="#bbb", ls=":", lw=0.6)
ax.axhline(0, color="#bbb", ls=":", lw=0.6)
ax.set_xlabel(r"$\log_{10}(d_\mathrm{base})$", fontsize=10)
ax.set_ylabel(r"$\log_{10}(d_K)$", fontsize=10)
ax.set_title("(a)  TP endpoints: base vs K distance", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.text(0.3, 1.8, "near base,\nfar from K", fontsize=8, color="#666", ha="center")
ax.text(1.8, 0.3, "near K,\nfar from base", fontsize=8, color="#666", ha="center")

# (b) d_base vs along_base
ax = axes[1]
ax.hexbin(np.log10(d_base + 0.1), along_base, gridsize=40,
          cmap="YlOrRd", mincnt=1, norm=LogNorm())
ax.axvline(0, color="#bbb", ls=":", lw=0.6)
ax.axhline(0, color="#bbb", ls="-", lw=0.6)
ax.set_xlabel(r"$\log_{10}(d_\mathrm{base})$", fontsize=10)
ax.set_ylabel(r"$\mathrm{along}_\mathrm{base}(\delta)$", fontsize=10)
rho, _ = spearmanr(along_base, d_base)
ax.set_title(f"(b)  Distance from base vs alignment to base\n"
             rf"Spearman $\rho$ = {rho:.2f}", fontsize=10)

# (c) ratio distribution
ax = axes[2]
log_ratio = np.clip(np.log10(ratio + 1e-9), -2, 3)
ax.hist(log_ratio, bins=60, color="#2a78d6", alpha=0.7, edgecolor="none")
ax.axvline(0, color="black", ls="--", lw=1, label=r"$d_\mathrm{base} = d_K$")
ax.set_xlabel(r"$\log_{10}(d_\mathrm{base} / d_K)$", fontsize=10)
ax.set_ylabel("count", fontsize=10)
ax.set_title(r"(c)  Ratio $d_\mathrm{base}/d_K$", fontsize=10)
ax.legend(fontsize=8)

lt = (ratio < 1).sum()
gt = (ratio >= 1).sum()
ax.text(-1.0, ax.get_ylim()[1] * 0.8,
        f"closer to base\n{lt} ({100 * lt / len(ratio):.0f}%)",
        fontsize=8, ha="center", color="#2a78d6")
ax.text(1.5, ax.get_ylim()[1] * 0.8,
        f"closer to K\n{gt} ({100 * gt / len(ratio):.0f}%)",
        fontsize=8, ha="center", color="#eb6834")

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, "fig_tp_landscape.pdf")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved {out}")
print(f"TPs plotted: {len(d_base)}")
print(f"Spearman(along_base, d_base) = {rho:.3f}")
