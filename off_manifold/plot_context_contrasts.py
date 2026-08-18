#!/usr/bin/env python
"""Reproduce the natural-domains off-manifold CONTRAST plots from a results tree:

  (1) context_matched vs context_mismatched, on wrap / no_wrap   (label: contextual_wrapping)
  (2) cross_task vs same-task (matched+mismatched), on the GLOBAL manifold (label: global_result)

Off-manifold distance per site = the chosen --offmetric of the intervened-rep cloud vs that
site's (context) manifold, read straight off disk (features.safetensors + the stored PCA
rotation — no model rerun). Default `residual_norm` = orthogonal residual OFF the top-d PCA
tangent, divided by the natural cloud's radius:
  * calibrated & bounded (unlike `projection`, which explodes ~1e6 on the standardized dead dims),
  * and it SEES structured OOD (unlike `mahalanobis`, which is within-tangent and blind to it).

Usage (from off_manifold/, causalab venv active):
  python plot_context_contrasts.py --results <SEED_DIR>
  python plot_context_contrasts.py --results <SEED_DIR> --offmetric mahalanobis   # for comparison
where SEED_DIR is a writer output tree, e.g.
  results_v2/dormancy/results/wrapped_results/results_v2/seed_0
Writes wrap_ctx_<offmetric>.png and wrap_global_<offmetric>.png into --out.
"""
import argparse
import glob
import importlib.util
import math
import os
import sys

import numpy as np
from safetensors.torch import load_file


def _load_reader(results_dir):
    """Import off-distribution-analyses.py with OFFDIST_RESULTS_DIR pointed at this tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    os.environ["OFFDIST_RESULTS_DIR"] = results_dir  # must precede results_layout import
    sys.path.insert(0, here)
    spec = importlib.util.spec_from_file_location(
        "oda", os.path.join(here, "off-distribution-analyses.py"))
    oda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oda)
    return oda


def _collect(oda, results_dir, offmetric, modes, d_cloud):
    """{(label, category, context, mode): [per-site off]} over every reps dir in the tree."""
    data = {}
    for d in sorted(glob.glob(os.path.join(results_dir, "intervened_reps", "*"))):
        parts = os.path.basename(d).split("__")
        if len(parts) != 3:  # ranking-only dir (no context suffix / no reps)
            continue
        label, cat, ctx = parts
        for mode in modes:
            rp = os.path.join(d, f"{mode}_intervened_reps.safetensors")
            if not os.path.exists(rp):
                continue
            reps = {k: v.float() for k, v in load_file(rp).items()}
            offs = []
            for sid, R in reps.items():
                try:
                    offs.append(oda._offmetrics(sid, ctx, R, offmetric, d_cloud, 64)["off"])
                except Exception:
                    pass
            offs = [o for o in offs if not (isinstance(o, float) and math.isnan(o))]
            if offs:
                data[(label, cat, ctx, mode)] = offs
                print(f"{label:20s} {cat:40s} {ctx:8s} {mode:9s} {offmetric} "
                      f"mean={np.mean(offs):.3f} n={len(offs)}", flush=True)
    return data


def _bars(ax, groups, ylabel, title):
    import matplotlib.pyplot as plt
    xs, cols = np.arange(len(groups)), ["#4c78a8", "#e45756", "#54a24b", "#b279a2"]
    for i, (lab, vals) in enumerate(groups):
        ax.bar(xs[i], np.mean(vals), yerr=np.std(vals) / max(len(vals) ** 0.5, 1),
               color=cols[i % 4], alpha=.75, capsize=4, zorder=2)
        ax.scatter(np.full(len(vals), xs[i]) + np.random.uniform(-.12, .12, len(vals)),
                   vals, c="k", s=14, zorder=3, alpha=.6)
    ax.set_xticks(xs); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.25); ax.set_ylabel(ylabel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="a writer output tree (seed dir) with intervened_reps/ + activation_manifolds/")
    ap.add_argument("--offmetric", default="residual_norm",
                    choices=["residual_norm", "residual", "mahalanobis", "projection"])
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian"])
    ap.add_argument("--d-cloud", type=int, default=8)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    oda = _load_reader(args.results)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = _collect(oda, args.results, args.offmetric, args.modes, args.d_cloud)
    ylabel = {"residual_norm": "orthogonal residual / cloud radius",
              "residual": "orthogonal residual", "mahalanobis": "Mahalanobis",
              "projection": "spline projection"}[args.offmetric]
    os.makedirs(args.out, exist_ok=True)

    # (1) contextual_wrapping — matched vs mismatched, on no_wrap & wrap
    fig, axes = plt.subplots(1, len(args.modes), figsize=(5.5 * len(args.modes), 4.6), squeeze=False)
    for ax, mode in zip(axes[0], args.modes):
        g = []
        for ctx in ["no_wrap", "wrap"]:
            for cat, nm in [("('context_matched',)", "matched"),
                            ("('context_mismatched',)", "mismatched")]:
                v = data.get(("contextual_wrapping", cat, ctx, mode))
                if v:
                    g.append((f"{nm}\n/{ctx}", v))
        if g:
            _bars(ax, g, ylabel, f"contextual_wrapping — {mode}")
    fig.suptitle(f"context_matched vs context_mismatched on wrap / no_wrap  ({args.offmetric})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = os.path.join(args.out, f"wrap_ctx_{args.offmetric}.png")
    fig.savefig(p1, dpi=140, bbox_inches="tight")

    # (2) global_result — cross_task vs same-task, on the global manifold
    fig, axes = plt.subplots(1, len(args.modes), figsize=(4.5 * len(args.modes), 4.6), squeeze=False)
    for ax, mode in zip(axes[0], args.modes):
        g = []
        for cat, nm in [("('context_matched', 'context_mismatched')", "same-task"),
                        ("('cross_task',)", "cross_task")]:
            v = data.get(("global_result", cat, "global", mode))
            if v:
                g.append((nm, v))
        if g:
            _bars(ax, g, ylabel, f"global_result — {mode}")
    fig.suptitle(f"cross_task vs same-task on the global manifold  ({args.offmetric})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p2 = os.path.join(args.out, f"wrap_global_{args.offmetric}.png")
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    print(f"saved:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
