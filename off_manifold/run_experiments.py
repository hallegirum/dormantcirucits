"""
Run the off-distribution analysis on a completed results directory (e.g. the
smoke run pulled back from the pod) — CPU-only, no model needed.

    python run_experiments.py --results-dir results_smoke_pod

Points the results layout at --results-dir (via OFFDIST_RESULTS_DIR, which
results_layout reads at import), then calls divergence_metrics (activation side:
projection / EMD / Bhattacharyya vs the fitted manifolds) and behaviour_divergence
(the paper's E_BC vs M_y). Writes divergence_activation.json / divergence_behaviour.json
into the results dir and prints a per-(label, category) STR/Gaussian/DAS summary.
"""

import argparse
import importlib
import os
import re


_CLOUD_KEYS = ("projection_distance", "emd_distance", "bhattacharyya_distance")


def _layer_of(site_id):
    m = re.search(r"Layer-(\d+)", site_id)
    return int(m.group(1)) if m else None


def _plot_by_layer(results, modes, out_path, metric="projection_distance"):
    """One subplot per (label, category): metric vs layer, a line per mode.
    Each site is a (layer, position), so per-site == per-layer here."""
    import matplotlib
    matplotlib.use("Agg")  # headless — save to file
    import matplotlib.pyplot as plt

    keys = sorted({(lab, cat) for (lab, cat, _m) in results})
    ncols = min(3, len(keys)) or 1
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = [a for row in axes for a in row]

    for ax, (lab, cat) in zip(flat_axes, keys):
        for m in modes:
            per_site = results.get((lab, cat, m))
            if not per_site:
                continue
            pts = []
            for site_id, view in per_site.items():
                metrics = view.get("global") or view.get("union")
                layer = _layer_of(site_id)
                if metrics and layer is not None and metric in metrics:
                    pts.append((layer, metrics[metric]))
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(f"{lab} / {cat}", fontsize=9)
        ax.set_xlabel("layer")
        ax.set_ylabel(metric.replace("_", " "))
        ax.legend(fontsize=8)
    for ax in flat_axes[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved plot ->", out_path)


def _mean_over_sites(per_site):
    """Average each metric across sites. Handles both the global-label shape
    ({site: {"global": {...}}}) and the contextual shape ({site: {"union": {...}}})."""
    rows = []
    for site_id, view in per_site.items():
        metrics = view.get("global") or view.get("union")
        if metrics:
            rows.append(metrics)
    if not rows:
        return None
    return {k: sum(r[k] for r in rows) / len(rows) for k in _CLOUD_KEYS if k in rows[0]}


def _print_activation(results, modes):
    print("\n===== ACTIVATION-SIDE (mean over sites) =====")
    # group by (label, category), one row per mode
    keys = sorted({(lab, cat) for (lab, cat, _m) in results})
    for lab, cat in keys:
        print(f"\n{lab}  /  {cat}")
        print(f"  {'mode':10s} {'projection':>12s} {'emd':>12s} {'bhattacharyya':>14s}  (n_sites)")
        for m in modes:
            per_site = results.get((lab, cat, m))
            if not per_site:
                continue
            agg = _mean_over_sites(per_site)
            if agg is None:
                print(f"  {m:10s} {'(no manifolds)':>40s}")
                continue
            print(f"  {m:10s} {agg['projection_distance']:12.3f} {agg['emd_distance']:12.3f} "
                  f"{agg['bhattacharyya_distance']:14.3f}  ({len(per_site)})")


def _print_behaviour(results, modes):
    print("\n===== BEHAVIOUR-SIDE  E_BC = mean d_B(patched output, M_y)  (mean over sites) =====")
    keys = sorted({(lab, cat) for (lab, cat, _m) in results})
    for lab, cat in keys:
        row = f"{lab} / {cat}:  "
        for m in modes:
            per_site = results.get((lab, cat, m))
            if per_site:
                vals = list(per_site.values())
                row += f"{m}={sum(vals)/len(vals):.3f}  "
        print("  " + row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_smoke_pod",
                    help="dir holding activation_manifolds/, intervened_reps/, output_manifolds/")
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian", "DAS"])
    args = ap.parse_args()

    # results_layout reads OFFDIST_RESULTS_DIR at import, so set it BEFORE importing.
    os.environ["OFFDIST_RESULTS_DIR"] = args.results_dir

    from activation_manifold import DEFAULT_COMBINATIONS
    oda = importlib.import_module("off-distribution-analyses")

    act = oda.divergence_metrics(
        DEFAULT_COMBINATIONS, modes=args.modes,
        save_to=os.path.join(args.results_dir, "divergence_activation.json"),
    )
    _print_activation(act, args.modes)
    _plot_by_layer(act, args.modes,
                   os.path.join(args.results_dir, "projection_by_layer.png"))

    try:
        beh = oda.behaviour_divergence(
            DEFAULT_COMBINATIONS, modes=args.modes,
            save_to=os.path.join(args.results_dir, "divergence_behaviour.json"),
        )
        _print_behaviour(beh, args.modes)
    except FileNotFoundError as e:
        print("\n[behaviour side skipped] ", e)

    print(f"\nWrote divergence_activation.json"
          f"{' + divergence_behaviour.json' if 'beh' in dir() else ''} to {args.results_dir}/")


if __name__ == "__main__":
    main()
