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
import json
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

    keys = sorted({(lab, cat, ctx) for (lab, cat, ctx, _m) in results})
    ncols = min(3, len(keys)) or 1
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), squeeze=False)
    flat_axes = [a for row in axes for a in row]

    for ax, (lab, cat, ctx) in zip(flat_axes, keys):
        for m in modes:
            per_site = results.get((lab, cat, ctx, m))
            if not per_site:
                continue
            pts = []
            for site_id, view in per_site.items():
                metrics = view.get("respective") or view.get("global") or view.get("union")
                layer = _layer_of(site_id)
                if metrics and layer is not None and metric in metrics:
                    pts.append((layer, metrics[metric]))
            pts.sort()
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=m)
        ax.set_title(f"{lab} / {cat}\n{ctx}", fontsize=8)
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
        metrics = view.get("respective") or view.get("global") or view.get("union")
        if metrics:
            rows.append(metrics)
    if not rows:
        return None
    return {k: sum(r[k] for r in rows) / len(rows) for k in _CLOUD_KEYS if k in rows[0]}


def _print_activation(results, modes):
    print("\n===== ACTIVATION-SIDE (mean over sites) =====")
    # group by (label, category), one row per mode
    keys = sorted({(lab, cat, ctx) for (lab, cat, ctx, _m) in results})
    for lab, cat, ctx in keys:
        print(f"\n{lab}  /  {cat}  /  context={ctx}")
        print(f"  {'mode':10s} {'projection':>12s} {'emd':>12s} {'bhattacharyya':>14s}  (n_sites)")
        for m in modes:
            per_site = results.get((lab, cat, ctx, m))
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
    keys = sorted({(lab, cat, ctx) for (lab, cat, ctx, _m) in results})
    for lab, cat, ctx in keys:
        row = f"{lab} / {cat} / {ctx}:  "
        for m in modes:
            per_site = results.get((lab, cat, ctx, m))
            if per_site:
                vals = list(per_site.values())
                row += f"{m}={sum(vals)/len(vals):.3f}  "
        print("  " + row)


def _site_value(site_entry, side, view, metric):
    """Pull the scalar to aggregate from one site's entry, for either side.
    activation: entry is {view: {metric: val}}.  behaviour: entry is a bare float (E_BC)."""
    if side == "behaviour":
        return float(site_entry)
    v = site_entry.get(view)
    return v[metric] if (v is not None and metric in v) else None


def aggregate_seeds(parent_dir, side="activation", view="respective", metric="projection_distance"):
    """Mean +/- SE across seeds, per (label, category, context, mode).

    side="activation" -> reads seed_*/divergence_activation.json, pulls
        [view][metric] per site (view in respective/union/gap/global).
    side="behaviour"  -> reads seed_*/divergence_behaviour.json, whose per-site
        values are bare floats (E_BC); view/metric are ignored.

    Reduces to mean-over-sites within a seed, then reports mean and standard error
    (std / sqrt(n_seeds)) across seeds. Writes {parent}/aggregate_{side}_*.json."""
    import ast
    import glob
    import math

    fname = "divergence_behaviour.json" if side == "behaviour" else "divergence_activation.json"
    seed_files = sorted(glob.glob(os.path.join(parent_dir, "seed_*", fname)))
    if not seed_files:
        print(f"[aggregate] no seed_*/{fname} under {parent_dir} — run run_experiments per seed first")
        return {}
    print(f"[aggregate] {side}: {len(seed_files)} seeds: {[os.path.basename(os.path.dirname(f)) for f in seed_files]}")

    # key -> list of per-seed (mean over sites) values
    per_key = {}
    for f in seed_files:
        d = json.load(open(f))
        for kstr, per_site in d.items():
            key = ast.literal_eval(kstr)  # (label, category, context, mode)
            vals = [x for x in (_site_value(e, side, view, metric) for e in per_site.values()) if x is not None]
            if vals:
                per_key.setdefault(key, []).append(sum(vals) / len(vals))

    out = {}
    tag = "E_BC" if side == "behaviour" else f"{view}/{metric}"
    print(f"\n===== AGGREGATE  {side}  {tag}  (mean +/- SE over seeds) =====")
    print(f"  {'label':20s}{'category':22s}{'context':9s}{'mode':10s}{'mean':>12s}{'SE':>11s}{'seeds':>6s}")
    for key in sorted(per_key, key=lambda k: tuple(map(str, k))):
        xs = per_key[key]
        mean = sum(xs) / len(xs)
        se = (math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
        lab, cat, ctx, mode = (list(key) + [None] * 4)[:4]
        print(f"  {str(lab):20s}{str(cat):22s}{str(ctx):9s}{str(mode):10s}{mean:12.3f}{se:11.3f}{len(xs):>6d}")
        out[str(key)] = {"mean": mean, "se": se, "n_seeds": len(xs), "per_seed": xs}

    suffix = "behaviour_EBC" if side == "behaviour" else f"activation_{view}_{metric}"
    dst = os.path.join(parent_dir, f"aggregate_{suffix}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", dst)
    return out


def discover_combinations(results_dir, default_combos):
    """Build the combinations dict from what's ACTUALLY on disk.

    The run on the cluster may have used a different `_CATEGORIES` than the local
    activation_manifold.py (or been interrupted part-way), so iterating the local
    DEFAULT_COMBINATIONS demands dirs that don't exist. Instead, scan
    intervened_reps/ for `{label}__{category}` dirs and analyse exactly those,
    keeping each label's variable/m_vars from the local combos (needed to pick the
    global-vs-union branch). Category is kept as the on-disk STRING, which
    label_category_dir reproduces verbatim."""
    import glob
    found = {}
    for d in sorted(glob.glob(os.path.join(results_dir, "intervened_reps", "*"))):
        base = os.path.basename(d)
        if "__" not in base or not os.path.isdir(d):
            continue
        # find_path writes {method}_top_sites.json into these dirs UP FRONT for every
        # category, so a dir existing proves nothing — require an actual reps file.
        if not glob.glob(os.path.join(d, "*_intervened_reps.safetensors")):
            continue
        label, cat = base.split("__")[0], base.split("__")[1]
        found.setdefault(label, set()).add(cat)   # context dirs repeat the category — dedup
    combos = {}
    for label, cats in found.items():
        proto = default_combos.get(label, {"variable": "result", "cf_vars": [], "m_vars": []})
        combos[label] = {**proto, "categories": sorted(cats)}
    return combos


def manifold_health(results_dir):
    """Quantitative manifold validity, from saved artifacts only (no model).

    All class-geometry metrics are computed in the RAW PCA feature space (not the
    standardized space) so near-zero-variance "dead" dims can't blow them up — the
    conditioning is reported separately as n_dead.

      explained  : 1 - recon_mse            fraction of variance the 1-D manifold captures (tier A)
      silhouette : sklearn silhouette        class separability, -1..1 (tier B)
      separab.   : between/within            class-centroid spacing / within-class spread (tier B)
      planarity  : (l1+l2)/sum(l)            do the centroids lie in a plane? circle -> ~1 (tier D)
      aspect     : l2/l1                     circle -> ~1, straight line -> ~0        (tier D)
      radial_cv  : CV of radii in that plane circle -> ~0                              (tier D)
      n_dead     : #dims with std < 1e-3     conditioning (inflates distances if > 0)
      decode_kl  : reconstruction_kl         functional validity, lower=better (tier C)
    """
    import glob
    import warnings
    import numpy as np
    from safetensors.torch import load_file
    from sklearn.metrics import silhouette_score
    from causalab.io.nested_artifacts import load_nested
    from causalab.neural.featurizer import Featurizer

    # LLM activations are ill-conditioned; sklearn/numpy emit overflow noise that
    # doesn't affect the (float64) metrics below.
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    rows = []
    for mdir in sorted(glob.glob(os.path.join(results_dir, "activation_manifolds", "*"))):
        base = os.path.basename(mdir)
        site = base.split("__")[0]
        fpath = os.path.join(mdir, "features.safetensors")
        lpath = os.path.join(mdir, "features_labels.json")
        if not (os.path.exists(fpath) and os.path.exists(lpath)):
            continue
        X = load_file(fpath)["features"].float().numpy()
        labels = np.array(json.load(open(lpath)))

        # tier A
        explained = np.nan
        mpath = os.path.join(mdir, "manifold_spline", "metadata.json")
        if os.path.exists(mpath):
            explained = 1.0 - json.load(open(mpath)).get("recon_mse", np.nan)

        # tier B — separability in raw PCA space
        classes = sorted(set(labels.tolist()))
        cents = np.stack([X[labels == c].mean(0) for c in classes])
        within = float(np.mean([np.linalg.norm(X[labels == c] - cents[i], axis=1).mean()
                                for i, c in enumerate(classes)]))
        D = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
        between = float(D[~np.eye(len(classes), dtype=bool)].mean())
        sil = float(silhouette_score(X, labels)) if len(classes) > 1 and len(X) > len(classes) else np.nan

        # tier D — is the centroid polygon a circle?
        Cc = cents - cents.mean(0)
        lam = np.linalg.svd(Cc, compute_uv=False) ** 2
        planarity = float(lam[:2].sum() / lam.sum()) if lam.sum() > 0 else np.nan
        aspect = float(lam[1] / lam[0]) if lam[0] > 0 else np.nan
        V = np.linalg.svd(Cc, full_matrices=False)[2][:2]
        radii = np.linalg.norm(Cc @ V.T, axis=1)
        radial_cv = float(radii.std() / radii.mean()) if radii.mean() > 0 else np.nan

        # conditioning
        n_dead = -1
        try:
            key = os.listdir(os.path.join(mdir, "models"))[0]
            payload, _ = load_nested(os.path.join(mdir, "models", key), "featurizers")
            std = Featurizer.from_dict(payload[site]).stages[1].featurizer._std
            n_dead = int((std < 1e-3).sum())
        except Exception:
            pass

        decode_kl = np.nan
        dpath = os.path.join(mdir, "decoding_eval.json")
        if os.path.exists(dpath):
            decode_kl = json.load(open(dpath)).get("reconstruction_kl", np.nan)

        rows.append(dict(pos="entity" if "Token-entity" in site else "last_token",
                         layer=_layer_of(site), n=len(X), k=X.shape[1], cls=len(classes),
                         explained=explained, sil=sil, separab=between / within if within else np.nan,
                         planarity=planarity, aspect=aspect, radial_cv=radial_cv,
                         n_dead=n_dead, decode_kl=decode_kl))

    if not rows:
        print("\n[health] no manifolds with saved features found in", results_dir)
        return rows
    print("\n===== MANIFOLD HEALTH =====")
    print(f"{'pos':11s}{'L':>4s}{'n':>6s}{'cls':>5s}{'explained':>10s}{'silhou':>8s}"
          f"{'separab':>9s}{'planar':>8s}{'aspect':>8s}{'radCV':>7s}{'dead':>6s}{'decodeKL':>10s}")
    for r in sorted(rows, key=lambda r: (r["pos"], r["layer"] if r["layer"] is not None else -1)):
        print(f"{r['pos']:11s}{r['layer'] if r['layer'] is not None else -1:>4d}{r['n']:>6d}{r['cls']:>5d}"
              f"{r['explained']:>10.3f}{r['sil']:>8.3f}{r['separab']:>9.2f}{r['planarity']:>8.3f}"
              f"{r['aspect']:>8.3f}{r['radial_cv']:>7.3f}{r['n_dead']:>6d}{r['decode_kl']:>10.4f}")
    with open(os.path.join(results_dir, "manifold_health.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("wrote", os.path.join(results_dir, "manifold_health.json"))
    return rows


def render_behaviour_manifold(results_dir, out_path):
    """Static 3D PNG of the behaviour manifold M_y (causalab only saves the
    interactive HTML for it). Belief spline + 7 class centroids, projected to 3D
    via the saved Hellinger PCA. From artifacts only — no model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from causalab.methods.spline.belief_fit import load_output_manifold
    from causalab.io.sklearn_pca import load_pca

    root = os.path.join(results_dir, "output_manifolds", "behaviour_manifold")
    spline, centroids_h = load_output_manifold(root, "spline_s0.0/result")
    pca = load_pca(os.path.join(root, "output_manifold"), "hellinger_pca")

    period = float(spline._periods[0]) if spline._periods else 1.0
    u = torch.linspace(0, period, 200).unsqueeze(-1)
    curve3d = pca.transform(spline.decode(u).detach().numpy())
    cent3d = pca.transform(centroids_h.detach().numpy())

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(curve3d[:, 0], curve3d[:, 1], curve3d[:, 2], "-", lw=1.5, alpha=0.85, label="M_y (belief spline)")
    ax.scatter(cent3d[:, 0], cent3d[:, 1], cent3d[:, 2], s=60, marker="D", label="class centroids")
    ax.set_title("Behaviour manifold M_y (Hellinger PCA)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print("saved behaviour-manifold PNG ->", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_smoke_pod",
                    help="dir holding activation_manifolds/, intervened_reps/, output_manifolds/")
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian", "DAS"])
    ap.add_argument("--aggregate", metavar="PARENT_DIR",
                    help="skip per-seed analysis; instead average the existing "
                         "PARENT_DIR/seed_*/divergence_*.json across seeds")
    ap.add_argument("--side", default="activation", choices=["activation", "behaviour","health"],
                    help="activation = projection/emd/bhatt vs manifold; behaviour = E_BC vs M_y")
    ap.add_argument("--view", default="respective", choices=["respective", "union", "gap", "global"],
                    help="activation side only")
    ap.add_argument("--metric", default="projection_distance",
                    choices=["projection_distance", "emd_distance", "bhattacharyya_distance"],
                    help="activation side only")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_seeds(args.aggregate, side=args.side, view=args.view, metric=args.metric)
        return

    # results_layout reads OFFDIST_RESULTS_DIR at import, so set it BEFORE importing.
    os.environ["OFFDIST_RESULTS_DIR"] = args.results_dir

    from activation_manifold import DEFAULT_COMBINATIONS
    oda = importlib.import_module("off-distribution-analyses")

    if args.side == "health":
        return manifold_health(args.results_dir)

    # Analyse what the run ACTUALLY produced, not what the local file declares.
    combos = discover_combinations(args.results_dir, DEFAULT_COMBINATIONS)
    if not combos:
        # Don't silently fall back to DEFAULT_COMBINATIONS — that just produces a
        # wall of [skip] lines and hides the real problem (usually a wrong path).
        looked = os.path.join(args.results_dir, "intervened_reps")
        print(f"\n[error] found no '{looked}/*/' dirs containing *_intervened_reps.safetensors")
        if not os.path.isdir(args.results_dir):
            print(f"        --results-dir does not exist: {args.results_dir}")
        else:
            print(f"        {args.results_dir} contains: {sorted(os.listdir(args.results_dir))}")
            if os.path.isdir(looked):
                print(f"        {looked} contains: {sorted(os.listdir(looked))[:8]}")
            else:
                print(f"        (no 'intervened_reps' subdir here — is the results root one level down?)")
        return
    print("\nanalysing (label -> categories found on disk):")
    for lab, c in combos.items():
        print(f"  {lab}: {c['categories']}")

    act = oda.divergence_metrics(
        combos, modes=args.modes,
        save_to=os.path.join(args.results_dir, "divergence_activation.json"),
    )
    _print_activation(act, args.modes)
    _plot_by_layer(act, args.modes,
                   os.path.join(args.results_dir, "projection_by_layer.png"))
    try:
        render_behaviour_manifold(args.results_dir,
                                  os.path.join(args.results_dir, "behaviour_manifold_3d.png"))
    except Exception as e:
        print(f"[behaviour-manifold PNG skipped] {type(e).__name__}: {e}")

    try:
        beh = oda.behaviour_divergence(
            combos, modes=args.modes,
            save_to=os.path.join(args.results_dir, "divergence_behaviour.json"),
        )
        _print_behaviour(beh, args.modes)
    except FileNotFoundError as e:
        print("\n[behaviour side skipped] ", e)

    print(f"\nWrote divergence_activation.json"
          f"{' + divergence_behaviour.json' if 'beh' in dir() else ''} to {args.results_dir}/")


if __name__ == "__main__":
    main()
