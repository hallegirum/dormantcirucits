"""
Compare intervened representations (STR / Gaussian / DAS) against the fitted
activation manifolds, per site.

Granularity: the (label, category, mode) loop already fixes which counterfactual
regime the reps come from (context_matched / context_mismatched / cross_task),
so each site's FULL rep cloud is compared against every context manifold saved
for that site (plus the global one, when fit). Reported per site:
  - per-context metrics,
  - "union": distance to the union of context manifolds (per-point min over
    contexts for projection; min over contexts for the cloud metrics),
  - "global": metrics vs the global manifold, when one exists on disk.
Conclusions are then drawn by contrasting these across categories.

On-disk layout this reads (see results_layout.py for the single definition):
  results/intervened_reps/{label}__{category}/{mode}_intervened_reps.safetensors
      dict: site_id -> (N, d_model) raw intervened activations
  results/activation_manifolds/{site_id}__{context}/models/{layer}__{index}/
      composed featurizer (raw -> PCA -> standardize -> spline manifold)
  results/output_manifolds/patched_output_distributions/{label}__{category}/...
  results/output_manifolds/behaviour_manifold/  (fitted M_y)

The manifold object reloaded here is a SplineManifold (torch.nn.Module):
control_points (C, intrinsic_dim) are the intrinsic coords of the C class
centroids (7 weekdays -> C=7, intrinsic_dim=1 on a period-7 circle), centroids
(C, k) their images in the standardized k=64 PCA space; encode/decode map
between the two.

Metrics:
  ambient (standardized k-space point clouds):
  - projection_distance : Gauss-Newton projection onto the spline
                          (SplineManifold.encode_to_nearest_point) — per-point
                          ||residual||, the true off-manifold distance.
  - emd_distance        : exact W1 between the rep cloud and a cloud sampled ON
                          the manifold. Point clouds carry no category
                          structure, so causalab's wasserstein1_* (which take
                          probability vectors over W categories) don't apply
                          here; POT's exact solver IS the standard library for
                          this (torch has no built-in EMD, scipy's is 1-D
                          only, and causalab itself computes all its
                          Wasserstein distances with POT's `ot.emd2`).
  - bhattacharyya_distance : closed-form between Gaussians fit to the clouds
                          (activations aren't probability vectors, so the
                          distribution formulas need a density model first;
                          Gaussian is the standard closed-form choice).
  intrinsic (weekday-bin histograms of the coordinate ALONG the manifold):
  currently COMMENTED OUT of the reported metrics pending review — the
  functions remain below for when we come back to them.

  behaviour (the paper's E_BC, single-position so no path integral):
  - behaviour_divergence : the real compute_score_single_path from causalab's
    distance_from_behavior_manifold — mean Bhattacharyya distance from each
    patched-run OUTPUT distribution to the nearest point on the behaviour
    manifold M_y. Inputs are produced by collect_behaviour.py (patched output
    distributions + the fitted belief manifold).
"""

import ast
import json
import os

import ot
import torch
from safetensors.torch import load_file

from causalab.io.nested_artifacts import load_nested
from causalab.methods.distances import hellinger, wasserstein1_cyclic, wasserstein1_noncyclic
from causalab.methods.scores.distance_from_behavior_manifold import compute_score_single_path
from causalab.methods.spline.belief_fit import load_output_manifold
from causalab.neural.featurizer import Featurizer
from results_layout import (
    INTERVENED_REPS, PATCHED_DISTS, BEHAVIOUR_MANIFOLD, ACT_MANIFOLDS,
    label_category_dir, site_context_dir, iter_site_contexts,
)

GLOBAL_CONTEXT = "global"


# --------------------------------------------------------------------------- #
#  Loading (paths mirror the writers in activation_manifold.py exactly)       #
# --------------------------------------------------------------------------- #
def load_intervened_reps(label, category, mode, results_dir=INTERVENED_REPS):
    """Returns {site_id: (N, d_model) float tensor}."""
    rep_path = os.path.join(label_category_dir(results_dir, label, category),
                            f"{mode}_intervened_reps.safetensors")
    if not os.path.exists(rep_path):
        raise FileNotFoundError(
            f"{rep_path} not found — run activation_manifold.py main() for "
            f"label={label!r} category={category!r} mode={mode!r} first."
        )
    return {k: v.float() for k, v in load_file(rep_path).items()}


def list_contexts(site_id, manifold_cvars):
    """The context strings that have a fitted manifold for this site AND belong
    to THIS label's conditioning variables. Manifold dirs are named
    {site_id}__{context} with context = str(context_key(ex, m_cvars)), e.g.
    "(('entity', 'Monday'),)". Since the dir name carries no label, a site shared
    by several labels accumulates contexts from all of them — we keep only those
    whose variable set == m_cvars (so "entity" doesn't pick up "number")."""
    wanted = set(manifold_cvars)
    out = []
    for c in iter_site_contexts(site_id):
        if c == GLOBAL_CONTEXT:
            continue
        vars_in_ctx = {var for var, _val in ast.literal_eval(c)}  # ("entity","Monday") -> "entity"
        if vars_in_ctx == wanted:
            out.append(c)
    return out


def load_manifold(site_id, context):
    """Reload the fitted manifold for (site, context): returns
    (to_manifold_space, manifold) where to_manifold_space maps raw (N, d_model)
    activations into the standardized k-space the spline lives in. Manifolds are
    stored per (site, context) without mode (a spline is a PCA fit, and DAS sites
    are the STR∪Gaussian union, so every site already has one)."""
    models_dir = os.path.join(site_context_dir(site_id, context), "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"no fitted manifold at {models_dir}")
    (key,) = os.listdir(models_dir)  # single unit per fit: {layer}__{index}
    payload, _ = load_nested(os.path.join(models_dir, key), "featurizers")
    composed = Featurizer.from_dict(payload[site_id])
    pre, manifold_stage = composed.stages[:-1], composed.stages[-1]

    def to_manifold_space(x, _pre=pre):
        for stage in _pre:
            x, _ = stage.featurize(x)
        return x.float()

    return to_manifold_space, manifold_stage.featurizer.manifold


def sample_manifold_points(manifold, n=256):
    """Sample a reference cloud ON the manifold by decoding a grid of intrinsic
    coordinates spanning the fitted control points."""
    cp = manifold.control_points  # (C, intrinsic_dim)
    d = cp.shape[1]
    per_dim = max(2, round(n ** (1.0 / d)))
    axes = [torch.linspace(cp[:, i].min(), cp[:, i].max(), per_dim) for i in range(d)]
    grid = torch.cartesian_prod(*axes).reshape(-1, d)
    with torch.no_grad():
        return manifold.decode(grid)


# --------------------------------------------------------------------------- #
#  Metrics                                                                    #
# --------------------------------------------------------------------------- #
def projection_distances(points, manifold):
    """Per-point distance to the manifold: Gauss-Newton projection residual
    norm, argmin_u ||decode(u) - x||. Returns (N,) tensor."""
    with torch.no_grad():
        _, residual = manifold.encode_to_nearest_point(points)
    return residual.norm(dim=-1)


def compute_emd(cloud_a, cloud_b):
    """Exact W1 (earth mover's) distance between two point clouds, uniform
    weights, Euclidean ground cost (POT exact solver)."""
    a = torch.full((cloud_a.shape[0],), 1.0 / cloud_a.shape[0], dtype=torch.float64)
    b = torch.full((cloud_b.shape[0],), 1.0 / cloud_b.shape[0], dtype=torch.float64)
    M = torch.cdist(cloud_a.double(), cloud_b.double())
    return float(ot.emd2(a.numpy(), b.numpy(), M.numpy()))


def gaussian_bhattacharyya(cloud_a, cloud_b, ridge=1e-3):
    """Closed-form Bhattacharyya distance between Gaussians fit to two clouds:
    1/8 dmu^T S^-1 dmu + 1/2 ln( det S / sqrt(det S_a det S_b) ), S = (S_a+S_b)/2.
    A ridge (scaled to mean variance) keeps covariances invertible when N < k."""
    Xa, Xb = cloud_a.double(), cloud_b.double()
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    ca = torch.cov(Xa.T) if Xa.shape[0] > 1 else torch.zeros(Xa.shape[1], Xa.shape[1], dtype=torch.float64)
    cb = torch.cov(Xb.T) if Xb.shape[0] > 1 else torch.zeros_like(ca)
    eye = torch.eye(ca.shape[0], dtype=torch.float64)
    eps = ridge * float((ca.diagonal().mean() + cb.diagonal().mean()) / 2 + 1e-12)
    ca, cb = ca + eps * eye, cb + eps * eye
    S = (ca + cb) / 2
    dmu = (mu_a - mu_b).unsqueeze(-1)
    maha = (dmu.T @ torch.linalg.solve(S, dmu)).squeeze() / 8
    logdet = (torch.linalg.slogdet(S)[1]
              - 0.5 * (torch.linalg.slogdet(ca)[1] + torch.linalg.slogdet(cb)[1])) / 2
    return float(maha + logdet)


def hellinger_to_bhattacharyya(h):
    """d_B = -ln(BC) with BC = 1 - H^2 — the relation used by causalab's
    distance_from_behavior_manifold."""
    return float(-torch.log((1.0 - h**2).clamp_min(1e-12)))


def intrinsic_histogram(points, manifold, n_bins=None):
    """Distribution of the coordinate ALONG the manifold (1-D intrinsic only):
    project points, histogram u over the period (cyclic) or control-point range.
    Returns (p, cyclic)."""
    with torch.no_grad():
        u, _ = manifold.encode_to_nearest_point(points)
    u = u[:, 0]
    periods = getattr(manifold, "_periods", None) or []
    cyclic = len(periods) > 0
    if cyclic:
        period = float(periods[0])
        u = u % period
        lo, hi = 0.0, period
        n_bins = n_bins or max(2, round(period))  # weekday circle: 7 bins
    else:
        cp = manifold.control_points[:, 0]
        lo, hi = float(cp.min()), float(cp.max())
        n_bins = n_bins or 10
    hist = torch.histc(u.float(), bins=n_bins, min=lo, max=hi)
    return hist / hist.sum().clamp_min(1.0), cyclic


def intrinsic_distribution_metrics(points, manifold, n_bins=None):
    """EMD + Bhattacharyya on the intrinsic (along-manifold) distribution vs
    uniform (manifolds are fit with balanced classes), via causalab's own
    wasserstein1_cyclic / hellinger."""
    p, cyclic = intrinsic_histogram(points, manifold, n_bins)
    q = torch.full_like(p, 1.0 / p.shape[0])
    w1 = wasserstein1_cyclic(p, q) if cyclic else wasserstein1_noncyclic(p, q)
    return {
        "intrinsic_emd": float(w1),
        "intrinsic_bhattacharyya": hellinger_to_bhattacharyya(hellinger(p, q)),
    }


def behaviour_divergence(combinations, modes,
                         results_dir=PATCHED_DISTS,
                         experiment_root=BEHAVIOUR_MANIFOLD,
                         output_manifold_sub="spline_s0.0",
                         save_to=None):
    """The paper's E_BC, single-position (no path integral): per site, the mean
    Bhattacharyya distance from each patched-run output distribution to the
    NEAREST point on the behaviour manifold M_y — computed by causalab's own
    compute_score_single_path. Iterates the SAME (label, category) pairs that
    were run (from `combinations`, not a cartesian product) crossed with modes.
    M_y is a single global belief manifold, so no context grouping here. Inputs:
      {results_dir}/{label}/{category}/{mode}_output_dists.safetensors
      {experiment_root}/output_manifold/{output_manifold_sub}/  (fitted M_y)
    Returns {(label, category, mode): {site_id: mean_d_B}}."""
    belief_spline, _ = load_output_manifold(experiment_root, output_manifold_sub)
    results = {}
    for label, combo in combinations.items():
        for category in combo["categories"]:
            for mode in modes:
                dist_path = os.path.join(label_category_dir(results_dir, label, category),
                                         f"{mode}_output_dists.safetensors")
                if not os.path.exists(dist_path):
                    raise FileNotFoundError(
                        f"{dist_path} not found — run collect_behaviour.py main() first."
                    )
                dists = load_file(dist_path)  # {site_id: (N, W) concept-slice probs}
                results[(label, category, mode)] = {
                    site_id: compute_score_single_path(p, belief_spline)
                    for site_id, p in dists.items()
                }
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results


# --------------------------------------------------------------------------- #
#  Driver                                                                     #
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ("projection_distance", "emd_distance", "bhattacharyya_distance")
# intrinsic metrics commented out for now — revisit later:
# _METRIC_KEYS += ("intrinsic_emd", "intrinsic_bhattacharyya")


def _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples):
    Z = to_space(R)
    on_manifold = sample_manifold_points(manifold, n_manifold_samples)
    per_point = projection_distances(Z, manifold)
    return per_point, {
        "projection_distance": float(per_point.mean()),
        "emd_distance": compute_emd(Z, on_manifold),
        "bhattacharyya_distance": gaussian_bhattacharyya(Z, on_manifold),
        # **intrinsic_distribution_metrics(Z, manifold),  # commented out for now
    }


def _union_over_contexts(R, site_id, m_cvars, n_manifold_samples):
    """Contextual label: compare the rep cloud to EACH context manifold, then
    reduce to the "union" — distance to the nearest context manifold. A rep that
    sits on its own weekday's manifold is on-manifold even though it's far from
    the other six; the union answers "is each rep near SOME natural context
    manifold" without needing to know which context each rep came from. For
    projection this is per-point (min over contexts, then mean); for the cloud
    metrics it's the min over contexts."""
    per_ctx, per_point_all = {}, []
    for ctx in list_contexts(site_id, m_cvars):
        to_space, manifold = load_manifold(site_id, ctx)
        per_point, per_ctx[ctx] = _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples)
        per_point_all.append(per_point)
    if not per_ctx:
        raise FileNotFoundError(
            f"no context manifolds for site {site_id!r} with m_cvars={m_cvars} under "
            f"{ACT_MANIFOLDS} — run activation_manifold.py main() so every top site is fit."
        )
    union = {k: min(v[k] for v in per_ctx.values()) for k in _METRIC_KEYS}
    union["projection_distance"] = float(torch.stack(per_point_all).min(dim=0).values.mean())
    return {"contexts": per_ctx, "union": union}


def divergence_metrics(combinations, modes,
                       results_dir=INTERVENED_REPS,
                       n_manifold_samples=256,
                       save_to=None):
    """Activation-side metrics for the SAME (label, category) pairs that were
    run, crossed with modes — NOT a cartesian product of labels×categories
    (each label carries its own category list, like DEFAULT_COMBINATIONS).

    Per site, the manifold(s) to compare against are decided by the label's
    manifold_conditioning_vars (m_cvars):
      m_cvars == []  (global label)     -> the single "global" manifold.
      m_cvars != []  (contextual label) -> the union over context manifolds.

    Returns {(label, category, mode): {site_id: {...}}} where the inner dict is
    {"global": {...}} for a global label, or {"contexts": {...}, "union": {...}}
    for a contextual one.
    """
    results = {}
    for label, combo in combinations.items():
        m_cvars = combo["m_vars"]
        for category in combo["categories"]:
            for mode in modes:
                reps = load_intervened_reps(label, category, mode, results_dir)
                per_site = {}
                for site_id, R in reps.items():
                    if m_cvars:
                        per_site[site_id] = _union_over_contexts(
                            R, site_id, m_cvars, n_manifold_samples)
                    else:
                        to_space, manifold = load_manifold(site_id, GLOBAL_CONTEXT)
                        _, m = _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples)
                        per_site[site_id] = {"global": m}
                results[(label, category, mode)] = per_site

    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results
