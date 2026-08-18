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
import math
import os
import sys

import numpy as np
import ot
import torch
from safetensors.torch import load_file

from causalab.io.nested_artifacts import load_nested
from causalab.methods.distances import hellinger, wasserstein1_cyclic, wasserstein1_noncyclic
from causalab.methods.scores.distance_from_behavior_manifold import compute_score_single_path
from causalab.methods.spline.belief_fit import load_output_manifold
from causalab.neural.featurizer import Featurizer


# Mirror the writer's per-case results tree. results_layout freezes its paths from
# OFFDIST_RESULTS_DIR AT IMPORT, so if the user didn't export it we derive it from --case
# (results/{case}) BEFORE importing results_layout below — otherwise the reader looks in
# the bare "results/" root and misses results/{case}/dormancy|intervened_reps/... An
# explicit env still wins (setdefault), e.g. a seed-swept OFFDIST_RESULTS_DIR=results/18/seed3.
def _prearg(flag, default=None):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


_case = _prearg("--case")
if _case is not None:
    os.environ.setdefault("OFFDIST_RESULTS_DIR", os.path.join("results", str(_case)))

from results_layout import (
    INTERVENED_REPS, PATCHED_DISTS, BEHAVIOUR_MANIFOLD, ACT_MANIFOLDS,
    label_category_dir, site_context_dir, iter_site_contexts,
    iter_label_category_contexts, dormancy_dir,
)

GLOBAL_CONTEXT = "global"
WRAP_CVAR = "wrapping"  # must match activation_manifold.WRAP_CVAR (predicate-based context)


# --------------------------------------------------------------------------- #
#  Loading (paths mirror the writers in activation_manifold.py exactly)       #
# --------------------------------------------------------------------------- #
def load_intervened_reps(label, category, mode, results_dir=INTERVENED_REPS, context=None):
    """Returns {site_id: (N, d_model) float tensor} for one (label, category, context)."""
    rep_path = os.path.join(label_category_dir(results_dir, label, category, context),
                            f"{mode}_intervened_reps.safetensors")
    if not os.path.exists(rep_path):
        raise FileNotFoundError(
            f"{rep_path} not found — run activation_manifold.py main() for "
            f"label={label!r} category={category!r} mode={mode!r} first."
        )
    return {k: v.float() for k, v in load_file(rep_path).items()}


def contexts_for(label, category, root=INTERVENED_REPS):
    """Which contexts this (label, category) has reps for, read off the dir names
    ({label}__{category}__{context}). Returns [None] for the old flat layout."""
    from results_layout import iter_label_category_contexts
    found = [ctx for lab, cat, ctx in iter_label_category_contexts(root)
             if lab == str(label) and cat == str(category)]
    return found or [None]


def list_contexts(site_id, manifold_cvars):
    """The context strings that have a fitted manifold for this site AND belong
    to THIS label's conditioning.

    Two context flavours are stored as {site_id}__{context}:
      * VARIABLE-based (e.g. m_vars=["entity"]) -> context is a tuple repr like
        "(('entity', 'Monday'),)". A site shared by several labels accumulates
        contexts from all of them, so we keep only those whose variable set ==
        m_cvars (so "entity" doesn't pick up "number").
      * PREDICATE-based (e.g. m_vars=["wrapping"]) -> context is a plain word like
        "wrap" / "no_wrap" (no tuple structure to parse). These aren't
        ast-parseable, so we take every non-global plain-word context as belonging
        to this label's split.
    """
    wanted = set(manifold_cvars)
    out = []
    for c in iter_site_contexts(site_id):
        if c == GLOBAL_CONTEXT:
            continue
        if c.startswith("("):  # variable-based tuple context -> filter by variable set
            if {var for var, _val in ast.literal_eval(c)} == wanted:
                out.append(c)
        elif manifold_cvars == [WRAP_CVAR]:  # plain-word context, and THIS is the predicate label
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
                         output_manifold_sub="spline_s0.0/result",  # causalab appends the target-variable subdir
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
            for context in contexts_for(label, category, results_dir):
                for mode in modes:
                    dist_path = os.path.join(
                        label_category_dir(results_dir, label, category, context),
                        f"{mode}_output_dists.safetensors")
                    if not os.path.exists(dist_path):  # partial run — skip, don't abort
                        print(f"  [skip] no patched dists for {label} / {category} / {context} / {mode}")
                        continue
                    dists = load_file(dist_path)  # {site_id: (N, W) concept-slice probs}
                    results[(label, category, context, mode)] = {
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

    Reps are now stored per context ({label}__{category}__{context}), so for a
    contextual label each site gets THREE views:
      "respective" : vs the rep's OWN context manifold — "is it where it should
                     be for its context?"  (the dormant-pathway test)
      "union"      : min over ALL context manifolds — "is it near SOME natural
                     manifold at all?"     (the null / plausibility check)
      "gap"        : respective - union. ~0 means it sits on its own manifold;
                     LARGE means it's on a manifold but the WRONG one — i.e. the
                     intervention pushed it into another context's representation.
    A global label ({} m_cvars) has one manifold, so respective == union == global.

    Returns {(label, category, context, mode): {site_id: {...}}}.
    """
    results = {}
    for label, combo in combinations.items():
        m_cvars = combo["m_vars"]
        for category in combo["categories"]:
            for context in contexts_for(label, category, results_dir):
                for mode in modes:
                    try:  # partial/interrupted runs: analyse what exists, skip the rest
                        reps = load_intervened_reps(label, category, mode, results_dir, context)
                    except FileNotFoundError:
                        print(f"  [skip] no reps for {label} / {category} / {context} / {mode}")
                        continue
                    per_site = {}
                    for site_id, R in reps.items():
                        if not m_cvars:  # single global manifold
                            to_space, manifold = load_manifold(site_id, GLOBAL_CONTEXT)
                            _, m = _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples)
                            per_site[site_id] = {"global": m}
                            continue
                        # RESPECTIVE: this context's own manifold
                        to_space, manifold = load_manifold(site_id, context)
                        _, resp = _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples)
                        # UNION: nearest of ALL this site's context manifolds
                        union = _union_over_contexts(R, site_id, m_cvars, n_manifold_samples)["union"]
                        gap = {k: resp[k] - union[k] for k in _METRIC_KEYS}
                        per_site[site_id] = {"respective": resp, "union": union, "gap": gap}
                    results[(label, category, context, mode)] = per_site

    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    return results


# --------------------------------------------------------------------------- #
#  Dormancy: off-manifold divergence  x  apparent circuit  x  true circuit     #
# --------------------------------------------------------------------------- #
# The scientific claim: off-manifold interventions INFLATE a component's apparent circuit
# membership. So per (mode, context) we join, per site: the apparent patching score
# (continuous dormancy signal, from run_dormancy_pipeline), true membership (circuit.json),
# and the off-manifold divergence of the intervened rep — vs the GLOBAL manifold and vs the
# site's OWN CONTEXT manifold. The apparent circuit is the TOP-K by score, k = |circuit ∩
# grid| (a self-calibrating cutoff, not an arbitrary threshold). Then FP = apparent ∧
# ¬true, FN = true ∧ ¬apparent, and we ask whether off-manifold divergence PREDICTS the FPs.
#
# Reads (per case; run with OFFDIST_RESULTS_DIR=results/{case}):
#   dormancy/circuit.json                                       {k, in_circuit, is_circuit}
#   intervened_reps/{case}__{cat}__{ctx}/apparent_scores.json       {mode: {site: score}}
#   intervened_reps/{case}__{cat}__{ctx}/{mode}_intervened_reps.safetensors
#   activation_manifolds/{site}__global/ , {site}__{ctx}/           (fitted manifolds)


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(x):
    """Average-tie ranks (1..n), so ties don't bias Spearman / the AUROC U-statistic."""
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(1, len(x) + 1)
    _vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = (csum - counts + csum + 1) / 2.0  # mean rank within each tie group
    return avg[inv]


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return _pearson(_rankdata(x), _rankdata(y))


def _corr(x, y):
    # drop NaN-aligned pairs (e.g. degenerate-cloud sites whose off-manifold is undefined)
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    return {"spearman": _spearman(x, y), "pearson": _pearson(x, y), "n": int(len(x))}


def _auroc(scores, labels):
    """AUROC via the rank (Mann-Whitney U) form: P(score of a positive > score of a
    negative). NaN when either class is empty."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


def _safe_mean(xs):
    xs = [v for v in xs if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(xs)) if xs else float("nan")


# --------------------------------------------------------------------------- #
#  Off-manifold metric B: local-PCA residual / Gaussian-Mahalanobis (density)  #
# --------------------------------------------------------------------------- #
# The concept spline is degenerate for tracr (result explains ~2% of variance -> the 1-D
# curve can't model the ~8-D cloud, so projection_distance = cloud thickness = noise). For
# the dormancy question ("did the intervention leave the NATURAL activation cloud?") we
# instead model the cloud directly: fit the top-d PCA tangent of the natural features, and
#   off-manifold  = ‖component of the intervened rep ORTHOGONAL to the tangent‖  (residual)
#   mahalanobis   = whitened distance WITHIN the tangent (atypicality along the manifold)
# This is the GLOBAL special case of a local-PCA (tangent-space) manifold — unsupervised,
# out-of-sample, curvature-agnostic, no spline/flow/DAS. Reuses the saved natural cloud
# (features.safetensors) + the subspace/standardize stages already on disk.
def load_cloud(site_id, context):
    """(to_space, natural): to_space maps raw (N, d_model) reps into the RAW PCA (subspace)
    space; natural is the (M, k) NATURAL cloud (features.safetensors), already in that space.

    We deliberately use ONLY the PCA subspace stage, NOT the standardize stage of the fitted
    featurizer. Standardize scales every PC to ~unit variance, which (a) makes the effective-
    rank cap in _cloud_offmanifold read the variance as uniform (so it never fires), and (b)
    divides near-zero-variance dead dims by ~0, sending an intervened rep to ~1e13 before the
    metric even runs (the IOI embedding stream). In raw PCA space the variance is concentrated
    in the top PCs, so effective rank + per-PC whitening (with a floor) both behave."""
    scd = site_context_dir(site_id, context)
    models_dir = os.path.join(scd, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"no fitted manifold at {models_dir}")
    (key,) = os.listdir(models_dir)
    payload, _ = load_nested(os.path.join(models_dir, key), "featurizers")
    composed = Featurizer.from_dict(payload[site_id])
    subspace = composed.stages[0]  # PCA only (drop standardize + manifold)

    def to_space(x, _s=subspace):
        z, _ = _s.featurize(x)
        return z.float()

    nat = load_file(os.path.join(scd, "features.safetensors"))["features"].float()
    return to_space, nat  # features.safetensors IS the subspace output


def _cloud_offmanifold(natural, Z, d, var_keep=0.99, ridge=1e-2, clip=8.0):
    """Local-PCA (global special case) off-manifold distances of points Z vs the natural
    cloud. Returns (residual, mahalanobis, d_eff) per point:
      residual    = ‖Z - mean, projected OFF the tangent‖  (the off-manifold distance)
      mahalanobis = whitened distance within the tangent (atypicality)

    HARDENED against degenerate clouds (the IOI token-embedding stream is ~rank-2; a fixed
    d=8 whitens near-zero-variance tail PCs and blows Mahalanobis up to ~1e7):
      * d is capped at the cloud's EFFECTIVE RANK — the PCs holding ``var_keep`` of the
        variance — so we never whiten dead dims.
      * each PC's whitening std is FLOORED at ``ridge·σ_max`` (relative to the dominant PC),
        so a tiny-variance dim can't produce an astronomically large whitened coordinate."""
    X = natural.double()
    mu = X.mean(0)
    Xc = X - mu
    Yc = Z.double() - mu
    _U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    sigma = S / max(X.shape[0] - 1, 1) ** 0.5                      # per-PC std
    # DEGENERATE cloud (e.g. the layer-0 embedding stream has ZERO variance in every dim):
    # there is no manifold to be off of, so Mahalanobis is undefined — return NaN (dropped
    # downstream) instead of dividing by ~0 and exploding to 1e13. Residual is still finite.
    if float(sigma[0]) <= 1e-8:
        return Yc.norm(dim=-1), torch.full((Z.shape[0],), float("nan"), dtype=torch.float64), 0
    # effective rank: #PCs to reach var_keep cumulative energy, capped at d
    energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
    d_eff = int(min(d, int((energy < var_keep).sum().item()) + 1, sigma.shape[0]))
    Vd = Vt[:d_eff]                                                # (d_eff, k) tangent basis
    sig = sigma[:d_eff].clamp_min(ridge * float(sigma[0]))         # floored whitening std
    onto = Yc @ Vd.T                                              # tangent coords
    residual = (Yc - onto @ Vd).norm(dim=-1)                      # orthogonal (off-manifold)
    # WINSORIZE each whitened coord at ±clip σ: on a near-degenerate (very tight) cloud a
    # deviation is legitimately thousands of std-devs, which is uninformative and non-
    # comparable across sites; capping bounds Mahalanobis to ~clip·√d_eff while a genuinely
    # off-manifold rep still saturates the cap (ranks highest).
    w = (onto / sig).clamp(-clip, clip)
    maha = w.pow(2).sum(-1).clamp_min(0).sqrt()                   # whitened, floored, capped
    return residual, maha, d_eff


def _offmetrics(site_id, context, R, offmetric, d_cloud, n_manifold_samples):
    """Off-manifold summary for one intervened-rep cloud R vs a (site, context) manifold.
    offmetric:
      'projection'    — spline projection residual (Design A original; explodes 1e6 on the
                        standardized dead dims — see the writeup).
      'mahalanobis'   — WITHIN-tangent whitened distance (atypicality along the manifold);
                        calibrated, but BLIND to structured OOD that leaves orthogonally.
      'residual'      — orthogonal residual OFF the top-d tangent (raw PCA units).
      'residual_norm' — that residual divided by the natural cloud's radius, so it's
                        comparable across sites ('off-manifold fraction of the spread'). This
                        is the one that separates matched/mismatched and cross_task/same-task.
    Returns {'off': <chosen median-per-site>, 'residual', 'residual_norm', 'mahalanobis', 'd'}."""
    if offmetric in ("residual", "residual_norm", "mahalanobis"):
        to_space, natural = load_cloud(site_id, context)
        Z = to_space(R)
        d = int(min(d_cloud, natural.shape[1], max(1, natural.shape[0] - 1)))
        resid, maha, d_eff = _cloud_offmanifold(natural, Z, d)
        # cloud radius = median distance of natural points from their mean; dividing the
        # Euclidean residual by it makes the OFF-manifold distance comparable across sites.
        cloud_rms = float((natural.double() - natural.double().mean(0)).norm(dim=-1).median())
        resid_n = resid / max(cloud_rms, 1e-9)
        primary = {"mahalanobis": maha, "residual": resid, "residual_norm": resid_n}[offmetric]
        # per-site summary = MEDIAN over examples (robust to a few extreme per-example values).
        return {"off": float(primary.nanmedian()),
                "residual": float(resid.median()),
                "residual_norm": float(resid_n.median()),
                "mahalanobis": float(maha.nanmedian()), "d": d_eff}
    to_space, manifold = load_manifold(site_id, context)   # 'projection' (spline)
    _, m = _metrics_vs_manifold(R, to_space, manifold, n_manifold_samples)
    return {"off": m["projection_distance"], "emd": m["emd_distance"]}


def _load_circuit():
    """circuit.json for the case in the CURRENT results tree (OFFDIST_RESULTS_DIR)."""
    with open(os.path.join(dormancy_dir(), "circuit.json")) as f:
        return json.load(f)  # {k, in_circuit, is_circuit:{site:bool}}


def _load_apparent(label, category, context):
    with open(os.path.join(label_category_dir(INTERVENED_REPS, label, category, context),
                           "apparent_scores.json")) as f:
        return json.load(f)  # {mode: {site_id: score}}


def _score_cell(rows, k):
    """Turn a per-site table (apparent, is_circuit, off_global, off_ctx) into the dormancy
    verdict for one (mode, context): top-k FP/FN, the FP-side / FN-side correlations, the
    AUROCs, and per-set divergence means. Mutates rows with apparent_in / FP / FN flags."""
    scored = [r for r in rows if not math.isnan(r["apparent"])]
    ranked = sorted(scored, key=lambda r: r["apparent"], reverse=True)
    apparent_circuit = {r["site"] for r in ranked[:k]}
    for r in rows:
        r["apparent_in"] = r["site"] in apparent_circuit
        r["FP"] = r["apparent_in"] and not r["is_circuit"]
        r["FN"] = (not r["apparent_in"]) and r["is_circuit"]

    noncirc = [r for r in rows if not r["is_circuit"]]  # FP side
    circ = [r for r in rows if r["is_circuit"]]         # FN side
    n_circ = len(circ)
    tp = sum(r["apparent_in"] for r in circ)

    def corr_on(sub, xkey):  # correlate an off-manifold measure with the CONTINUOUS apparent
        return _corr([r[xkey] for r in sub], [r["apparent"] for r in sub])

    return {
        "n_sites": len(rows), "k": k, "n_circuit": n_circ,
        "precision": (tp / k) if k else float("nan"),
        "recall": (tp / n_circ) if n_circ else float("nan"),
        "FP_sites": [r["site"] for r in rows if r["FP"]],
        "FN_sites": [r["site"] for r in rows if r["FN"]],
        # the patching method's OWN detection quality, threshold-free
        "auroc_apparent_vs_circuit": _auroc([r["apparent"] for r in rows],
                                            [int(r["is_circuit"]) for r in rows]),
        # DORMANCY — FP side (non-circuit sites): does off-manifold predict false membership?
        "fp": {
            "n": len(noncirc),
            "corr_apparent_off_global": corr_on(noncirc, "off_global"),  # continuous
            "corr_apparent_off_ctx": corr_on(noncirc, "off_ctx"),
            "auroc_off_global_vs_FP": _auroc([r["off_global"] for r in noncirc],  # binary
                                             [int(r["FP"]) for r in noncirc]),
            "auroc_off_ctx_vs_FP": _auroc([r["off_ctx"] for r in noncirc],
                                          [int(r["FP"]) for r in noncirc]),
            "mean_off_global_FP": _safe_mean([r["off_global"] for r in noncirc if r["FP"]]),
            "mean_off_global_TN": _safe_mean([r["off_global"] for r in noncirc if not r["FP"]]),
        },
        # FN side (circuit sites): are the MISSED real components distinctive?
        "fn": {
            "n": n_circ,
            "corr_apparent_off_global": corr_on(circ, "off_global"),
            "corr_apparent_off_ctx": corr_on(circ, "off_ctx"),
            "mean_off_global_FN": _safe_mean([r["off_global"] for r in circ if r["FN"]]),
            "mean_off_global_hit": _safe_mean([r["off_global"] for r in circ if not r["FN"]]),
        },
        "per_site": rows,
    }


def dormancy_analysis(case, modes, n_manifold_samples=256, offmetric="mahalanobis",
                      d_cloud=8, save_to=None):
    """Per (mode, context) for one case: correlate off-manifold divergence with apparent
    circuit membership. Off-manifold is measured BOTH ways — vs the global manifold
    (``off_global``) and vs the site's own context manifold (``off_ctx``) — so we can see
    which better predicts dormancy.

    ``offmetric`` selects HOW off-manifold is measured:
      'projection'  — spline projection residual (Design A original; degenerate for tracr).
      'residual'    — local-PCA orthogonal residual to the top-``d_cloud`` tangent (density).
      'mahalanobis' — whitened distance within that tangent.
    Kept SEPARATE per case. Returns {(mode, context): cell}; per_site rows are written so
    pooled_dormancy can consume the JSON directly."""
    circ = _load_circuit()
    k, is_circuit = circ["k"], circ["is_circuit"]
    combos = [(cat, ctx) for lab, cat, ctx in iter_label_category_contexts(INTERVENED_REPS)
              if lab == str(case)]

    results = {}
    for category, context in combos:
        try:
            apparent = _load_apparent(case, category, context)
        except FileNotFoundError:
            print(f"  [skip] no apparent_scores for {case}/{category}/{context}")
            continue
        for mode in modes:
            try:
                reps = load_intervened_reps(case, category, mode, INTERVENED_REPS, context)
            except FileNotFoundError:
                print(f"  [skip] no reps for {case}/{category}/{context}/{mode}")
                continue
            rows = []
            for site_id, R in reps.items():
                g = _offmetrics(site_id, GLOBAL_CONTEXT, R, offmetric, d_cloud, n_manifold_samples)
                if context not in (None, GLOBAL_CONTEXT):  # respective (own-context) manifold
                    try:
                        c = _offmetrics(site_id, context, R, offmetric, d_cloud, n_manifold_samples)
                    except FileNotFoundError:
                        c = g  # no context manifold -> fall back to global
                else:
                    c = g  # global-only case (e.g. tracr 18): off_ctx == off_global
                rows.append({
                    "site": site_id,
                    "is_circuit": bool(is_circuit.get(site_id, False)),
                    "apparent": float(apparent.get(mode, {}).get(site_id, float("nan"))),
                    "off_global": g["off"],
                    "off_ctx": c["off"],
                })
            # key by (mode, category, context): IOI has TWO categories per context
            # (context_matched / context_mismatched); keying by context alone collided them.
            results[(mode, category, str(context))] = _score_cell(rows, k)

    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({f"{m}||{cat}||{c}": v for (m, cat, c) in results
                       for v in [results[(m, cat, c)]]}, f, indent=2)
    return results


def pooled_dormancy(dormancy_json_paths, save_to=None):
    """Pool per-site rows across cases for statistical power the small per-case grids lack.

    Off-manifold and apparent are z-scored WITHIN each (case, mode, context) cell before
    pooling — this removes per-model scale/units so a positive off-manifold->apparent
    relationship is comparable across models. Valid because build_full_grid gives EVERY
    model the same component granularity (mlp + heads + both residual streams). Consumes the
    per-case JSONs written by dormancy_analysis (each carries its own per_site rows), so no
    manifolds/reps are re-read and no results-tree switching is needed."""
    from collections import defaultdict
    by_mode = defaultdict(list)  # STR (on-manifold control) vs Gaussian (off-manifold treatment)
    for path in dormancy_json_paths:
        with open(path) as f:
            cells = json.load(f)  # {"mode||cat||context": cell}
        for mc, cell in cells.items():
            mode = mc.split("||")[0]
            rows = [r for r in cell.get("per_site", [])
                    if not (isinstance(r["apparent"], float) and math.isnan(r["apparent"]))]
            if len(rows) < 3:
                continue
            zg = _zscore([r["off_global"] for r in rows])
            zc = _zscore([r["off_ctx"] for r in rows])
            za = _zscore([r["apparent"] for r in rows])
            for i, r in enumerate(rows):
                by_mode[mode].append({**r, "z_off_global": float(zg[i]),
                                      "z_off_ctx": float(zc[i]), "z_apparent": float(za[i])})

    def _summary(pooled):
        noncirc = [r for r in pooled if not r["is_circuit"]]  # FP side
        circ = [r for r in pooled if r["is_circuit"]]         # FN side

        def corr_on(sub, xk):
            return _corr([r[xk] for r in sub], [r["z_apparent"] for r in sub])

        return {
            "n_sites_pooled": len(pooled),
            "fp": {"n": len(noncirc),
                   "corr_apparent_off_global": corr_on(noncirc, "z_off_global"),
                   "corr_apparent_off_ctx": corr_on(noncirc, "z_off_ctx"),
                   "auroc_off_global_vs_FP": _auroc([r["z_off_global"] for r in noncirc],
                                                    [int(r.get("FP", False)) for r in noncirc]),
                   "auroc_off_ctx_vs_FP": _auroc([r["z_off_ctx"] for r in noncirc],
                                                 [int(r.get("FP", False)) for r in noncirc])},
            "fn": {"n": len(circ),
                   "corr_apparent_off_global": corr_on(circ, "z_off_global"),
                   "corr_apparent_off_ctx": corr_on(circ, "z_off_ctx")}}

    # PER MODE — mixing STR (control) and Gaussian (treatment) cancels the dormancy signal.
    out = {"n_cases": len(dormancy_json_paths),
           "by_mode": {m: _summary(rows) for m, rows in by_mode.items()}}
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
    return out


def _spec_of(site_id):
    """Parse a site.id string into its POSITION-INDEPENDENT spec (layer, component, head),
    matching circuit_specs' keys — so apparent scores at different token positions can be
    combined for the same component."""
    import re
    m = re.search(r"Layer-(\d+)", site_id)
    if not m:
        return None
    L = int(m.group(1))
    if site_id.startswith("MLP"):
        return (L, "mlp_activation", None)
    if site_id.startswith("AttentionHead"):
        return (L, "attn_result", int(re.search(r"Head-(\d+)", site_id).group(1)))
    if site_id.startswith("ResidualStream"):
        return (L, "block_output" if "block_output" in site_id else "block_input", None)
    return None


def combine_positions(roots, case, modes, save_to=None):
    """Position-aware apparent circuit. IOI's circuit is DISTRIBUTED across token positions
    (name-movers read at END, s-inhibition / duplicate-token at S2), so single-position
    patching can't recover it — the apparent circuit comes out near-random. This combines the
    apparent scores from several per-position result trees BY SPEC (layer, component, head):
    a component's combined score = its MAX over positions (important at ANY position). Then it
    reports whether the COMBINED apparent circuit clears chance (AUROC vs true membership) —
    the gate that must pass before the IOI dormancy numbers mean anything.

    roots: list of results/{case}__{position} directories (each a writer run at one position).
    Returns {(mode, category, context): {auroc, precision, recall, per_position_auroc, ...}}."""
    import glob

    circ0 = json.load(open(os.path.join(roots[0], "dormancy", "circuit.json")))
    is_circ = {}
    for sid, v in circ0["is_circuit"].items():
        s = _spec_of(sid)
        if s is not None:
            is_circ[s] = bool(v) or is_circ.get(s, False)
    k = sum(is_circ.values())

    # discover (category, context) from the first root's reps dirs
    combos = set()
    for d in glob.glob(os.path.join(roots[0], "intervened_reps", "*")):
        parts = os.path.basename(d).split("__")
        if len(parts) == 3 and parts[0] == str(case):
            combos.add((parts[1], parts[2]))

    def _apparent(root, cat, ctx, mode):
        p = os.path.join(root, "intervened_reps", f"{case}__{cat}__{ctx}", "apparent_scores.json")
        return json.load(open(p)).get(mode, {}) if os.path.exists(p) else {}

    def _auc_by_spec(app):  # AUROC(apparent ~ is_circuit) over specs
        specs = [s for s in {_spec_of(sid) for sid in app} if s is not None]
        return _auroc([max(float(app[sid]) for sid in app if _spec_of(sid) == s) for s in specs],
                      [int(is_circ.get(s, False)) for s in specs]), specs

    results = {}
    for cat, ctx in sorted(combos):
        for mode in modes:
            per_pos_auc = {}
            comb = {}  # spec -> max apparent over positions
            for root in roots:
                app = _apparent(root, cat, ctx, mode)
                if not app:
                    continue
                per_pos_auc[os.path.basename(root)] = round(_auc_by_spec(app)[0], 3)
                for sid, sc in app.items():
                    s = _spec_of(sid)
                    if s is not None:
                        comb[s] = max(comb.get(s, float("-inf")), float(sc))
            if not comb:
                continue
            specs = list(comb)
            ranked = sorted(specs, key=lambda s: comb[s], reverse=True)
            topk = set(ranked[:k])
            tp = sum(is_circ.get(s, False) for s in topk)
            results[(mode, cat, ctx)] = {
                "k": k, "n_specs": len(specs),
                "auroc_combined_vs_circuit": _auroc([comb[s] for s in specs],
                                                    [int(is_circ.get(s, False)) for s in specs]),
                "per_position_auroc": per_pos_auc,
                "precision": tp / k if k else float("nan"),
                "recall": tp / max(sum(is_circ.values()), 1),
            }
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump({f"{m}||{c}||{x}": v for (m, c, x), v in results.items()}, f, indent=2)
    return results


def aggregate_seeds(root, modes=("STR", "Gaussian"), save_to=None,
                    fname="dormancy_mahalanobis.json"):
    """Aggregate a case × seed dormancy sweep (results/sweep/{case}_seed{seed}/dormancy/
    dormancy.json). Two views:

      POOLED-per-seed (the headline): within each seed, pool the per-site rows across ALL
      cases (pooled_dormancy z-scores within each cell first, so per-model scale is removed),
      giving ONE high-N FP-side correlation per seed; then report mean ± std ACROSS seeds. Each
      seed is an independent replicate (different counterfactual sampling), so the std is a
      proper error bar. This is 'combine across cases per seed, average across seeds' — both.

      PER-CASE (stability): for each case, the mean ± std over seeds of its FP-side spearman
      (averaged over that case's cells) — shows which individual cases exhibit dormancy.

    Returns {"pooled": {...mean/std/per_seed...}, "per_case": {case: {...}}}."""
    import glob
    import re
    from collections import defaultdict

    # the reader saves dormancy_{offmetric}.json; prefer the mahalanobis one, fall back to a
    # bare dormancy.json, so this works whether the sweep ran with or without an --offmetric.
    by_seed, by_case = defaultdict(list), defaultdict(list)
    for dd in sorted(glob.glob(os.path.join(root, "*_seed*", "dormancy"))):
        cand = ([os.path.join(dd, fname)] if os.path.exists(os.path.join(dd, fname))
                else sorted(glob.glob(os.path.join(dd, "dormancy*.json"))))
        if not cand:
            continue
        p = cand[0]
        d = os.path.basename(os.path.dirname(dd))  # {case}_seed{seed}
        m = re.match(r"(.+)_seed(\d+)$", d)
        if not m:
            continue
        case, seed = m.group(1), int(m.group(2))
        by_seed[seed].append(p)
        by_case[case].append(p)

    # --- POOLED per seed (per MODE), then mean ± std across seeds ---
    per_seed = [(seed, pooled_dormancy(ps)) for seed, ps in sorted(by_seed.items())]

    def _across_seeds(mode, dotted):  # dotted into by_mode[mode], e.g. "fp.corr_apparent_off_global.spearman"
        vals = []
        for _seed, pl in per_seed:
            cur = pl.get("by_mode", {}).get(mode)
            for kk in dotted.split("."):
                cur = cur.get(kk) if isinstance(cur, dict) else None
            if cur is not None and not (isinstance(cur, float) and math.isnan(cur)):
                vals.append(float(cur))
        return {"mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"), "per_seed": vals}

    pooled = {"n_seeds": len(per_seed), "n_cases": len(by_case), "by_mode": {}}
    for mode in modes:
        pooled["by_mode"][mode] = {
            "fp_spearman_global": _across_seeds(mode, "fp.corr_apparent_off_global.spearman"),
            "fp_spearman_ctx": _across_seeds(mode, "fp.corr_apparent_off_ctx.spearman"),
            "fp_auroc_global": _across_seeds(mode, "fp.auroc_off_global_vs_FP"),
            "fn_spearman_global": _across_seeds(mode, "fn.corr_apparent_off_global.spearman"),
        }

    # --- per case, PER MODE: mean ± std of FP spearman over seeds (that mode's cells) ---
    per_case = {}
    for case, ps in sorted(by_case.items()):
        per_case[case] = {}
        for mode in modes:
            seed_means = []
            for p in ps:
                cells = json.load(open(p))
                xs = [c["fp"]["corr_apparent_off_global"]["spearman"] for k, c in cells.items()
                      if k.split("||")[0] == mode
                      and not math.isnan(c["fp"]["corr_apparent_off_global"]["spearman"])]
                if xs:
                    seed_means.append(float(np.mean(xs)))
            per_case[case][mode] = {
                "mean": float(np.mean(seed_means)) if seed_means else float("nan"),
                "std": float(np.std(seed_means)) if seed_means else float("nan"),
                "n_seeds": len(seed_means)}

    # circuit-recovery quality: mean AUROC(apparent~circuit) per mode across all cells.
    # (Is the apparent circuit even above chance? — the FP/FN partition is only meaningful
    # when it is. Also report the bimodality: on clean tracr circuits STR is often exactly
    # 0 or 1, so the mean hides a "perfect-or-failed" split — frac_perfect flags that.)
    recov = defaultdict(list)
    for ps in by_seed.values():
        for p in ps:
            for k, c in json.load(open(p)).items():
                a = c.get("auroc_apparent_vs_circuit")
                if a is not None and not (isinstance(a, float) and math.isnan(a)):
                    recov[k.split("||")[0]].append(float(a))
    pooled["circuit_recovery"] = {
        m: {"auroc_mean": float(np.mean(v)), "auroc_std": float(np.std(v)),
            "auroc_median": float(np.median(v)),
            "frac_perfect": float(np.mean(np.array(v) >= 0.999)),
            "frac_failed": float(np.mean(np.array(v) <= 0.001)), "n_cells": len(v)}
        for m, v in recov.items()}

    out = {"pooled": pooled, "per_case": per_case}
    if save_to:
        os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
        with open(save_to, "w") as f:
            json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Dormancy correlation. Set OFFDIST_RESULTS_DIR=results/{case} to point "
                    "at the writer's output tree.")
    ap.add_argument("--case", help="case id, e.g. 18 or ioi (single-case analysis)")
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian"])
    ap.add_argument("--offmetric", default="mahalanobis",
                    choices=["projection", "residual", "residual_norm", "mahalanobis"],
                    help="off-manifold distance: mahalanobis=within-tangent (default, isotropic "
                         "interventions); residual_norm=orthogonal residual / cloud radius "
                         "(structured OOD — matched/mismatched, cross_task); residual=raw "
                         "orthogonal; projection=spline (Design A, explodes on dead dims)")
    ap.add_argument("--d-cloud", type=int, default=8,
                    help="tangent dim for residual/mahalanobis (top-d PCA of the natural cloud)")
    ap.add_argument("--pooled", nargs="*", default=None,
                    help="paths to per-case dormancy.json files to pool (runs pooled_dormancy "
                         "instead of a single-case analysis)")
    ap.add_argument("--combine", nargs="*", default=None,
                    help="results/{case}__{position} dirs to COMBINE apparent scores across "
                         "positions (IOI position fix); reports whether the combined apparent "
                         "circuit clears chance. Needs --case.")
    ap.add_argument("--aggregate", default=None,
                    help="root of a case×seed sweep (results/sweep) — pool across cases per "
                         "seed, then average across seeds. Prints the headline dormancy effect.")
    args = ap.parse_args()

    if args.aggregate:
        out = aggregate_seeds(args.aggregate, args.modes,
                              fname=f"dormancy_{args.offmetric}.json",
                              save_to=os.path.join(args.aggregate, f"aggregate_{args.offmetric}.json"))
        p = out["pooled"]
        print(f"=== dormancy sweep aggregate ({p['n_cases']} cases × {p['n_seeds']} seeds) ===")
        print("  (Gaussian = off-manifold TREATMENT = the dormancy test; STR = on-manifold CONTROL)")
        cr = p.get("circuit_recovery", {})
        for mode in args.modes:
            r = cr.get(mode)
            if r:
                print(f"  CIRCUIT RECOVERY {mode}: AUROC(app~circ) mean={r['auroc_mean']:.3f} "
                      f"± {r['auroc_std']:.3f} (median {r['auroc_median']:.2f}) | "
                      f"perfect(=1.0) {r['frac_perfect']:.0%}, failed(=0.0) {r['frac_failed']:.0%}")
        for mode in args.modes:
            bm = p["by_mode"].get(mode, {})
            print(f"  --- {mode} ---")
            for lbl, key in [("FP off_global~apparent (spearman)", "fp_spearman_global"),
                             ("FP AUROC(off_global→FP)         ", "fp_auroc_global"),
                             ("FN off_global~apparent (spearman)", "fn_spearman_global")]:
                v = bm.get(key, {"mean": float("nan"), "std": float("nan"), "per_seed": []})
                print(f"    {lbl}: {v['mean']:+.3f} ± {v['std']:.3f}  (per-seed {[round(x,2) for x in v['per_seed']]})")
        gm = "Gaussian" if "Gaussian" in args.modes else args.modes[-1]
        print(f"  per-case FP spearman [{gm}] (mean±std over seeds), sorted:")
        def _key(kv):
            m = kv[1].get(gm, {}).get("mean", float("nan"))
            return -(m if not math.isnan(m) else -9)
        for case, v in sorted(out["per_case"].items(), key=_key):
            vg = v.get(gm, {"mean": float("nan"), "std": float("nan"), "n_seeds": 0})
            print(f"     case {case:>3}: {vg['mean']:+.2f} ± {vg['std']:.2f}  (n_seeds={vg['n_seeds']})")
        import sys as _s; _s.exit(0)

    if args.combine:
        out = combine_positions(args.combine, args.case, args.modes,
                                save_to=os.path.join(dormancy_dir(), "combined_positions.json"))
        print(f"=== combine_positions (case {args.case}) — apparent circuit AUROC ===")
        for (mode, cat, ctx), v in sorted(out.items()):
            cx = ctx.split("'")[3] if "'" in ctx else ctx
            print(f"[{mode} | {cat.replace('context_','')} | {cx}] "
                  f"COMBINED AUROC(app~circ)={v['auroc_combined_vs_circuit']:.2f} "
                  f"P={v['precision']:.2f} R={v['recall']:.2f} | per-position: {v['per_position_auroc']}")
        import sys as _s; _s.exit(0)

    if args.pooled:
        out = pooled_dormancy(args.pooled, save_to=os.path.join(dormancy_dir(), "pooled.json"))
        print(json.dumps(out, indent=2))
    else:
        res = dormancy_analysis(args.case, args.modes, offmetric=args.offmetric,
                                d_cloud=args.d_cloud,
                                save_to=os.path.join(dormancy_dir(),
                                                     f"dormancy_{args.offmetric}.json"))
        for (mode, cat, ctx), cell in res.items():
            fp = cell["fp"]["corr_apparent_off_global"]
            print(f"[{mode} | {cat} | {ctx}] P={cell['precision']:.2f} R={cell['recall']:.2f} "
                  f"AUROC(apparent~circuit)={cell['auroc_apparent_vs_circuit']:.2f} | "
                  f"FP off_global~apparent: spearman={fp['spearman']:.2f} (n={fp['n']}) | "
                  f"AUROC(off_global~FP)={cell['fp']['auroc_off_global_vs_FP']:.2f}")
