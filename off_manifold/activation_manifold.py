
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # linalg.householder_product not implemented on MPS
import json
import logging
import math
import random


# causalab logs the (harmless, caught) categorical-viz failures — "could not
# convert string to float: 'Saturday'" — as WARNINGs with full tracebacks, which
# floods the log. Drop ONLY those records (keeps all other logs, incl. the Step-3b
# decoding line) by filtering the two loggers that emit them.
class _DropVizFailures(logging.Filter):
    def filter(self, record):  # noqa: A003
        return "visualization failed" not in record.getMessage().lower()


for _viz_logger in ("causalab.analyses.subspace._visualization",
                    "causalab.analyses.activation_manifold.fitting_pipeline"):
    logging.getLogger(_viz_logger).addFilter(_DropVizFailures())

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from context_datasets import create_contextual_prompts, create_global_prompts, create_counterfactual_dataset, _context_of
from causalab.tasks.natural_domains_arithmetic.causal_models import create_causal_model
from causalab.tasks.loader import load_task
from causalab.io.pipelines import load_pipeline
from causalab.analyses.activation_manifold.fitting_pipeline import ManifoldFittingConfig, run_manifold_fitting_pipeline
from causalab.neural.activations.collect import collect_features
from causalab.neural.units import InterchangeTarget
from causalab.neural.featurizer import Featurizer
from path_finder import find_path
from causalab.analyses.subspace.das import find_das_subspace
from causalab.analyses.subspace.pca import find_pca_subspace
from causalab.methods.metric import string_equality_checker, compute_reference_distributions
from causalab.runner.helpers import get_output_token_ids, resolve_intervention_metric
from results_layout import INTERVENED_REPS, PATCHED_DISTS, label_category_dir, site_context_dir
from collect_behaviour import build_behaviour_manifold, collect_patched_distributions


# combinations: label -> (contextual_vars, manifold_conditioning_vars, cf_conditioning_vars)
# contextual_vars        : how the base prompts are grouped when generating counterfactuals
#   cf_conditioning_vars   : how each counterfactual is built from its base (must be
#                            non-empty for context_matched / context_mismatched).
#   manifold_conditioning  : how the base prompts are grouped when fitting the manifold
#                            ([] = one global manifold).
def cyclic_embedding(values):
    # map an ordered cyclic variable to points on a circle: value -> [cos, sin].
    # Use as embeddings={intervention_variable: cyclic_embedding(causal_model.values[var])}.
    idx = {v: i for i, v in enumerate(values)}
    n = len(values)
    return lambda v: [math.cos(2 * math.pi * idx[v] / n), math.sin(2 * math.pi * idx[v] / n)]


# The class variable and the token position are 1:1 — a manifold over variable V
# is collected where V is encoded — so a combo carries a single `variable` and we
# derive the position from it.
POSITION_FOR = {"result": "last_token", "entity": "entity", "number": "number"}

# combinations: label -> dict with keys
#   variable   : the manifold's class variable (default "result"); sets the token
#                position via POSITION_FOR
#   cf_vars    : how each counterfactual is built from its base
#   m_vars     : how base prompts are grouped when fitting manifolds ([] = one global)
#   group_by   : optional {var: fn} passed to create_contextual_prompts to bucket a
#                conditioning var into coarse contexts (avoids 1-class-per-context)
#   categories : counterfactual types to run
# DEGENERACY RULE (single template, "{number} days after {entity}"): a non-trivial
# manifold needs (a) >=2 class values AND (b) within-class thickness from some FREE
# prompt variable that precedes the position and is neither `variable` nor fixed by
# `m_vars`. With only two free vars (entity, number), fixing one to condition removes
# a degree of freedom -> degenerate. So conditioning REQUIRES either group_by
# (bucket the conditioning var, keeping within-bucket variation) or extra templates.
# Each element of `categories` is a SEPARATE dataset (own path-finding, manifolds,
# reps, dists) saved under its own label__category dir — so the analysis can
# contrast the counterfactual types. Keep them single-type tuples; a multi-type
# tuple like ("context_matched","context_mismatched") would MIX both into one
# dataset (create_counterfactual_dataset emits one cf per type per base).
# cross_task adds a 3rd pass and needs related_models (set up below) — drop it
# from the list for a faster first run.
_CATEGORIES = [("context_matched", "context_mismatched"),("context_matched",), ("context_mismatched",), ("cross_task",)]
DEFAULT_COMBINATIONS = {
    "global_result": {"variable": "result", "cf_vars": ["entity"], "m_vars": [],
                      "categories": _CATEGORIES},
    "global_entity": {"variable": "entity", "cf_vars": ["entity"], "m_vars": [],
                      "categories": _CATEGORIES},
}


def set_seed(seed):
    """Seed every RNG the run touches: python `random` (DAS shuffles the training
    indices with it), numpy, and torch global (Gaussian `randn_like` and the DAS
    subspace init). The dataset builders in context_datasets use their OWN local
    random.Random(seed), so we also thread `seed` into them explicitly below —
    this call covers the torch/DAS side. NOTE: model forward passes on MPS/bf16
    are not bit-deterministic even seeded; prefer float32 (+ optionally
    torch.use_deterministic_algorithms) on the cluster for tighter reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataset_sequential(domain, num_examples, model_name, task_name, mode,
                              combinations=DEFAULT_COMBINATIONS, related_models=None,
                              device=None, dtype=None, num_sites=10, batch_size=32,
                              pipeline=None, layers=None, n_pathfind=None, seed=42):

    set_seed(seed)  # seeds torch/random for the Gaussian scan + DAS inside find_path
    causal_model = create_causal_model(NaturalDomainConfig(domain_type=domain))
    task = load_task(task_name, task_cfg=NaturalDomainConfig(domain_type=domain))
    if pipeline is None:
        pipeline = load_pipeline(model_name=model_name, task=task, max_new_tokens=1, device=device, dtype=dtype)
    num_per_context = num_examples // (12 if domain == "months" else 7)
    if related_models is None:
        related_models = [create_causal_model(NaturalDomainConfig(domain_type="months" if domain == "weekdays" else "weekdays"))]

    out = {}
    for label, combo in combinations.items():
        cf_cvars, m_cvars = combo["cf_vars"], combo["m_vars"]
        combo_position = POSITION_FOR[combo.get("variable", "result")]  # position derived from the class variable
        # base prompts: grouped contextual set if the manifold is conditioned, else global
        print(f"Creating base prompts for {label} (variable={combo.get('variable','result')}, m_vars={m_cvars}, position={combo_position})...")
        if m_cvars:
            grouped = create_contextual_prompts(causal_model, conditioning_vars=m_cvars,
                                                n_per_context=num_per_context, seed=seed,
                                                group_by=combo.get("group_by"))
            bases = [ex for prompts in grouped.values() for ex in prompts]
        else:
            bases = create_global_prompts(causal_model, n=num_examples, seed=seed)

        for category in combo["categories"]:
            cf = create_counterfactual_dataset(
                    causal_model, bases, cf_cvars, n=len(bases), seed=seed, categories=category,
                    related_models=related_models if "cross_task" in category else None,
                )
            # path-finding only RANKS sites, so a subset of counterfactuals suffices;
            # the full `cf` is still stored for manifold fitting + intervened reps.
            cf_pathfind = cf[:n_pathfind] if n_pathfind else cf
            sites = find_path(model_name, task_name, domain, cf_pathfind, mode, combo_position, device, dtype,
                                num_sites, batch_size, pipeline=pipeline, layers=layers,
                                output_dir=label_category_dir(INTERVENED_REPS, label, category))
            out[(label, category)] = (cf, sites)
    return out


def context_key(ex, manifold_cvars):
    # an example's manifold-context key; names the manifold output dirs.
    return tuple(sorted(_context_of(ex["input"], manifold_cvars).items())) if manifold_cvars else "global"


def _decoding_extras(data, pipeline, task, intervention_variable, batch_size):
    """Inputs the fitting pipeline's decoding validity test needs (tier C): the
    per-class reference OUTPUT distributions, score tokens, class count, and a KL
    comparison. ONLY meaningful when the manifold's class variable is the OUTPUT
    variable (result) — steering to the "Thursday" centroid should output Thursday.
    Returns {} (decoding stays off) on any failure, so an expensive run never dies
    for a validity metric. Verify on a pod smoke run before trusting the numbers."""
    try:
        if task.intervention_variable is None:
            task.intervention_variable = intervention_variable
        score_token_ids, n_classes = get_output_token_ids(task, pipeline)
        if score_token_ids is None:
            return {}
        ref_dists = compute_reference_distributions(
            dataset=data, score_token_ids=score_token_ids, n_classes=n_classes,
            example_to_class=task.intervention_value_index, pipeline=pipeline,
            batch_size=batch_size, score_token_index=0,
        )
        _string_metric, comparison_fn = resolve_intervention_metric("kl")
        return dict(skip_decoding_eval=False, score_token_ids=score_token_ids,
                    ref_dists=ref_dists, n_classes=n_classes, comparison_fn=comparison_fn,
                    score_variable_values={intervention_variable: task.intervention_values})
    except Exception as e:
        print(f"  [decoding test disabled for this fit] {type(e).__name__}: {e}")
        return {}


def generate_activation_manifold(target_sites, cf_dataset, manifold_cvars, pipeline, causal_model,
                                 batch_size=32, k_features=64, intervention_variable="result",
                                 embeddings=None, seed=42, task=None, decoding_eval=False):

    # group the flat counterfactual dataset by the base's context so we fit one
    # manifold per context (per site). Empty manifold_cvars -> one global group.
    groups = {}
    for ex in cf_dataset:
        groups.setdefault(context_key(ex, manifold_cvars), []).append(ex)

    # embeddings map the causal variable to intrinsic coords for parameter extraction /
    # visualization; default to the ones the NDA causal model already defines. Pass
    # {intervention_variable: cyclic_embedding(causal_model.values[intervention_variable])}
    # for the [cos, sin] circle instead of the 1D index embedding.
    if embeddings is None:
        embeddings = causal_model.embeddings

    manifolds = {}
    for context, data in groups.items():
        for site in target_sites:
            out_dir = site_context_dir(site.id, context)
            # Skip if this (site, context) manifold is already fit. Manifolds are
            # mode-independent (a PCA fit on the same data), so when STR and Gaussian
            # rank the SAME site we fit it once, not twice. (Also makes the whole
            # stage resumable if a run is interrupted.)
            if os.path.isdir(os.path.join(out_dir, "models")):
                print(f"  manifold already fit for {site.id} / {context} — skipping")
                continue
            site.set_featurizer(Featurizer())  # reset to identity so features are collected raw
            target = InterchangeTarget([[site]])
            # The manifold is ALWAYS fit in a PCA subspace — an intervention-INDEPENDENT
            # frame. DAS is a causal-mediation method; using it to define the manifold
            # would confound the off-manifold analysis, so DAS is applied only as a
            # treatment in get_intervened_representations, never here.
            features = find_pca_subspace(
                target=target,
                train_dataset=data,
                pipeline=pipeline,
                k_features=k_features,
                batch_size=batch_size,
                intervention_variable=intervention_variable,
                embeddings=embeddings,
            )["features"]
            # Save features + per-example class labels next to the manifold so the
            # health report can compute class separability (silhouette) offline (tier B).
            os.makedirs(out_dir, exist_ok=True)
            save_file({"features": features.detach().contiguous().cpu()},
                      os.path.join(out_dir, "features.safetensors"))
            with open(os.path.join(out_dir, "features_labels.json"), "w") as f:
                json.dump([str(ex["input"][intervention_variable]) for ex in data], f)
            # Decoding validity test (tier C) — only for the result manifold (class ==
            # output variable). Fail-safe: _decoding_extras returns {} on any error.
            extras = ({} if not (decoding_eval and task is not None and intervention_variable == "result")
                      else _decoding_extras(data, pipeline, task, intervention_variable, batch_size))
            config = ManifoldFittingConfig(
                pipeline=pipeline,
                interchange_target=target,
                features=features,
                train_dataset=data,
                causal_model=causal_model,
                output_dir=out_dir,
                k_features=k_features,
                batch_size=batch_size,
                intervention_variable=intervention_variable,
                embeddings=embeddings,
                periodic_info=causal_model.periods,
                seed=seed,
                **extras,
            )
            result = run_manifold_fitting_pipeline(config)
            manifolds[(site.id, context)] = result
            # Persist the decoding validity result (tier C) — causalab only returns
            # it in-memory / logs it, so save it next to the manifold for inspection.
            if "reconstruction_kl" in result:
                with open(os.path.join(out_dir, "decoding_eval.json"), "w") as f:
                    json.dump({"reconstruction_kl": result["reconstruction_kl"],
                               "reconstruction_kls_per_class": result.get("reconstruction_kls_per_class")},
                              f, indent=2)
    return manifolds


def get_intervened_representations(target_sites, counterfactual_dataset, pipeline, mode,
                                   causal_model=None, batch_size=32, sigma=3.0,
                                   k_features=64, output_dir=None):
    """
    h_intervened at each patched site, in raw d_model space (project through each
    context's manifold featurizer later). What the intervention writes depends on mode:
      STR      : full interchange   -> the counterfactual source's activation.
      Gaussian : noising            -> base + sigma * N(0, I) (the scan's replace_fn).
      DAS      : subspace interchange -> base with only the trained DAS-subspace
                 coords replaced by the source's: inverse_featurize(f_src, err_base).
    """
    cache = os.path.join(output_dir, f"{mode}_intervened_reps.safetensors") if output_dir else None
    if cache and os.path.exists(cache):
        return load_file(cache)

    for site in target_sites:
        site.set_featurizer(Featurizer())  # reset to identity -> raw d_model activations
    units = list(target_sites)

    def collect(inputs):
        ds = [{"input": x, "counterfactual_inputs": [x]} for x in inputs]
        return collect_features(dataset=ds, pipeline=pipeline, model_units=units, batch_size=batch_size)

    if mode == "STR":
        src = collect([ex["counterfactual_inputs"][0] for ex in counterfactual_dataset])
        reps = {s.id: src[s.id] for s in units}
    elif mode == "Gaussian":
        base = collect([ex["input"] for ex in counterfactual_dataset])
        reps = {s.id: base[s.id] + sigma * torch.randn_like(base[s.id]) for s in units}
    elif mode == "DAS":
        base = collect([ex["input"] for ex in counterfactual_dataset])
        src = collect([ex["counterfactual_inputs"][0] for ex in counterfactual_dataset])
        n_train = max(1, int(0.8 * len(counterfactual_dataset)))
        reps = {}
        for site in units:
            # trains the rotation and installs the SubspaceFeaturizer on the site
            find_das_subspace(
                target=InterchangeTarget([[site]]),
                train_dataset=counterfactual_dataset[:n_train],
                test_dataset=counterfactual_dataset[n_train:],
                pipeline=pipeline,
                causal_model=causal_model,
                k_features=k_features,
                batch_size=batch_size,
                output_dir=os.path.join(output_dir or ".", "das", site.id),
                metric=string_equality_checker,
            )
            with torch.no_grad():
                f_src, _ = site.featurizer.featurize(src[site.id])
                _, err_base = site.featurizer.featurize(base[site.id])
                reps[site.id] = site.featurizer.inverse_featurize(f_src, err_base)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    reps = {k: v.detach().contiguous().cpu() for k, v in reps.items()}
    if cache:
        os.makedirs(output_dir, exist_ok=True)
        save_file(reps, cache)
    return reps


def main():
    """
    - determine domain, model_name, num_examples, position, mode from arguments, run the path
      finding algorithm and then run data generation and activation manifold generation on the
      contextual and global datasets, and then collect intervened representations.
      Returns (intervened_representations, manifolds), both keyed by (label, category, mode).
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="weekdays")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--task-name", default="natural_domains_arithmetic")
    ap.add_argument("--num-examples", type=int, default=700)
    ap.add_argument("--mode", nargs="+", default=["STR", "Gaussian", "DAS"])
    ap.add_argument("--num-sites", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict the path-finding scan to these layers (default: all)")
    ap.add_argument("--n-pathfind", type=int, default=None,
                    help="subsample this many counterfactuals for path-finding (default: all)")
    ap.add_argument("--seed", type=int, default=42,
                    help="one seed for dataset generation + torch/DAS randomness; sweep it "
                         "across runs (set OFFDIST_RESULTS_DIR per seed so they don't overwrite)")
    ap.add_argument("--k-features", type=int, default=64,
                    help="PCA subspace dim for the manifold AND the DAS rotation (must match "
                         "between fit and intervention). 64 is the paper's value; use fewer if "
                         "num-examples is small (dead near-zero-variance dims blow up distances)")
    ap.add_argument("--decoding-eval", action="store_true",
                    help="run the functional decoding validity test (tier C): steer to each "
                         "result-centroid and score the output vs the per-class reference. "
                         "Adds a model pass; result-manifold only; fail-safe if it errors")
    args = ap.parse_args()

    # one model load, reused across path-finding, manifold fitting and intervened rep
    print("Loading model and pipeline...")
    causal_model = create_causal_model(NaturalDomainConfig(domain_type=args.domain))
    task = load_task(args.task_name, task_cfg=NaturalDomainConfig(domain_type=args.domain))
    pipeline = load_pipeline(model_name=args.model_name, task=task, max_new_tokens=1, device=args.device, dtype=args.dtype)

    # Behaviour manifold M_y (natural output distributions) — independent of cf/sites.
    print("Fitting behaviour manifold M_y...")
    build_behaviour_manifold(domain=args.domain, model_name=args.model_name,
                             device=args.device, dtype=args.dtype, seed=args.seed)

    print("Pipeline and Model loaded")
    # SINGLE dataset + path-finding pass; both the activation side (manifolds,
    # intervened reps) and the behaviour side (patched output dists) consume the
    # SAME (cf, sites), so the two analyses can never diverge and path-finding
    # runs once instead of twice.
    datasets = create_dataset_sequential(
        args.domain, args.num_examples, args.model_name, args.task_name,
        args.mode, DEFAULT_COMBINATIONS, None, args.device, args.dtype, args.num_sites,
        pipeline=pipeline, layers=args.layers, n_pathfind=args.n_pathfind, seed=args.seed,
    )

    manifolds, intervened, output_dists = {}, {}, {}
    for (label, category), (cf, sites_by_mode) in datasets.items():
        print(f"Generating manifolds, intervened reps and patched dists for {label}/{category}...")
        print("Sites:", [s.id for s in sites_by_mode["STR"]])
        combo = DEFAULT_COMBINATIONS[label]
        m_cvars = combo["m_vars"]
        for mode_name, sites in sites_by_mode.items():
            if mode_name != "DAS":  # manifolds are mode-independent; DAS sites ⊆ STR∪Gaussian
                manifolds[(label, category, mode_name)] = generate_activation_manifold(
                    sites, cf, m_cvars, pipeline, causal_model, k_features=args.k_features,
                    intervention_variable=combo.get("variable", "result"), seed=args.seed,
                    task=task, decoding_eval=args.decoding_eval)
            # get_intervened_representations FIRST: for DAS it trains + installs the
            # subspace featurizer on the site objects, which collect_patched_distributions
            # then reuses (so DAS is trained once and both sides use the same rotation).
            intervened[(label, category, mode_name)] = get_intervened_representations(
                sites, cf, pipeline, mode_name, causal_model=causal_model,
                k_features=args.k_features,
                output_dir=label_category_dir(INTERVENED_REPS, label, category))
            output_dists[(label, category, mode_name)] = collect_patched_distributions(
                sites, cf, pipeline, task, mode_name, causal_model=causal_model,
                k_features=args.k_features,
                output_dir=label_category_dir(PATCHED_DISTS, label, category))
    return intervened, manifolds, output_dists


if __name__ == "__main__":
    main()
