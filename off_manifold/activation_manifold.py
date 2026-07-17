
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # linalg.householder_product not implemented on MPS
import math

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
from causalab.methods.metric import string_equality_checker


# combinations: label -> (cf_conditioning_vars, manifold_conditioning_vars, categories)
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


DEFAULT_COMBINATIONS = {
    "global": (["entity"], [], [("context_matched", "context_mismatched")]),
    # "entity": (["entity"], ["entity"], ("context_matched", "context_mismatched")),
    # "number": (["number"], ["number"], ("context_matched", "context_mismatched")),
}


def create_dataset_sequential(domain, num_examples, model_name, task_name, mode, position,
                              combinations=DEFAULT_COMBINATIONS, related_models=None,
                              device=None, dtype=None, num_sites=10, batch_size=32,
                              pipeline=None, layers=None, n_pathfind=None):

    causal_model = create_causal_model(NaturalDomainConfig(domain_type=domain))
    task = load_task(task_name, task_cfg=NaturalDomainConfig(domain_type=domain))
    if pipeline is None:
        pipeline = load_pipeline(model_name=model_name, task=task, max_new_tokens=1, device=device, dtype=dtype)
    num_per_context = num_examples // (12 if domain == "months" else 7)
    if related_models is None:
        related_models = [create_causal_model(NaturalDomainConfig(domain_type="months" if domain == "weekdays" else "weekdays"))]

    out = {}
    for label, (cf_cvars, m_cvars, categories) in combinations.items():
        # base prompts: grouped contextual set if the manifold is conditioned, else global
        print(f"Creating base prompts for {label} (cf_cvars={cf_cvars}, m_cvars={m_cvars})...")
        if m_cvars:
            grouped = create_contextual_prompts(causal_model, conditioning_vars=m_cvars, n_per_context=num_per_context)
            bases = [ex for prompts in grouped.values() for ex in prompts]
        else:
            bases = create_global_prompts(causal_model, n=num_examples)

        for category in categories:
            cf = create_counterfactual_dataset(
                    causal_model, bases, cf_cvars, n=len(bases), categories=category,
                    related_models=related_models if "cross_task" == category else None,
                )
            # path-finding only RANKS sites, so a subset of counterfactuals suffices;
            # the full `cf` is still stored for manifold fitting + intervened reps.
            cf_pathfind = cf[:n_pathfind] if n_pathfind else cf
            sites = find_path(model_name, task_name, domain, cf_pathfind, mode, position, device, dtype,
                                num_sites, batch_size, pipeline=pipeline, layers=layers,
                                output_dir=f"path_finder_results/{label}/{category}")
            out[(label, category)] = (cf, sites)
    return out


def generate_activation_manifold(target_sites, cf_dataset, manifold_cvars, pipeline, mode, causal_model,
                                 batch_size=32, k_features=64, intervention_variable="result", embeddings=None):

    # group the flat counterfactual dataset by the base's context so we fit one
    # manifold per context (per site). Empty manifold_cvars -> one global group.
    groups = {}
    for ex in cf_dataset:
        key = tuple(sorted(_context_of(ex["input"], manifold_cvars).items())) if manifold_cvars else "global"
        groups.setdefault(key, []).append(ex)

    # embeddings map the causal variable to intrinsic coords for parameter extraction /
    # visualization; default to the ones the NDA causal model already defines. Pass
    # {intervention_variable: cyclic_embedding(causal_model.values[intervention_variable])}
    # for the [cos, sin] circle instead of the 1D index embedding.
    if embeddings is None:
        embeddings = causal_model.embeddings

    manifolds = {}
    for context, data in groups.items():
        for site in target_sites:
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
            config = ManifoldFittingConfig(
                pipeline=pipeline,
                interchange_target=target,
                features=features,
                train_dataset=data,
                causal_model=causal_model,
                output_dir=f"manifold_results/{site.id}/{context}/{mode}",
                k_features=k_features,
                batch_size=batch_size,
                intervention_variable=intervention_variable,
                embeddings=embeddings,
                periodic_info=causal_model.periods,
            )
            manifolds[(site.id, context)] = run_manifold_fitting_pipeline(config)
    return manifolds


def get_intervened_representations(target_sites, counterfactual_dataset, pipeline, batch_size=32):
    # h_intervened at a patched site = the counterfactual source's activation there,
    # i.e. the value STR interchange writes in. All sites in a single forward pass.
    sources = [
        {"input": ex["counterfactual_inputs"][0], "counterfactual_inputs": [ex["counterfactual_inputs"][0]]}
        for ex in counterfactual_dataset
    ]
    for site in target_sites:
        site.set_featurizer(Featurizer())  # reset to identity -> raw h_intervened (project later per context)
    feats = collect_features(dataset=sources, pipeline=pipeline, model_units=list(target_sites), batch_size=batch_size)
    return {site.id: feats[site.id].detach() for site in target_sites}


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
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--mode", nargs="+", default=["STR", "Gaussian", "DAS"])
    ap.add_argument("--num-sites", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="restrict the path-finding scan to these layers (default: all)")
    ap.add_argument("--n-pathfind", type=int, default=None,
                    help="subsample this many counterfactuals for path-finding (default: all)")
    args = ap.parse_args()

    # one model load, reused across path-finding, manifold fitting and intervened rep
    print("Loading model and pipeline...")
    causal_model = create_causal_model(NaturalDomainConfig(domain_type=args.domain))
    task = load_task(args.task_name, task_cfg=NaturalDomainConfig(domain_type=args.domain))
    pipeline = load_pipeline(model_name=args.model_name, task=task, max_new_tokens=1, device=args.device, dtype=args.dtype)

    print("Pipeline and Model loaded")
    datasets = create_dataset_sequential(
        args.domain, args.num_examples, args.model_name, args.task_name,
        args.mode, args.position, DEFAULT_COMBINATIONS, None, args.device, args.dtype, args.num_sites,
        pipeline=pipeline, layers=args.layers, n_pathfind=args.n_pathfind,
    )

    manifolds, intervened = {}, {}
    for (label, category), (cf, sites_by_mode) in datasets.items():
        print(f"Generating manifolds and intervened representations for {label}/{category}...")
        print("Sites:", [s.id for s in sites_by_mode["STR"]])
        print("Sites:", [s.id for s in sites_by_mode["Gaussian"]])
        m_cvars = DEFAULT_COMBINATIONS[label][1]
        for mode_name, sites in sites_by_mode.items():
            manifolds[(label, category, mode_name)] = generate_activation_manifold(sites, cf, m_cvars, pipeline, mode_name, causal_model)
            intervened[(label, category, mode_name)] = get_intervened_representations(sites, cf, pipeline)
    return intervened, manifolds


if __name__ == "__main__":
    main()
