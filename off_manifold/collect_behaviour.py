"""
Behaviour-side collection for the off-distribution analysis (the paper's E_BC).

Two stages, both maximal reuse of causalab:

1. collect_patched_distributions — the patched-run OUTPUT distributions.
   Instead of patching stored reps back in, we re-run the actual interventions
   with output_scores=True (the reps saved by get_intervened_representations
   are exactly what these interventions write, so the outputs are identical):
     STR      -> run_interchange_interventions        (full swap at the site)
     Gaussian -> run_interpolation_interventions      (same add-noise replace_fn
                                                       as path_finder's scan)
     DAS      -> run_interchange_interventions with the trained SubspaceFeaturizer
                 installed (interchange through a featurizer IS the DAS patch)
   Scores -> concept-slice probabilities via get_output_token_ids +
   class_probabilities(full_vocab_softmax=True) — the identical conversion
   output_manifold's _build_belief_artifacts uses for the natural dists.
   Saved to manifold_results/{label}/{category}/{mode}_output_dists.safetensors
   as {site_id: (N, W)}; sums <= 1 per row, as compute_score expects.

2. build_behaviour_manifold — the behaviour (belief) manifold M_y.
   Just runs causalab's output_manifold analysis (collects the model's natural
   output distributions, fits the Hellinger-space spline), with the cfg composed
   from the SHIPPED yamls (configs/task/natural_domains_arithmetic_{domain}.yaml
   + configs/analysis/output_manifold.yaml). Artifacts land under
   {experiment_root}/output_manifold/spline_s0.0/ and are reloaded later by
   belief_fit.load_output_manifold for compute_score_single_path.
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file

import causalab
from causalab.analyses.output_manifold.main import main as output_manifold_main
from causalab.analyses.subspace.das import find_das_subspace
from causalab.methods.metric import class_probabilities, string_equality_checker
from causalab.methods.trained_subspace.subspace import SubspaceFeaturizer
from causalab.neural.activations.interchange_mode import run_interchange_interventions
from causalab.neural.activations.interpolate import run_interpolation_interventions
from causalab.neural.featurizer import Featurizer
from causalab.neural.units import InterchangeTarget
from causalab.runner.helpers import get_output_token_ids
from results_layout import BEHAVIOUR_MANIFOLD

_CONFIGS = os.path.join(os.path.dirname(causalab.__file__), "configs")


def _concept_probs(raw, score_token_ids):
    """raw["scores"]: per batch, a per-step list of (B, V) logits. Take the last
    step and convert to (B, W) concept-slice probabilities — same as
    _build_belief_artifacts does for the natural distributions."""
    probs = [
        class_probabilities(batch_scores[-1], score_token_ids, full_vocab_softmax=True)
        for batch_scores in raw["scores"]
    ]
    return torch.cat(probs).float()


def collect_patched_distributions(target_sites, counterfactual_dataset, pipeline, task, mode,
                                  causal_model=None, batch_size=32, sigma=3.0,
                                  k_features=64, output_dir=None):
    """Patched-run output distributions per site: {site_id: (N, W)}. Cached to
    {output_dir}/{mode}_output_dists.safetensors."""
    cache = os.path.join(output_dir, f"{mode}_output_dists.safetensors") if output_dir else None
    if cache and os.path.exists(cache):
        return load_file(cache)

    score_token_ids, _ = get_output_token_ids(task, pipeline)

    dists = {}
    for site in target_sites:
        if mode == "DAS":
            # interchange through a trained SubspaceFeaturizer IS the DAS patch.
            # Reuse the featurizer if get_intervened_representations already
            # trained it on this site object; otherwise train it here.
            if not isinstance(site.featurizer, SubspaceFeaturizer):
                n_train = max(1, int(0.8 * len(counterfactual_dataset)))
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
        else:
            site.set_featurizer(Featurizer())  # identity -> raw-activation intervention

        target = InterchangeTarget([[site]])
        if mode == "Gaussian":
            def add_noise(f_base, f_src, **_):
                # same GN definition as path_finder.run_gaussian_scan: N(0, sigma*std(base))
                return f_base + sigma * f_base.std() * torch.randn_like(f_base)
            raw = run_interpolation_interventions(
                pipeline=pipeline, counterfactual_dataset=counterfactual_dataset,
                interchange_target=target, fn=add_noise, params={},
                batch_size=batch_size, output_scores=True,
            )
        else:  # STR and DAS are both interchange; the featurizer decides which
            raw = run_interchange_interventions(
                pipeline=pipeline, counterfactual_dataset=counterfactual_dataset,
                interchange_target=target, batch_size=batch_size, output_scores=True,
            )
        dists[site.id] = _concept_probs(raw, score_token_ids)

    if cache:
        os.makedirs(output_dir, exist_ok=True)
        save_file({k: v.contiguous() for k, v in dists.items()}, cache)
    return dists


def build_behaviour_manifold(domain="weekdays", model_name="Qwen/Qwen2.5-3B-Instruct",
                             device=None, dtype=None, experiment_root=BEHAVIOUR_MANIFOLD,
                             seed=42, batch_size=32,target_variable="result"):
    """Fit the behaviour (belief) manifold M_y by running causalab's
    output_manifold analysis with the shipped configs. Idempotent: the natural
    distributions are cached, and the fit artifacts land under
    {experiment_root}/output_manifold/spline_s0.0/."""
    cfg = OmegaConf.create({
        "experiment_root": experiment_root,
        "seed": seed,
        "model": {"name": model_name, "id": model_name, "device": device, "dtype": dtype},
        "task": OmegaConf.load(os.path.join(_CONFIGS, "task", f"natural_domains_arithmetic_{domain}.yaml")),
        "output_manifold": OmegaConf.load(os.path.join(_CONFIGS, "analysis", "output_manifold.yaml")),
    })
    # main() reads cfg.task.get("target_variable") (the shipped runner configs set
    # task.target_variable); the task yaml alone doesn't define it, so set it here.
    cfg.task.target_variable = target_variable
    cfg.output_manifold.batch_size = batch_size
    return output_manifold_main(cfg)


def main():
    """Standalone fit of the behaviour manifold M_y only. The patched output
    distributions are collected by activation_manifold.main (which shares its
    single (cf, sites) pass with the activation side), so there is no separate
    collection loop here — this entry point exists just to (re)fit M_y on the
    cluster, e.g. once before a seed sweep."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="weekdays")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Fitting behaviour manifold M_y (natural output distributions)...")
    return build_behaviour_manifold(domain=args.domain, model_name=args.model_name,
                                    device=args.device, dtype=args.dtype, seed=args.seed)


if __name__ == "__main__":
    main()
