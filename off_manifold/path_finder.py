

from typing import List
import json
import os
import torch

from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.tasks.loader import load_task
from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig
from causalab.io.pipelines import load_pipeline
from causalab.runner.helpers import build_targets_for_grid
from causalab.methods.interchange.layer_scan import run_layer_scan
from causalab.methods.metric import make_causal_metric, score_intervention_outputs
from causalab.neural.activations.interpolate import run_interpolation_interventions


def _top_sites(scores, targets, num_sites):
    # scores: {(layer, pos): float}; return the num_sites highest-scoring units.
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:num_sites]
    return [targets[key].flatten()[0] for key in ranked]


def _rank_and_save(scores, targets, num_sites, method, output_dir=None):
    # scores: {(layer, pos): float} from run_layer_scan / run_gaussian_scan (already the
    # interchange-exchange accuracy per site). Save the top-`num_sites` (layer, position,
    # score) for `method` to disk, then return the corresponding units.
    ranked = sorted(scores, key=lambda k: scores[k], reverse=True)[:num_sites]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        payload = [
            {"layer": key[0], "position": key[1], "score": float(scores[key])}
            for key in ranked
        ]
        with open(os.path.join(output_dir, f"{method}_top_sites.json"), "w") as f:
            json.dump(payload, f, indent=2)
    return [targets[key].flatten()[0] for key in ranked]


def run_gaussian_scan(targets, dataset, pipeline, batch_size, causal_model, sigma=3.0):
    # Zhang-style noising: add Gaussian noise to the raw activation at the site
    # (identity featurizer), then score how often the answer still matches. A site
    # ENCODES the variable when noising it BREAKS the answer, so importance = 1 - acc.
    def add_noise(f_base, f_src, **_):
        return f_base + sigma * torch.randn_like(f_base)

    raw_results = {}
    for key, target in targets.items():
        raw_results[key] = run_interpolation_interventions(
            pipeline=pipeline,
            counterfactual_dataset=dataset,
            interchange_target=target,
            fn=add_noise,
            params={},
            batch_size=batch_size,
            output_scores=True,
        )
    acc = score_intervention_outputs(
        raw_results=raw_results,
        dataset=dataset,
        metric=make_causal_metric(),
        causal_model=causal_model,
    )
    return {k: 1.0 - v for k, v in acc.items()}


def find_path(
    model_name: str,
    task_name: str,
    domain: str,
    counterfactual_examples: List[CounterfactualExample],
    mode: list[str],  # any of "STR", "Gaussian", "DAS"
    position: str,
    device: str | None,
    dtype: str | None,
    num_sites: int,
    batch_size: int,
    pipeline=None,
    layers: list[int] | None = None,
    output_dir: str | None = None,
):
    """
    DAS : uses the union of the STR and Gaussian sites (subspaces are learned there).
    STR / Gaussian : return the top-num_sites (layer, comp) where `position` is encoded.
    Returns dict[mode -> list[AtomicModelUnit]]. Pass `pipeline` to reuse a loaded
    model across calls; pass `layers` to scan a subset instead of every layer.
    """
    task = load_task(task_name, task_cfg=NaturalDomainConfig(domain_type=domain))
    if pipeline is None:
        pipeline = load_pipeline(
            model_name=model_name, task=task, max_new_tokens=1, device=device, dtype=dtype,
        )
    if layers is None:
        layers = list(range(pipeline.get_num_layers()))
    targets, _positions = build_targets_for_grid(pipeline, task, layers, [position])
    metric = make_causal_metric()  # default: string-equality check vs causal label

    # Compute each scan at most once, even when DAS needs both.
    str_sites = gauss_sites = None
    if "STR" in mode or "DAS" in mode:
        scores = run_layer_scan(
            interchange_targets=targets,
            dataset=counterfactual_examples,
            pipeline=pipeline,
            batch_size=batch_size,
            metric=metric,
            causal_model=task.causal_model,
        )
        str_sites = _rank_and_save(scores, targets, num_sites, "STR", output_dir)
    if "Gaussian" in mode or "DAS" in mode:
        scores = run_gaussian_scan(
            targets, counterfactual_examples, pipeline, batch_size, task.causal_model
        )
        gauss_sites = _rank_and_save(scores, targets, num_sites, "Gaussian", output_dir)

    sites: dict[str, list] = {}
    if "STR" in mode:
        sites["STR"] = str_sites
    if "Gaussian" in mode:
        sites["Gaussian"] = gauss_sites
    if "DAS" in mode:
        # union, deduped: a site top-ranked by both scans appears once (else it
        # would be DAS-trained twice downstream)
        seen = {s.id for s in str_sites}
        sites["DAS"] = str_sites + [s for s in gauss_sites if s.id not in seen]
    return sites


