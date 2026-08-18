

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


# patch_at -> (mlp site, attn site). "corr" (default) = where hl_ll_corr stores each
# variable and where dormancy is a representation-level signal; "write" = residual
# writes (d_model, consistent, for the direction analysis); "internal" = attn hook_z.
_PATCH_SITES = {
    "corr": ("mlp_activation", "attn_result"),
    "write": ("mlp_output", "attn_result"),
    "internal": ("mlp_activation", None),
}


def build_grid(pipeline, task, component, layers, positions=None, heads=None, patch_at="corr"):
    """Guard: build a (layer x position/head) grid of interchange targets of ONE
    component type. Generalises build_targets_for_grid (which is residual-only).

        component : "residual" | "mlp" | "attention"
        patch_at  : "corr" (default) -> mlp.hook_post + attn.hook_result (correspondence)
                    "write"          -> mlp_output   + attn.hook_result (residual writes)
                    "internal"       -> mlp.hook_post + attn.hook_z

    Returns {key -> InterchangeTarget} (keys (layer,pos_id) or (layer,head)), same
    shape build_targets_for_grid / the scans expect.
    """
    from causalab.neural.activations.targets import (
        build_residual_stream_targets,
        build_mlp_targets,
        build_attention_head_targets,
    )

    mlp_loc, attn_loc = _PATCH_SITES[patch_at]
    tp_lookup = task.create_token_positions(pipeline)
    token_positions = (
        [tp_lookup[n] for n in positions] if positions else list(tp_lookup.values())
    )

    if component == "residual":
        return build_residual_stream_targets(
            pipeline, layers, token_positions,
            mode="one_target_per_unit", target_output=(patch_at != "internal"),
        )
    if component == "mlp":
        return build_mlp_targets(
            pipeline, layers, token_positions, mode="one_target_per_unit", location=mlp_loc,
        )
    if component == "attention":
        if heads is None:
            heads = list(range(int(pipeline.model.cfg.n_heads)))
        return build_attention_head_targets(
            pipeline, layers, heads, token_positions[0], mode="one_target_per_unit",
            location=attn_loc,
        )
    raise ValueError(f"unknown component {component!r} (residual|mlp|attention)")


def build_full_grid(pipeline, task, layers, positions=None, patch_at="corr",
                    include_residual=True):
    """The WHOLE patchable universe at circuit granularity, keyed by ``site.id``.

    One grid over ALL components — every MLP (per layer) + every attention head (per
    layer x head) + [optional] BOTH residual streams per layer — instead of one component
    type per run. Two reasons:

    * dormancy is scored by a SINGLE top-k ranking across the whole model, so every
      component must be ranked together (per-component runs can't share a cutoff), and the
      full universe gives the per-(mode,context) correlation a usable N.
    * matching hl_ll_corr's FULL spec set: the IOI correspondence maps to heads, an MLP,
      AND both residual streams (all_nodes_hook -> resid_pre = block_input; hook_out ->
      resid_post = block_output). So residual is built in BOTH directions (target_output
      False and True), independent of ``patch_at``.

    Keyed by ``site.id`` (unique across component types), the shape compute_patching_scores,
    the reps, and the membership map all key on downstream.
    """
    from causalab.neural.activations.targets import build_residual_stream_targets

    grid: dict = {}

    def _add(sub):
        for target in sub.values():
            grid[target.flatten()[0].id] = target

    _add(build_grid(pipeline, task, "mlp", layers, positions=positions, patch_at=patch_at))
    _add(build_grid(pipeline, task, "attention", layers, positions=positions, patch_at=patch_at))
    if include_residual:
        tp_lookup = task.create_token_positions(pipeline)
        token_positions = (
            [tp_lookup[n] for n in positions] if positions else list(tp_lookup.values())
        )
        # both residual sites, so the full correspondence matches (resid_pre=block_input,
        # resid_post=block_output). _site_to_spec keeps the two distinct.
        for target_output in (False, True):
            _add(build_residual_stream_targets(
                pipeline, layers, token_positions,
                mode="one_target_per_unit", target_output=target_output,
            ))
    return grid


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
        # GN baseline: noise ~ N(0, sigma*std(base)) — a scalar std over the
        # activations ("3 times the standard deviation of the ... from the
        # textset"), NOT a flat sigma. f_src is unused: noising ignores the source.
        return f_base + sigma * f_base.std() * torch.randn_like(f_base)

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

def compute_patching_scores(grid, cf_dataset, pipeline, modes, batch_size, target_pos=-1,
                            metric="flip"):
    """Per-site 'apparent circuit' score, tracr-native. ``metric`` chooses the readout:

      'flip' (default, coarse/binary):
        STR      -> fraction where patching flips the argmax answer to the SOURCE's result.
        Gaussian -> fraction where noising BREAKS the base answer (argmax != base).
        Cheap, but insensitive to graded effects — it under-reports circuits with backup /
        redundancy (e.g. IOI), because a component can shift the margin without flipping argmax.

      'logit_diff' (continuous, standard IOI-style):
        the causal effect on the logit gap LD = logit[base_result] - logit[src_result]:
        score = mean_i (LD_clean_i - LD_patched_i). Patching a circuit component erodes the
        base's preference for its OWN answer over the source's, so a high score = in-circuit.
        Needs one clean (unpatched) pass. Requires INTEGER result indices (IOI token ids /
        class indices); for label-valued tracr results use 'flip'.

    High score = the method calls the site in-circuit; high at a site NOT in hl_ll_corr = a
    dormancy false positive. Returns {mode: {key -> score}}."""
    from causalab.neural.activations.interchange_mode import run_interchange_interventions
    from causalab.neural.featurizer import Featurizer

    # the interchange must swap RAW activations — reset any fitted manifold/subspace
    # featurizer left on the sites (by generate_activation_manifold) to identity, else
    # featurize() runs the slab through the spline manifold and shape-mismatches.
    for target in grid.values():
        for unit in target.flatten():
            unit.set_featurizer(Featurizer())

    base_r = [ex["input"]["result"] for ex in cf_dataset]
    src_r = [ex["counterfactual_inputs"][0]["result"] for ex in cf_dataset]

    if metric == "logit_diff":
        return _logit_diff_scores(grid, cf_dataset, pipeline, modes, batch_size,
                                  target_pos, base_r, src_r)
    if metric != "flip":
        raise ValueError(f"unknown metric {metric!r} (flip | logit_diff)")

    def _answers(raw):
        # raw["string"]: per-batch list; each batch = list of per-example decoded
        # (each decoded = per-position label list). Flatten, take target_pos.
        out = []
        for batch in raw.get("string", []):
            for ex in (batch if isinstance(batch, list) else [batch]):
                out.append(ex[target_pos] if isinstance(ex, (list, tuple)) else ex)
        return out

    def add_noise(f_base, f_src, **_):
        return f_base + 3.0 * f_base.std() * torch.randn_like(f_base)

    scores: dict = {}
    for mode in modes:
        scores[mode] = {}
        for key, target in grid.items():
            if mode == "STR":
                raw = run_interchange_interventions(
                    pipeline, cf_dataset, target, batch_size=batch_size, output_scores=False)
                ans = _answers(raw)
                scores[mode][key] = sum(a == s for a, s in zip(ans, src_r)) / max(1, len(ans))
            elif mode == "Gaussian":
                raw = run_interpolation_interventions(
                    pipeline, cf_dataset, target, fn=add_noise, params={},
                    batch_size=batch_size, output_scores=False)
                ans = _answers(raw)
                scores[mode][key] = sum(a != b for a, b in zip(ans, base_r)) / max(1, len(ans))
    return scores


def _logit_diff_scores(grid, cf_dataset, pipeline, modes, batch_size, target_pos, base_r, src_r):
    """Continuous IOI-style logit-difference apparent-circuit score (see compute_patching_scores).
    score[mode][site] = mean_i (LD_clean_i - LD_patched_i), LD = logit[base_result] - logit[src_result]."""
    from causalab.neural.activations.interchange_mode import run_interchange_interventions

    def _col(v):  # result value -> logit column (IOI token id / class index used directly)
        if isinstance(v, bool) or not isinstance(v, (int,)) and not hasattr(v, "__index__"):
            raise ValueError(
                f"metric='logit_diff' needs INTEGER result indices, got {v!r}. Use metric='flip' "
                f"for label-valued (tracr) results, or map labels to logit columns first.")
        return int(v)

    base_col = torch.tensor([_col(v) for v in base_r])
    src_col = torch.tensor([_col(v) for v in src_r])
    idx = torch.arange(len(cf_dataset))

    def _logits_at(score_batches):
        # list of per-batch (b, pos, vocab) -> (N, vocab) at target_pos
        L = torch.cat([s if torch.is_tensor(s) else torch.as_tensor(s) for s in score_batches], 0)
        return L[:, target_pos, :].float()

    def _ld(L):
        return L[idx, base_col] - L[idx, src_col]

    # clean (unpatched) logits, batched over the base inputs
    base_inputs = [ex["input"] for ex in cf_dataset]
    clean = [pipeline.generate(base_inputs[i:i + batch_size])["scores"]
             for i in range(0, len(base_inputs), batch_size)]
    LD_clean = _ld(_logits_at(clean))

    def add_noise(f_base, f_src, **_):
        return f_base + 3.0 * f_base.std() * torch.randn_like(f_base)

    scores: dict = {}
    for mode in modes:
        scores[mode] = {}
        for key, target in grid.items():
            if mode == "STR":
                raw = run_interchange_interventions(
                    pipeline, cf_dataset, target, batch_size=batch_size, output_scores=True)
            elif mode == "Gaussian":
                raw = run_interpolation_interventions(
                    pipeline, cf_dataset, target, fn=add_noise, params={},
                    batch_size=batch_size, output_scores=True)
            else:
                raise ValueError(f"logit_diff supports STR|Gaussian, not {mode!r}")
            LD_patched = _ld(_logits_at(raw["scores"]))
            scores[mode][key] = float((LD_clean - LD_patched).mean())
    return scores


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


