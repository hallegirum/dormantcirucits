"""
interpbench_loading.py
======================
Glue between InterpBench's `circuits_benchmark` loader and our stack:

  1. build_tracr_pipeline  — wire a loaded HookedTracrTransformer's OWN encoders
     into a TracrPipeline (the "rewire": encode/decode come from the model).
  2. dump_correspondence   — print hl_ll_corr's structure so you can confirm the
     TracrHLNode / LLNode / TorchIndex field names on the pod BEFORE trusting (3).
  3. partition_targets     — split the model's components into CIRCUIT vs
     NON-CIRCUIT AtomicModelUnits (using causalab's LM_units), driven by
     hl_ll_corr. These are the two axes you patch along for the dormancy study.

Requires the circuits_benchmark loading path (NOT the bare-checkpoint path in
spike_minimal.py): only that path produces the encoders and the correspondence.

    from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader
    loader = get_ll_model_loader(task, interp_bench=True)
    hl_ll_corr, model = loader.load_ll_model_and_correspondence(device="cpu")
    pipeline, bos = build_tracr_pipeline(model)
    dump_correspondence(hl_ll_corr)                       # confirm fields first
    token_pos = interpbench_task.create_token_positions(pipeline)["last_token"]
    circuit, noncircuit = partition_targets(hl_ll_corr, model, token_pos)
"""

from __future__ import annotations

import re
from typing import Any, Callable

from causalab.neural.featurizer import Featurizer
from causalab.neural.LM_units import ResidualStream, MLP, AttentionHead
from causalab.neural.units import AtomicModelUnit

from tracr_pipeline import TracrPipeline


# --------------------------------------------------------------------------- #
#  1. Rewire: pull encode/decode from the model's own tracr encoders
# --------------------------------------------------------------------------- #
def build_pipelines(ll_model, case, device: str = "cpu"):
    """Wrap BOTH the trained LL model and the reference HL model as TracrPipelines,
    and return a model-derived ``hl_fn`` (the ground truth — no hand-written algo).

    Returns ``(ll_pipeline, hl_pipeline, bos, hl_fn)`` where:
      - ll_pipeline runs interventions on the trained low-level model
      - hl_pipeline runs the compiled high-level tracr program (case.get_hl_model())
      - hl_fn(seq) -> per-position output list, by running the HL program
    Both pipelines share the tracr encoders (SIIT trains the LL model into the HL
    model's encoding), so LL-vs-HL comparison cancels any BOS/decode convention.

    Note the decode does NOT pre-strip BOS: the tracr output decoder already drops
    the leading BOS-output, so ``decode(full_argmax_row)`` yields exactly seq_len
    per-position labels.
    """
    hl_model = case.get_hl_model(device=device)  # HookedTracrTransformer (carries encoders)
    enc = hl_model.tracr_input_encoder
    dec = hl_model.tracr_output_encoder

    def encode(raw_input):
        return enc.encode(list(raw_input))  # raw_input already includes BOS

    def decode(ids):
        return dec.decode(list(ids))  # tracr decoder strips BOS itself -> seq_len labels

    ll_pipeline = TracrPipeline(ll_model, encode=encode, decode=decode, device=device)
    hl_pipeline = TracrPipeline(hl_model, encode=encode, decode=decode, device=device)
    bos = enc.bos_token

    def hl_fn(seq):
        out = hl_pipeline.generate([{"raw_input": [bos] + list(seq)}])["string"]
        return list(out)

    return ll_pipeline, hl_pipeline, bos, hl_fn


# --------------------------------------------------------------------------- #
#  2. Inspector — confirm the real field names before trusting (3)
# --------------------------------------------------------------------------- #
def dump_correspondence(hl_ll_corr, limit: int = 12) -> None:
    """Print hl_ll_corr's structure: HL node fields + the LL nodes it maps to.

    Run this on the pod once. It tells you whether LLNode exposes .name/.index
    (assumed below) or different attribute names, and how TorchIndex prints.
    """
    print(f"hl_ll_corr: {type(hl_ll_corr).__name__}, {len(hl_ll_corr)} entries\n")
    for i, (hl, lls) in enumerate(hl_ll_corr.items()):
        if i >= limit:
            print(f"... (+{len(hl_ll_corr) - limit} more)")
            break
        print(f"HL  name={getattr(hl,'name',hl)!r:40s} label={getattr(hl,'label','?')!r} "
              f"index={getattr(hl,'index','?')!r}")
        for ll in (lls if hasattr(lls, "__iter__") else [lls]):
            print(f"    LL  name={getattr(ll,'name',ll)!r} index={getattr(ll,'index','?')!r}")


# --------------------------------------------------------------------------- #
#  3. hl_ll_corr -> circuit / non-circuit AtomicModelUnits (LM_units)
# --------------------------------------------------------------------------- #
# A "component spec" is (layer, component_type, head|None) — the granularity at
# which circuit membership is defined (per-head for attention, per-layer for MLP).
ComponentSpec = tuple[int, str, "int | None"]


def _head_from_index(index: Any) -> int | None:
    """Head from a TorchIndex like [:, :, h, :] (head slot = position 2).
    VERIFY against dump_correspondence output — the TorchIndex API varies."""
    idx = getattr(index, "as_index", index)
    try:
        h = idx[2]
        return int(h) if isinstance(h, int) else None
    except (TypeError, IndexError):
        return None


def _hook_to_spec(hook: str, index: Any) -> ComponentSpec | None:
    """Map a TL hook name to a component spec, or None if it isn't a patch site."""
    m = re.match(r"blocks\.(\d+)\.(.+)", hook)
    if not m:
        return None
    layer, rest = int(m.group(1)), m.group(2)
    if rest == "mlp.hook_post":
        return (layer, "mlp_activation", None)
    if rest == "hook_mlp_out":
        return (layer, "mlp_output", None)
    if rest == "attn.hook_result":  # InterpBench correspondences use this (post-W_O)
        return (layer, "attn_result", _head_from_index(index))
    if rest == "attn.hook_z":
        return (layer, "head_attention_value_output", _head_from_index(index))
    if rest.startswith("hook_resid"):
        return (layer, "block_output" if "post" in rest else "block_input", None)
    return None


def _spec_to_unit(
    spec: ComponentSpec, token_position, featurizer: Featurizer
) -> AtomicModelUnit:
    """Build the matching LM_units AtomicModelUnit for a component spec."""
    layer, comp, head = spec
    if comp == "mlp_activation":
        return MLP(layer, token_position, location="mlp_activation", featurizer=featurizer)
    if comp == "mlp_output":
        return MLP(layer, token_position, location="mlp_output", featurizer=featurizer)
    if comp == "attn_result":  # per-head hook_result (needs cfg.use_attn_result=True)
        return AttentionHead(layer, head, token_position, location="attn_result",
                             featurizer=featurizer)
    if comp == "head_attention_value_output":
        return AttentionHead(layer, head, token_position, featurizer=featurizer)
    if comp in ("block_output", "block_input"):
        return ResidualStream(layer, token_position, target_output=(comp == "block_output"),
                              featurizer=featurizer)
    raise ValueError(f"Unhandled component spec {spec}")


def circuit_specs(hl_ll_corr, use_ll: bool = True) -> set[ComponentSpec]:
    """The set of component specs that ARE in the circuit (from hl_ll_corr).

    use_ll=True reads the low-level nodes (values) — where to patch in the TRAINED
    model. Set use_ll=False to key off the HL node's own hook. Confirm field names
    with dump_correspondence first.
    """
    specs: set[ComponentSpec] = set()
    for hl, lls in hl_ll_corr.items():
        nodes = (lls if hasattr(lls, "__iter__") else [lls]) if use_ll else [hl]
        for node in nodes:
            hook = getattr(node, "name", None)
            if not hook:
                continue
            s = _hook_to_spec(hook, getattr(node, "index", None))
            if s is not None:
                specs.add(s)
    return specs


def _component_universe(model) -> set[ComponentSpec]:
    """Every patchable component at circuit granularity: each MLP (per layer) and
    each attention head (per layer, per head)."""
    n_layers, n_heads = int(model.cfg.n_layers), int(model.cfg.n_heads)
    universe: set[ComponentSpec] = set()
    for layer in range(n_layers):
        universe.add((layer, "mlp_activation", None))
        for head in range(n_heads):
            universe.add((layer, "attn_result", head))  # InterpBench heads live at hook_result
    return universe


def partition_targets(
    hl_ll_corr,
    model,
    token_position,
    featurizer_factory: Callable[[ComponentSpec], Featurizer] | None = None,
    use_ll: bool = True,
) -> tuple[list[AtomicModelUnit], list[AtomicModelUnit]]:
    """Split the model's components into (circuit, non_circuit) AtomicModelUnits.

    The two returned lists ARE the axes you patch along for the dormancy study:
    interchange each unit (wrap as ``InterchangeTarget([[unit]])``) and compare
    on/off-manifold behaviour of circuit vs non-circuit components. Featurizers
    default to identity (fit per-site later); pass ``featurizer_factory`` to
    supply fitted ones.

    Note: resid circuit entries (if any) are included as circuit units but resid
    is not enumerated for the non-circuit set — non-circuit is over the discrete
    MLP/head component universe, matching the edges.pkl granularity.
    """
    feat = featurizer_factory or (lambda _spec: Featurizer())
    circuit = circuit_specs(hl_ll_corr, use_ll=use_ll)
    universe = _component_universe(model)

    circuit_units = [_spec_to_unit(s, token_position, feat(s)) for s in sorted(circuit, key=str)]
    non_specs = universe - circuit
    non_units = [_spec_to_unit(s, token_position, feat(s)) for s in sorted(non_specs, key=str)]
    return circuit_units, non_units
