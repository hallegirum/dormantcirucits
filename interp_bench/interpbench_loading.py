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
def _encoder_vocab(encoder, specials: set) -> list:
    """Task alphabet off a tracr encoder, so the causal-model config need not be
    hand-registered. tracr's CategoricalEncoder exposes ``.encoding_map`` (value->id);
    sort by id for a canonical order and drop special tokens (bos/pad). Falls back to a
    ``.vocab`` attribute. VERIFY names on the pod with :func:`dump_encoders` if it raises."""
    m = getattr(encoder, "encoding_map", None)
    if isinstance(m, dict) and m:
        return [v for v in sorted(m, key=m.get) if v not in specials]
    v = getattr(encoder, "vocab", None)
    if v is not None:
        return [x for x in v if x not in specials]
    raise AttributeError(
        f"cannot read a vocab off {type(encoder).__name__}: no .encoding_map/.vocab. "
        f"Run dump_encoders(hl_model) and adjust _encoder_vocab to the real attribute."
    )


def dump_encoders(hl_model) -> None:
    """Print the tracr input/output encoders' structure (run once on the pod to confirm
    the attribute names :func:`_encoder_vocab` assumes)."""
    for name in ("tracr_input_encoder", "tracr_output_encoder"):
        enc = getattr(hl_model, name, None)
        print(f"{name}: {type(enc).__name__}")
        for attr in ("encoding_map", "decoding_map", "vocab", "bos_token", "pad_token"):
            if hasattr(enc, attr):
                print(f"    .{attr} = {getattr(enc, attr)!r}")


def build_pipelines(ll_model, case, device: str = "cpu"):
    """Wrap the trained LL model (and, for tracr cases, the reference HL model) as
    TracrPipelines. Returns ``(ll_pipeline, hl_pipeline, bos, hl_fn, vocab, out_vocab)``;
    ``vocab``/``out_vocab`` are derived from the tracr encoders (``None`` for IOI) so the
    causal model's config can be built at runtime instead of hand-registered.

    Two encoding regimes:

    * **tracr cases (18, …)**: the HL model is a HookedTracrTransformer carrying
      ``tracr_input_encoder`` / ``tracr_output_encoder``. SIIT trains the LL model
      into the HL model's encoding, so both share the encoders; ``hl_fn`` runs the HL
      program as ground truth. (decode does NOT pre-strip BOS — the tracr decoder
      already drops the BOS-output, so decode(full_argmax_row) = seq_len labels.)

    * **IOI**: ``IOI_HL`` has NO tracr encoders and operates directly on token ids;
      the LL model instead carries a real HF ``.tokenizer`` and get_clean_data returns
      already-tokenised inputs. So encode is identity (ids in) and decode uses the
      tokenizer. There is no HL pipeline / hl_fn (the result labels come from the
      dataset traces), so those are returned as ``None``.
    """
    hl_model = case.get_hl_model(device=device)

    dump_encoders(hl_model)
    # --- tracr path: HL model carries the encoders ---
    if hasattr(hl_model, "tracr_input_encoder"):
        enc, dec = hl_model.tracr_input_encoder, hl_model.tracr_output_encoder

        def encode(raw_input):
            return enc.encode(list(raw_input))  # raw_input already includes BOS

        def decode(ids):
            return dec.decode(list(ids))  # tracr decoder strips BOS itself

        ll_pipeline = TracrPipeline(ll_model, encode=encode, decode=decode, device=device)
        hl_pipeline = TracrPipeline(hl_model, encode=encode, decode=decode, device=device)
        bos = enc.bos_token

        # config source of truth: derive the alphabets from the encoders themselves
        specials = {bos, getattr(enc, "pad_token", None), None}
        vocab = _encoder_vocab(enc, specials)
        out_vocab = _encoder_vocab(dec, {None})

        def hl_fn(seq):
            return list(hl_pipeline.generate([{"raw_input": [bos] + list(seq)}])["string"])

        return ll_pipeline, hl_pipeline, bos, hl_fn, vocab, out_vocab

    # --- IOI path: no tracr encoders; inputs are already token ids ---
    # Stay in TOKEN-ID space end to end: get_clean_data returns token ids, and the
    # IOI trace stores result as a token id (_ioi_io_name). So encode/decode are
    # identity — the per-position "answer" is an argmax token id, directly comparable
    # to result. (Recover names for display later with ll_model.tokenizer.decode.)
    def encode(raw_input):  # already token ids
        return raw_input.tolist() if hasattr(raw_input, "tolist") else list(raw_input)

    def decode(ids):  # keep token ids (comparable to trace["result"])
        return [int(i) for i in ids]

    ll_pipeline = TracrPipeline(ll_model, encode=encode, decode=decode, device=device)
    # no HL pipeline / hl_fn / tracr vocab for IOI (labels come from the dataset traces)
    return ll_pipeline, None, None, None, None, None


def build_poly_pipeline(ll_model, device: str = "cpu"):
    """Wrap a polysemantic-benchmark low-level HookedTransformer as a TracrPipeline. Like
    IOI, poly stays in TOKEN-ID space: raw_input is [task, BOS, tok…] token ids, output is a
    per-position class id — so encode/decode are identity. Returns just the ll_pipeline (no
    HL pipeline / hl_fn: the studied `result` comes from the dataset labels, not a program)."""
    def encode(raw_input):  # already token ids ([task, BOS, tok…])
        return raw_input.tolist() if hasattr(raw_input, "tolist") else list(raw_input)

    def decode(ids):        # per-position class ids, comparable to trace["result"]
        return [int(i) for i in ids]

    return TracrPipeline(ll_model, encode=encode, decode=decode, device=device)


def poly_membership(corr_mapping, n_tasks: int) -> dict:
    """Per-component circuit membership across the combined tasks, from PolyHLModel.corr_mapping.

    corr_mapping: {LL-component-key -> [role-in-task-0, role-in-task-1, …]} where each slot is
    a list of HL roles (non-empty => in that task's circuit) or None. Keys look like
    'blocks.0.attn.hook_z.1' (head 1) or 'blocks.1.mlp.hook_post'. Returns
    {spec (layer, comp, head|None) -> {"tasks": [i where non-None], "label": ...}} with
    label in {own/shared/router/none} semantics you resolve per base task downstream:
      - in >=2 tasks  -> 'shared' (the polysemantic, dormancy-prone components)
      - in exactly 1  -> that task only
      - in 0          -> 'none' (or the task router / input, handled separately)."""
    out = {}
    for key, roles in corr_mapping.items():
        spec = _hook_to_spec(_polykey_to_hook(key), _polykey_index(key))
        if spec is None:
            continue
        in_tasks = [i for i, r in enumerate(roles) if r is not None]
        out[spec] = {"tasks": in_tasks,
                     "label": ("shared" if len(in_tasks) >= 2 else
                               f"task{in_tasks[0]}" if len(in_tasks) == 1 else "none")}
    return out


def _polykey_to_hook(key: str) -> str:
    # 'blocks.0.attn.hook_z.1' -> 'blocks.0.attn.hook_z' ; 'blocks.1.mlp.hook_post' unchanged
    import re
    m = re.match(r"(blocks\.\d+\.attn\.hook_z)\.(\d+)$", key)
    return m.group(1) if m else key


def _polykey_index(key: str):
    # head index for attn keys ('…hook_z.1' -> a [:, :, 1, :]-style index), else whole-tensor
    import re

    class _Idx:
        def __init__(self, h):
            self.as_index = (slice(None), slice(None), h, slice(None))
    m = re.match(r"blocks\.\d+\.attn\.hook_z\.(\d+)$", key)
    return _Idx(int(m.group(1))) if m else None


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
    h = idx[2]
    return int(h) 


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
    if rest in ("attn.hook_result", "attn.hook_z"):
        return (layer, "attn_result", _head_from_index(index))
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
    dump_correspondence(hl_ll_corr)
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
