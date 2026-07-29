"""
tracr_pipeline.py
=================
A causalab ``Pipeline`` for InterpBench / tracr models (TransformerLens
``HookedTransformer``), implemented on **native TL hooks** instead of pyvene.

Why this exists
---------------
causalab's intervention stack is pyvene-based, and pyvene assumes a HuggingFace
model (`model.config`, `model.generate(...)`). A ``HookedTransformer`` exposes
`.cfg` and a different `.generate`, so pyvene cannot wrap it. But the pyvene
object is only ever touched at four seams — build / interchange-generate /
collect / release — and everything above them (CausalModel, InterchangeTarget,
Featurizer, the off-manifold metrics) is model-agnostic.

So this subclass re-implements those seams (build / interchange-generate /
collect / release, plus the interpolation `set_interpolation` used by Gaussian
patching) on ``run_with_cache`` / ``run_with_hooks``. The intervention
*semantics* are causalab's own: we reuse each unit's ``Featurizer.featurize`` /
``inverse_featurize``, so the maths mirror ``FeatureInterchangeIntervention``
and ``FeatureInterpolateIntervention`` in causalab/neural/featurizer.py. The
backend imports **no pyvene at all**.

The handle
----------
``build_intervention`` returns a ``_NativePlan`` that carries only the
``InterchangeTarget``. Every location is a real ``AtomicModelUnit``; the TL
hook name is derived on demand from ``(component_type, layer)``. Nothing off
the unit is copied — no parallel unit class to keep in sync.

Validation gate
---------------
The maths mirror causalab's pyvene path by *reading*, not by testing. Before
trusting any InterpBench result, run ``parity_check.py``: the same
base/source/unit through an ``LMPipeline`` (pyvene) and through this pipeline on
the same model must give matching intervened logits and collected features.

Scope note
----------
Implemented for residual-stream style sites (``block_output`` etc.). Head-level
components and the cross-model ``source_representations`` path raise a clear
``NotImplementedError`` rather than silently doing the wrong thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from causalab.neural.pipeline import Pipeline
from causalab.neural.units import AtomicModelUnit, InterchangeTarget


# --------------------------------------------------------------------------- #
#  component_type -> TransformerLens hook name
# --------------------------------------------------------------------------- #
# causalab/pyvene name the site semantically ("block_output" == the residual
# stream after block L). TransformerLens names the same tensor by its hook.
# Only the sites the off-manifold work reads are mapped; anything else errors.
# component_type -> TL hook. Names match causalab LM_units where they exist;
# "mlp_activation" is the d_mlp hidden = tracr's hook_post; "attn_result" is the
# per-head output AFTER W_O (d_model) that InterpBench correspondences use.
#
# InterpBench (tracr) correspondences use only { attn.hook_result, mlp.hook_post }.
# The rest are the standard patchable TL sites (confirm the exact set for a given
# model with: sorted({ll.name for _,lls in hl_ll_corr.items() for ll in lls})).
_COMPONENT_TO_HOOK: dict[str, str] = {
    # residual stream
    "block_input": "blocks.{layer}.hook_resid_pre",
    "block_output": "blocks.{layer}.hook_resid_post",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    # mlp
    "mlp_activation": "blocks.{layer}.mlp.hook_post",  # d_mlp hidden (tracr MLP SOps)
    "mlp_output": "blocks.{layer}.hook_mlp_out",
    # attention (per-head, 4D: batch,pos,head,·)
    "head_attention_value_output": "blocks.{layer}.attn.hook_z",   # (·,head,d_head) — pyvene parity
    "attn_result": "blocks.{layer}.attn.hook_result",              # (·,head,d_model) — InterpBench corr
    "attention_output": "blocks.{layer}.hook_attn_out",
}

# component_types whose activation is (batch, pos, n_heads, ·): the unit's `.head`
# selects a slab, and its indices arrive 2-level as [heads, positions] (see
# AttentionHead.index_component in causalab/neural/LM_units.py). hook_result needs
# the model built with cfg.use_attn_result=True (else the hook is absent).
_HEAD_COMPONENTS = frozenset(
    {"head_attention_value_output", "head_attention_value_input", "attn_result"}
)


def hook_name(unit: AtomicModelUnit) -> str:
    """TL hook for a unit, derived from its (component_type, layer)."""
    try:
        template = _COMPONENT_TO_HOOK[unit.component_type]
    except KeyError as exc:
        raise NotImplementedError(
            f"TracrPipeline has no TL-hook mapping for component_type "
            f"'{unit.component_type}'. Add it to _COMPONENT_TO_HOOK."
        ) from exc
    return template.format(layer=unit.layer)


def _is_head(unit: AtomicModelUnit) -> bool:
    return str(getattr(unit, "component_type", "")) in _HEAD_COMPONENTS


def _slab(unit: AtomicModelUnit, act: Tensor) -> Tensor:
    """The (B, S, D_eff) tensor a unit operates on. For head units that's the
    single head's slice of hook_z; otherwise the activation itself."""
    return act[:, :, unit.head, :] if _is_head(unit) else act  # type: ignore[attr-defined]


def _write_slab(unit: AtomicModelUnit, act: Tensor, new_slab: Tensor) -> Tensor:
    """Scatter an edited slab back. For head units, into the head's slot of hook_z."""
    if _is_head(unit):
        out = act.clone()
        out[:, :, unit.head, :] = new_slab  # type: ignore[attr-defined]
        return out
    return new_slab


def _positions(unit: AtomicModelUnit, loc: Any) -> Any:
    """Position indices for a unit. Head units carry a 2-level [heads, positions]
    index (AttentionHead.index_component); the head is already on the unit, so we
    take the positions half."""
    return loc[1] if _is_head(unit) else loc


# --------------------------------------------------------------------------- #
#  The opaque handle (pyvene IntervenableModel's native analogue)
# --------------------------------------------------------------------------- #
@dataclass
class _NativePlan:
    """Native stand-in for a pyvene ``IntervenableModel``.

    Holds only the ``InterchangeTarget`` — every location is a real
    ``AtomicModelUnit`` and the hook name is computed via :func:`hook_name`.
    Locations / feature indices are not stored: like pyvene, they arrive
    per-batch on ``intervenable_generate`` / ``intervenable_collect``.
    """

    target: InterchangeTarget
    intervention_type: str
    # interpolation only: fn(f_base, f_src, **params) -> features, set per-batch
    # by set_interpolation (the native analogue of set_interventions_interpolation).
    fn: Callable[..., Tensor] | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def units(self) -> list[tuple[int, AtomicModelUnit]]:
        """(group_index, unit) in the group-major order that
        ``prepare_intervenable_inputs`` uses to build its index lists."""
        return [
            (g, unit) for g, group in enumerate(self.target) for unit in group
        ]

    def set_interpolation(self, fn: Callable[..., Tensor], **params: Any) -> None:
        """Native analogue of the per-intervention ``set_interpolation``."""
        self.fn = fn
        self.params = params

    def release(self) -> None:  # symmetry with delete_intervenable_model
        pass


# --------------------------------------------------------------------------- #
#  The pipeline
# --------------------------------------------------------------------------- #
class TracrPipeline(Pipeline):
    """A ``Pipeline`` over a tracr/InterpBench ``HookedTransformer``.

    Parameters
    ----------
    model :
        A TransformerLens ``HookedTransformer`` (already weight-loaded).
    encode :
        ``encode(raw_input) -> Sequence[int]`` mapping one task input (whatever
        lives in ``trace["raw_input"]``) to a token-id sequence. For tracr this
        wraps the case's input encoder (prepend BOS, symbol->id).
    decode :
        ``decode(token_ids: Sequence[int]) -> Any`` mapping an output token-id
        row back to a label (tracr's output decoder). Used by :meth:`dump`.
    pad_token_id :
        Id used to right-pad ragged batches. tracr models ignore padding at
        non-target positions; defaults to 0.
    device :
        Overrides the model's device if given.
    """

    def __init__(
        self,
        model: Any,
        *,
        encode: Callable[[Any], Sequence[int]],
        decode: Callable[[Sequence[int]], Any],
        pad_token_id: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.model_or_name = model
        self.model = model
        self.tokenizer = None  # tracr has no HF tokenizer; encode/decode replace it
        self._encode = encode
        self._decode = decode
        self.pad_token_id = pad_token_id
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.model.to(self.device)
        # Off-manifold code occasionally reads these off the pipeline.
        self.max_new_tokens = 1

    # -- Pipeline requires _setup_model, but we take a live model in __init__. --
    def _setup_model(self) -> None:  # pragma: no cover - unused path
        pass

    # ------------------------------------------------------------------ #
    #  I/O: load / dump / generate
    # ------------------------------------------------------------------ #
    def load(self, input: list[Any]) -> dict[str, Tensor]:
        """Encode a batch of traces to ``{"input_ids": LongTensor(batch, seq)}``."""
        rows = [list(self._encode(item["raw_input"])) for item in input]
        width = max(len(r) for r in rows)
        padded = [r + [self.pad_token_id] * (width - len(r)) for r in rows]
        tokens = torch.as_tensor(padded, dtype=torch.long, device=self.device)
        return {"input_ids": tokens, "tokens" :tokens}

    def dump(self, model_output: Any, *, is_logits: bool = True) -> Any:
        """Decode logits (or token ids) to task labels."""
        if isinstance(model_output, dict):
            model_output = model_output.get("sequences", model_output.get("scores"))
        out = torch.as_tensor(model_output)
        token_ids = out.argmax(dim=-1) if (is_logits and out.dim() >= 3) else out
        decoded = [self._decode(row.tolist()) for row in token_ids]
        return decoded[0] if len(decoded) == 1 else decoded

    def generate(self, input: list[Any], **gen_kwargs: Any) -> dict[str, Any]:
        """Single forward pass (tracr tasks are not autoregressive)."""
        tokens = self.load(input)["tokens"]
        with torch.no_grad():
            logits = self.model(tokens)
        logits = logits.detach().cpu()
        return {
            "sequences": logits.argmax(dim=-1),
            "scores": logits,
            "string": self.dump(logits, is_logits=True),
        }

    # ------------------------------------------------------------------ #
    #  Intervention seam 1/4: build
    # ------------------------------------------------------------------ #
    def build_intervention(
        self,
        model_units: InterchangeTarget | list[AtomicModelUnit],
        intervention_type: str = "interchange",
    ) -> _NativePlan:
        """Native analogue of ``prepare_intervenable_model``."""
        if intervention_type not in ("interchange", "collect", "interpolation"):
            raise NotImplementedError(
                f"TracrPipeline supports 'interchange', 'collect', 'interpolation'; "
                f"got '{intervention_type}'. (mask/add/replace are not used by the "
                f"off-manifold analysis.)"
            )
        target = (
            InterchangeTarget([model_units])
            if isinstance(model_units, list)
            else model_units
        )
        plan = _NativePlan(target=target, intervention_type=intervention_type)
        for _, unit in plan.units():  # fail fast on unmapped components
            hook_name(unit)
        return plan

    # ------------------------------------------------------------------ #
    #  Intervention seam 2/4: interchange-generate
    # ------------------------------------------------------------------ #
    def intervenable_generate(
        self,
        intervenable_model: _NativePlan,
        base: dict[str, Tensor],
        sources: list[dict[str, Tensor]] | None,
        map: dict[str, Any],  # noqa: A002 - matches base-class name
        feature_indices: list[list[list[int] | None]] | None,
        source_representations: list[Tensor] | None = None,
        **gen_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one forward pass with the intervention applied via TL hooks.

        At each unit's mapped positions, combine the featurized base and source
        (whole-vector swap, DAS-subspace swap, or — for ``interpolation`` —
        ``plan.fn(f_base, f_src, **params)``), then inverse-featurize back into
        the residual stream. Mirrors ``FeatureInterchangeIntervention`` /
        ``FeatureInterpolateIntervention``.
        """
        plan = intervenable_model
        if source_representations is not None:
            raise NotImplementedError(
                "Cross-model / pre-collected source_representations path is not "
                "implemented for the native backend yet; pass `sources` instead."
            )
        if sources is None:
            raise ValueError("intervenable_generate needs `sources`.")
        if plan.intervention_type == "interpolation" and plan.fn is None:
            raise ValueError(
                "interpolation plan has no fn; call set_interpolation() first "
                "(run_interpolation_interventions does this per batch)."
            )

        units = plan.units()
        src_locs, base_locs = map["sources->base"]  # each: [unit][batch][pos...]

        # 1. Featurize each source group's activation once, per unit's hook.
        #    For head units, featurize only that head's (B, S, d_head) slab.
        group_caches: dict[int, Any] = {}
        src_feats: list[Tensor] = []
        for group, unit in units:
            if group not in group_caches:
                with torch.no_grad():
                    _, group_caches[group] = self.model.run_with_cache(
                        sources[group]["tokens"]
                    )
            slab = _slab(unit, group_caches[group][hook_name(unit)])
            f_src, _ = unit.featurizer.featurize(slab)
            src_feats.append(f_src)

        # 2. Base forward with one hook per unit that writes the combined result in.
        def make_hook(unit_i: int) -> tuple[str, Callable[..., Tensor]]:
            _, unit = units[unit_i]
            f_src = src_feats[unit_i]
            b_pos = _positions(unit, base_locs[unit_i])
            s_pos = _positions(unit, src_locs[unit_i])
            # feature_indices are identical across the batch (see prepare_intervenable_inputs)
            feat = (
                None if feature_indices is None else feature_indices[unit_i][0]
            )

            def hook_fn(act: Tensor, **_kw: Any) -> Tensor:
                slab = _slab(unit, act)  # (B, S, D_eff): head slice or full act
                f_base, base_err = unit.featurizer.featurize(slab)
                dev = f_base.device
                rows = torch.arange(f_base.shape[0], device=dev).unsqueeze(1)  # (B, 1)
                # positions are rectangular across the batch (pyvene requires this too)
                b_idx = torch.as_tensor(b_pos, device=dev)  # (B, n_pos)
                s_idx = torch.as_tensor(s_pos, device=dev)  # (B, n_pos)
                gb = f_base[rows, b_idx]  # (B, n_pos, D_eff)
                gs = f_src[rows, s_idx]
                combined = self._combine(plan, gb, gs, feat)
                f_out = f_base.clone()
                f_out[rows, b_idx] = combined
                new_slab = unit.featurizer.inverse_featurize(f_out, base_err).to(act.dtype)
                return _write_slab(unit, act, new_slab)

            return hook_name(unit), hook_fn

        fwd_hooks = [make_hook(i) for i in range(len(units))]
        with torch.no_grad():
            logits = self.model.run_with_hooks(base["tokens"], fwd_hooks=fwd_hooks)

        logits = logits.detach().cpu()
        result: dict[str, Any] = {
            "sequences": logits.argmax(dim=-1),
            "string": self.dump(logits, is_logits=True),
        }
        if gen_kwargs.get("output_scores", True):
            result["scores"] = logits
        return result

    @staticmethod
    def _combine(
        plan: _NativePlan, gb: Tensor, gs: Tensor, feat: list[int] | None
    ) -> Tensor:
        """Combine gathered base/source features at the intervention positions.

        gb, gs: (B, n_pos, n_feat). Returns the same shape.
          - interpolation : plan.fn(f_base=gb, f_src=gs, **params)  (feat ignored,
                            matching FeatureInterpolateIntervention)
          - interchange, no subspace : gs                      (full-vector swap)
          - interchange, subspace    : gb with feat dims taken from gs (DAS)
        """
        if plan.intervention_type == "interpolation":
            return plan.fn(f_base=gb, f_src=gs, **plan.params)  # type: ignore[misc]
        if not feat:  # None or empty -> whole-vector interchange
            return gs
        out = gb.clone()
        idx = torch.as_tensor(feat, device=gb.device)
        out[..., idx] = gs[..., idx]
        return out

    # ------------------------------------------------------------------ #
    #  Intervention seam 3/4: collect
    # ------------------------------------------------------------------ #
    def intervenable_collect(
        self,
        intervenable_model: _NativePlan,
        loaded_inputs: dict[str, Tensor],
        indices: list[list[list[int]]],  # [unit][batch][pos...]
        return_model_output: bool = False,
    ) -> list[Tensor] | tuple[list[Tensor], Any]:
        """Native analogue of ``_collect_activations_single_batch``.

        Returns one tensor per unit of gathered *featurized* activations,
        shape ``(batch, n_positions, n_features)`` — matching what the pyvene
        collect intervention hands back. When ``return_model_output`` is set, the
        second element exposes ``.logits`` (as ``collect_features`` expects).
        """
        plan = intervenable_model
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(loaded_inputs["tokens"])

        out: list[Tensor] = []
        for unit_i, (_, unit) in enumerate(plan.units()):
            slab = _slab(unit, cache[hook_name(unit)])  # head slice or full act
            f, _ = unit.featurizer.featurize(slab)  # (B, S, n_feat)
            pos = _positions(unit, indices[unit_i])
            gathered = torch.stack(
                [f[b, torch.as_tensor(pos[b], device=f.device)] for b in range(f.shape[0])]
            )
            out.append(gathered)

        if return_model_output:
            return out, SimpleNamespace(logits=logits)
        return out

    # ------------------------------------------------------------------ #
    #  Intervention seam 4/4: release
    # ------------------------------------------------------------------ #
    def release_intervention(self, intervenable_model: _NativePlan) -> None:
        """Native analogue of ``delete_intervenable_model`` (nothing to free)."""
        intervenable_model.release()


def _model_device(model: Any) -> torch.device:
    cfg_dev = getattr(getattr(model, "cfg", None), "device", None)
    if cfg_dev is not None:
        return torch.device(cfg_dev)
    return next(model.parameters()).device
