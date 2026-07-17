"""Contextual + global prompts and a categorized counterfactual dataset for the
off-manifold / dormant-recruitment experiments.

Generic over any ``CausalModel`` (works for weekdays, months, alphabet, ...).
Nothing here is task-specific: "context" is defined by *conditioning on a subset
of the causal model's input variables*, which is the general form of the
(entity, number) contexts.

Three intervention categories, matching the geometric partition in the design:

    context_matched     source shares the base's context (conditioning vars held
                        fixed), differs only in the free inputs -> a different
                        target value reached WITHIN the base's context.
                        On-contextual / benign negative control.

    context_mismatched  GRADED: exactly one conditioning var is flipped, the rest
                        stay matched to the base -> still a valid same-task prompt
                        (on the global manifold) but its representation may sit off
                        the base's *contextual* manifold. Off-contextual candidate.

    cross_task          source drawn from a related task (months for weekdays,
                        letters for numbers, ...) -> off the *global* manifold
                        entirely. Off-global reference.

Sampling strategy is forward-only (no mechanism inverse) so it transfers to any
causal model. When a constraint is present (a predicate, or the model's
input_filter) it ENUMERATES the small free-input space and picks a valid
completion exactly, rather than rejection-sampling; it only falls back to
rejection sampling when the free space is too large to enumerate.

Design notes / deviations are flagged inline with ``# DESIGN:``.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Iterable

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace
from causalab.causal.counterfactual_dataset import CounterfactualExample
 
# --------------------------------------------------------------------------- #
# Core sampling helpers                                                        #
# --------------------------------------------------------------------------- #
def _free_space_size(model: CausalModel, fixed: dict[str, Any]) -> int:
    """Number of raw completions of the free inputs (pre input_filter)."""
    size = 1
    for v in model.inputs:
        if v not in fixed:
            size *= len(model.values[v])
    return size


def _enumerate_completions(
    model: CausalModel, fixed: dict[str, Any]
) -> list[CausalTrace]:
    """All valid input traces with ``fixed`` pinned and free inputs enumerated.

    Respects ``model.input_filter``. Forward-only: we run the model's mechanisms
    forward on each completion; we never invert them.
    """
    free = [v for v in model.inputs if v not in fixed]
    out: list[CausalTrace] = []
    for combo in itertools.product(*[model.values[v] for v in free]):
        inputs = dict(fixed)
        inputs.update(dict(zip(free, combo)))
        trace = model.new_trace(inputs)
        if model.input_filter is None or model.input_filter(trace):
            out.append(trace)
    return out


def _sample_input_fixed(
    model: CausalModel,
    fixed: dict[str, Any],
    rng: random.Random,
    predicate: Callable[[CausalTrace], bool] | None = None,
    max_enumerate: int = 50_000,
    max_tries: int = 2000,
) -> CausalTrace:
    """Sample an input trace with ``fixed`` pinned and the rest chosen to satisfy
    ``predicate``.

    Parameters
    ----------
    model : the causal model for the task at hand.
    fixed : dict variable -> value pinning the conditioning variables (context).
    rng   : local random generator (deterministic; global RNG untouched).
    predicate : optional constraint on the constructed trace (e.g. "result must
        differ from the base"). ``None`` = any valid input.

    Strategy (forward-only, transferable to any causal model):
      * No constraint (no predicate and no input_filter) -> a single cheap random
        draw. This is the fitting-prompt path.
      * Constrained and the free space is small (<= ``max_enumerate``) ->
        ENUMERATE all valid completions and pick uniformly among those passing
        ``predicate``. Exact (no wasted draws) and an over-constrained context
        fails LOUDLY (empty candidate set) instead of silently spinning.
      * Constrained and the free space is huge -> fall back to rejection sampling.
    """
    free = [v for v in model.inputs if v not in fixed]
    needs_search = predicate is not None or model.input_filter is not None

    # Unconstrained: one shot, no enumeration.
    if not needs_search:
        inputs = dict(fixed)
        for v in free:
            inputs[v] = rng.choice(model.values[v])
        return model.new_trace(inputs)

    # Constrained + enumerable: exact selection.
    if _free_space_size(model, fixed) <= max_enumerate:
        candidates = _enumerate_completions(model, fixed)  # respects input_filter
        if predicate is not None:
            candidates = [t for t in candidates if predicate(t)]
        if not candidates:
            raise RuntimeError(
                f"No input with fixed={fixed} satisfies the predicate — the "
                f"context is over-constrained (no free variable can change the "
                f"target). Loosen the conditioning or the predicate."
            )
        return rng.choice(candidates)

    # Constrained + free space too large to enumerate: rejection sampling.
    for _ in range(max_tries):
        inputs = dict(fixed)
        for v in free:
            inputs[v] = rng.choice(model.values[v])
        trace = model.new_trace(inputs)
        if model.input_filter is not None and not model.input_filter(trace):
            continue
        if predicate is not None and not predicate(trace):
            continue
        return trace
    raise RuntimeError(
        f"Rejection sampling failed for fixed={fixed} after {max_tries} tries "
        f"(free space > max_enumerate and predicate rarely satisfied)."
    )


def _context_of(trace: CausalTrace, conditioning_vars: list[str]) -> dict[str, Any]:
    """The base's context = its assignment to the conditioning variables."""
    return {v: trace[v] for v in conditioning_vars}


# --------------------------------------------------------------------------- #
# Global + contextual prompt sets (used to FIT the manifolds)                  #
# --------------------------------------------------------------------------- #
def create_global_prompts(
    model: CausalModel,
    n: int,
    seed: int = 42,
) -> list[CounterfactualExample]:
    """Unconditional samples — the substrate for the *global* manifold.

    Wrapped as self-paired CounterfactualExamples so the global set feeds
    collect_features / find_path in the same shape as the contextual sets.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = _sample_input_fixed(model, {}, rng)
        out.append(CounterfactualExample(input=t, counterfactual_inputs=[t]))
    return out


def _classes_for(
    model: CausalModel,
    var: str,
    group_by: dict[str, Callable[[CausalModel, str], dict[Any, list[Any]]]] | None,
    context_values: dict[str, Iterable[Any]] | None,
) -> dict[Any, list[Any]]:
    """Resolve ``{class_label: [members]}`` for a conditioning variable.

    Default = singleton classes (one per raw value). ``group_by[var]`` overrides
    with a task grouping (e.g. residue classes). ``context_values[var]`` then
    filters members and drops any now-empty classes.
    """
    if group_by is not None and var in group_by:
        classes = {k: list(v) for k, v in group_by[var](model, var).items()}
    else:
        classes = {val: [val] for val in model.values[var]}
    if context_values is not None and var in context_values:
        allowed = set(context_values[var])
        classes = {
            lab: [m for m in members if m in allowed]
            for lab, members in classes.items()
        }
        classes = {lab: members for lab, members in classes.items() if members}
    return classes


def create_contextual_prompts(
    model: CausalModel,
    conditioning_vars: list[str],
    n_per_context: int,
    seed: int = 42,
    context_values: dict[str, Iterable[Any]] | None = None,
    group_by: dict[str, Callable[[CausalModel, str], dict[Any, list[Any]]]] | None = None,
) -> dict[tuple, list[CausalTrace]]:
    """Prompt sets grouped by context — the substrate for *contextual* manifolds.

    Parameters
    ----------
    model : the causal model underlying the task at hand.
    conditioning_vars : the variables held fixed to define a context.
    n_per_context : number of prompts sampled per context.
    context_values : optional restriction of which values each conditioning var
        may take (applied to class members; empty classes are dropped).
    group_by : optional ``{var: grouping_fn}`` where ``grouping_fn(model, var)``
        returns ``{class_label: [member_values]}``. A context is then one *class*,
        and its prompts draw the conditioning var uniformly from ALL members of
        the class (unbiased pooling) rather than a single representative. Vars not
        in ``group_by`` use singleton classes (one context per raw value).

    A "context" is an assignment of a class to each conditioning var (e.g.
    entity="Monday", or number-class="adds-1"={one, eight}). For each context we
    sample ``n_per_context`` prompts: the conditioning vars are drawn from their
    class members, the remaining inputs are random, so the target sweeps its range
    within the context. Returns ``{context_key: [traces...]}`` where
    ``context_key`` is the tuple of class labels (equal to the raw value for
    singleton classes). The actual member drawn per prompt is recoverable from
    ``trace[var]``.

    DESIGN — call twice (["entity"] then ["number"]) for the two-axis scheme.
    Member pooling is uniform-random per prompt; for balanced coverage of small
    classes raise ``n_per_context`` (or enumerate members explicitly).
    """
    rng = random.Random(seed)

    # Per conditioning var: {class_label: [member_values]} (singletons by default).
    classes_per_var: list[dict[Any, list[Any]]] = [
        _classes_for(model, v, group_by, context_values) for v in conditioning_vars
    ]

    out: dict[tuple, list[CausalTrace]] = {}
    for label_combo in itertools.product(*[list(c.keys()) for c in classes_per_var]):
        members = [classes_per_var[i][label] for i, label in enumerate(label_combo)]
        prompts: list[CausalTrace] = []
        for _ in range(n_per_context):
            # DESIGN: draw each conditioning var uniformly from ALL its class
            # members (unbiased pooling), then sample the free inputs. No "must
            # differ" predicate — for FITTING we want the natural within-context
            # spread, not a curated set.
            fixed = {v: rng.choice(members[i]) for i, v in enumerate(conditioning_vars)}
            trace = _sample_input_fixed(model, fixed, rng)
            prompts.append(CounterfactualExample(input=trace, counterfactual_inputs=[trace]))
        out[label_combo] = prompts
    return out


# --------------------------------------------------------------------------- #
# Categorized counterfactual dataset (used to RUN the interventions)           #
# --------------------------------------------------------------------------- #
def create_counterfactual_dataset(
    model: CausalModel,
    base_prompts :list[CounterfactualExample],
    conditioning_vars: list[str],
    n: int,
    seed: int = 42,
    categories: tuple[str, ...] = ("context_matched", "context_mismatched", "cross_task"),
    related_models: list[CausalModel] | None = None,
    target_variable: str = "result",
) -> list[dict[str, Any]]:
    """Build a paired, category-labelled counterfactual dataset.

    For each of ``n`` base inputs, emit ONE counterfactual per active category,
    all sharing the same base. Pairing by base gives matched samples across
    conditions (for paired cross-condition statistics). Total = n * len(categories).

    Each returned example is a ``CounterfactualExample`` plus two extra keys the
    interchange machinery ignores but the analysis reads:
        "category" : one of the category strings.
        "context"  : {var: base_value} for the conditioning vars (for grouping
                     divergences by context downstream).

    Parameters
    ----------
    model : causal model underlying the task at hand.
    conditioning_vars : variables that define a context.
    categories : which counterfactual types to emit. context_matched = same
        context as base; context_mismatched = graded one-knob context change,
        same task; cross_task = a related but different task (weekdays vs months).
    related_models : sources for cross_task (required iff "cross_task" active).
    """
    if "cross_task" in categories and not related_models:
        raise ValueError("cross_task category requires related_models (>=1).")

    rng = random.Random(seed)
    # Dedicated rngs per related model keep cross-task sampling reproducible and
    # independent of how many other categories are active.
    related_rngs = [
        random.Random(seed + 1000 + i) for i in range(len(related_models or []))
    ]

    examples: list[dict[str, Any]] = []
    for example in base_prompts:
        base = example["input"]
        base_ctx = _context_of(base,conditioning_vars)
        base_target = base[target_variable]

        for cat in categories:
            cf = _make_counterfactual(
                model=model,
                base_ctx=base_ctx,
                base_target=base_target,
                conditioning_vars=conditioning_vars,
                target_variable=target_variable,
                category=cat,
                rng=rng,
                related_models=related_models,
                related_rngs=related_rngs,
            )
            examples.append(
                CounterfactualExample(input = base, counterfactual_inputs=[cf])
    
            )
    return examples


def _make_counterfactual(
    *,
    model: CausalModel,
    base_ctx: dict[str, Any],
    base_target: Any,
    conditioning_vars: list[str],
    target_variable: str,
    category: str,
    rng: random.Random,
    related_models: list[CausalModel] | None,
    related_rngs: list[random.Random],
) -> CausalTrace:
    """Construct one counterfactual source trace for the given category."""

    # Require the target to actually change, else the "patch" is a no-op and the
    # example carries no causal signal.
    def _target_differs(t: CausalTrace) -> bool:
        return t[target_variable] != base_target

    if category == "context_matched":
        # Same context (all conditioning vars pinned to base), free vars chosen so
        # the target differs. On-contextual: source lies on base's context slice.
        return _sample_input_fixed(
            model, dict(base_ctx), rng, predicate=_target_differs
        )

    if category == "context_mismatched":
        # GRADED mismatch: flip exactly ONE conditioning var to a different value,
        # keep the other conditioning vars matched to the base, choose free vars so
        # the target differs. With a single conditioning var this is a full context
        # change; with several it is a one-knob (graded) shift.
        flip_var = rng.choice(conditioning_vars)
        alt_values = [v for v in model.values[flip_var] if v != base_ctx[flip_var]]
        if not alt_values:
            raise ValueError(
                f"Cannot flip conditioning var {flip_var!r}: it has only one value."
            )
        fixed = dict(base_ctx)
        fixed[flip_var] = rng.choice(alt_values)
        return _sample_input_fixed(model, fixed, rng, predicate=_target_differs)

    if category == "cross_task":
        # Off-global: a valid prompt from a *related* task. Run through the base
        # model's pipeline it yields a genuinely off-manifold activation. No
        # target/context predicate — the source is foreign, so "target value" is
        # not comparable. Uses a dedicated rng for reproducibility.
        assert related_models is not None  # guarded by the caller
        i = rng.randrange(len(related_models))
        return _sample_input_fixed(related_models[i], {}, related_rngs[i])

    raise ValueError(f"Unknown category: {category!r}")


# --------------------------------------------------------------------------- #
# Optional: aggregation helper for cyclic natural-domain tasks                 #
# --------------------------------------------------------------------------- #
def modclass_groups(model: CausalModel, var: str) -> dict[int, list[Any]]:
    """Group a variable's values by residue class, keeping ALL members.

    Use as an entry in the ``group_by`` arg of create_contextual_prompts to make
    each context one arithmetic equivalence class (e.g. numbers that add the same
    amount mod M), pooling every member rather than a single representative.
    Falls back to singleton classes if the model has no modulus (non-cyclic task).

    Returns ``{residue: [members...]}``.

    DESIGN: reads the modulus off the task config stashed on the model
    (``model._nda_config``). NDA-specific, hence an opt-in hook.
    """
    cfg = getattr(model, "_nda_config", None)
    modulus = getattr(cfg, "modulus", None)
    values = list(model.values[var])
    if not modulus:
        return {v: [v] for v in values}  # singletons: no arithmetic grouping
    groups: dict[int, list[Any]] = {}
    for i, v in enumerate(values):
        groups.setdefault(i % modulus, []).append(v)
    return groups


