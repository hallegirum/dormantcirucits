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
from causalab.causal.trace import CausalTrace, Mechanism, input_var
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
    context_predicate: Callable[[CausalTrace], bool] | None = None,
    max_enumerate: int = 50_000,
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
    max_enumerate : threshold below which _sample_input_fixed enumerates the free
        input space exactly (else it rejection-samples). For SYMBOLIC causal models
        (weekdays: new_trace is arithmetic) the 50k default is cheap. For InterpBench
        tracr models new_trace runs a full HL-transformer FORWARD, so enumerating the
        space is catastrophic (millions of forwards) — pass a SMALL value there to force
        cheap rejection sampling (the "target differs" predicate is easily satisfied).
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
                base_trace=base,
                context_predicate=context_predicate,
                max_enumerate=max_enumerate,
            )
            examples.append(
                CounterfactualExample(input=base, counterfactual_inputs=[cf])
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
    base_trace: CausalTrace | None = None,
    context_predicate: Callable[[CausalTrace], bool] | None = None,
    max_enumerate: int = 50_000,
) -> CausalTrace:
    """Construct one counterfactual source trace for the given category."""

    # Require the target to actually change, else the "patch" is a no-op and the
    # example carries no causal signal.
    def _target_differs(t: CausalTrace) -> bool:
        return t[target_variable] != base_target

    # PREDICATE-DEFINED CONTEXT (e.g. wrapping). The context is a derived joint
    # property of (entity, number), not a pinned variable, so matched/mismatched
    # are expressed by the predicate's truth value rather than by fixing vars.
    # Nothing is pinned — the source may change entity AND/OR number to flip the
    # regime (option 2), which keeps every base feasible; pinning the entity would
    # be unsatisfiable for Sunday, which can never be non-wrapping.
    if context_predicate is not None and category in ("context_matched", "context_mismatched"):
        base_side = context_predicate(base_trace)
        want = base_side if category == "context_matched" else (not base_side)
        return _sample_input_fixed(
            model, {}, rng,
            predicate=lambda t: (context_predicate(t) == want) and _target_differs(t),
            max_enumerate=max_enumerate,
        )

    if category == "context_matched":
        # Same context (all conditioning vars pinned to base), free vars chosen so
        # the target differs. On-contextual: source lies on base's context slice.
        return _sample_input_fixed(
            model, dict(base_ctx), rng, predicate=_target_differs,
            max_enumerate=max_enumerate,
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
        return _sample_input_fixed(model, fixed, rng, predicate=_target_differs,
                                   max_enumerate=max_enumerate)

    if category == "cross_task":
        # Off-global: a valid prompt from a *related* task. Run through the base
        # model's pipeline it yields a genuinely off-manifold activation. No
        # target/context predicate — the source is foreign, so "target value" is
        # not comparable. Uses a dedicated rng for reproducibility.
        assert related_models is not None  # guarded by the caller
        i = rng.randrange(len(related_models))
        return _sample_input_fixed(related_models[i], {}, related_rngs[i])

    raise ValueError(f"Unknown category: {category!r}")

def wrap_predicate(model: CausalModel) -> Callable[[CausalTrace], bool]:
    """Return ``wraps(trace)`` for a cyclic natural-domain task.

    ``result = (entity_index + number_int) % modulus``, so an example *wraps*
    exactly when that sum crosses the modulus::

        wraps  <=>  entity_index + number_int >= modulus

    (Note this is NOT a condition on ``result % modulus`` — the modulo has
    already been applied, so it carries no information about crossing.)

    Index/int/modulus are read off the model (``embeddings`` expose the same
    maps the mechanism uses; ``periods`` gives the modulus), so this works
    unchanged for months (modulus 12) as well as weekdays (7).
    """
    ent_idx = model.embeddings["entity"]
    num_int = model.embeddings["number"]
    modulus = int(model.periods["result"])

    def wraps(trace: CausalTrace) -> bool:
        return int(ent_idx(trace["entity"])[0]) + int(num_int(trace["number"])[0]) >= modulus

    return wraps


def create_wrap_prompts(
    model: CausalModel,
    wrapping: bool,
    n: int,
    seed: int = 42,
) -> list[CounterfactualExample]:
    """Base prompts restricted to the wrapping (or non-wrapping) regime.

    The wrap regime is a JOINT property of (entity, number), so unlike
    create_contextual_prompts we can't pin a variable — we pass a predicate to
    ``_sample_input_fixed``, which enumerates and filters exactly.

    Feasibility note (weekdays, numbers 1..7, entity 0..6): wrap-availability is
    entity-dependent — Sunday ALWAYS wraps (6+1>=7) and Monday wraps only with
    "seven". The non-wrapping regime also never produces result="Monday" (the
    sum is >=1), so it has 6 class centroids vs 7 for wrapping.
    """
    wraps = wrap_predicate(model)
    predicate = wraps if wrapping else (lambda t: not wraps(t))
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = _sample_input_fixed(model, {}, rng, predicate)
        out.append(CounterfactualExample(input=t, counterfactual_inputs=[t]))
    return out

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


# --------------------------------------------------------------------------- #
# IOI: causal model + trace pool (observational, not generative)              #
# --------------------------------------------------------------------------- #
# IOI is OBSERVATIONAL: the sentences are FIXED data from the IOIDataset, not sampled from
# a causal model (there's no tracr program to run). So — unlike the tracr/natural-domain
# helpers above, which SAMPLE model.values — create_ioi_causal_model builds the variable
# SCHEMA and the traces straight from the data in ONE pass. The traces are real
# CausalTraces (so the spline fitter's trace._values works), with raw_input modelled as an
# INPUT (the token ids are given, not computed) — no dict intermediates, no _values hacks.
def ioi_name_ids(clean: Any) -> set[int]:
    """Single-token name ids (GPT-2 names are one token, usually with a leading space)."""
    tok = clean.tokenizer
    name_ids: set[int] = set()
    for nm in clean.names:
        for variant in (" " + nm, nm):
            enc = tok.encode(variant)
            if len(enc) == 1:
                name_ids.add(enc[0])
    return name_ids


def _ioi_records(clean: Any) -> list[dict[str, Any]]:
    """Per-example IOI facts derived from the token pattern (there's no metadata field).

    Returns ``[{template_type, IO, S, result, raw_input}]``. S (the repeated subject) is
    the 3rd name occurrence; the sentence is ABBA when S sits in the 2nd name slot (IO =
    1st name), else BABA (IO = 2nd name). ``result`` is the IO's token id (the one-hot
    target's last-position argmax), and ``raw_input`` is the token-id list.
    """
    import torch

    inputs = clean.get_inputs()      # (N, seq) token ids — fetch ONCE
    targets = clean.get_targets()    # list of (seq, d_vocab_out) one-hots — ONCE
    name_ids = ioi_name_ids(clean)

    records: list[dict[str, Any]] = []
    for i in range(len(inputs)):
        toks = inputs[i].tolist() if torch.is_tensor(inputs[i]) else list(inputs[i])
        occ = [t for t in toks if t in name_ids]  # name occurrences in order
        if len(occ) < 3:
            raise ValueError(
                f"IOI example {i}: expected 3 name occurrences, got {len(occ)} in {toks} "
                f"— some names may be multi-token; widen name_ids."
            )
        slot1, slot2, s_id = occ[0], occ[1], occ[2]
        ttype = "ABBA" if s_id == slot2 else "BABA"
        io_id = slot1 if s_id == slot2 else slot2
        tgt = targets[i]
        result = int(tgt[-1].argmax()) if hasattr(tgt, "argmax") else io_id
        records.append({"template_type": ttype, "IO": io_id, "S": s_id,
                        "result": result, "raw_input": toks})
    return records


def create_ioi_causal_model(clean: Any) -> tuple[CausalModel, dict[str, list]]:
    """Build the IOI ``CausalModel`` AND its per-order trace pool in ONE pass — the IOI
    counterpart to a tracr task's ``create_causal_model``.

    Variables
        IO             indirect-object name id  (the answer; the fitted manifold parameter)
        S              subject name id          (the repeated name)
        template_type  ABBA / BABA              (context axis -> one manifold per value)
        raw_input      the token ids            (an INPUT, provided per example — the
                                                 sentence is data, so it's set, not computed)
        result         = IO
        raw_output     = result

    Returns ``(causal_model, pool)`` with ``pool = {template_type: [CausalTrace]}`` ready
    for :func:`create_ioi_counterfactual_dataset`. Because ``raw_input`` is an input the
    traces are built directly by ``new_trace`` — no post-hoc ``_values`` surgery.
    """
    from collections import defaultdict

    records = _ioi_records(clean)
    io_ids = sorted({r["result"] for r in records})
    s_ids = sorted({r["S"] for r in records})
    orders = sorted({r["template_type"] for r in records})

    mechanisms = {
        "IO": input_var(io_ids),
        "S": input_var(s_ids),
        "template_type": input_var(orders),
        # raw_input is an INPUT (parent-less): provided per example, never sampled. compute
        # returns None only as a guard; new_trace always sets it from the data. This is
        # what lets the traces be built directly, with no _values injection.
        "raw_input": Mechanism(parents=[], compute=lambda t: None),
        "result": Mechanism(parents=["IO"], compute=lambda t: t["IO"]),
        "raw_output": Mechanism(parents=["result"], compute=lambda t: t["result"]),
    }
    values = {"IO": io_ids, "S": s_ids, "template_type": orders,
              "raw_input": None, "result": io_ids, "raw_output": io_ids}
    idx = {v: i for i, v in enumerate(io_ids)}
    embeddings = {"result": (lambda v, _m=idx: [float(_m[v])])}
    causal_model = CausalModel(mechanisms, values, id="ioi", embeddings=embeddings)

    pool: dict[str, list] = defaultdict(list)
    for r in records:
        trace = causal_model.new_trace({
            "IO": r["IO"], "S": r["S"],
            "template_type": r["template_type"], "raw_input": r["raw_input"],
        })
        pool[r["template_type"]].append(trace)
    print(f"[IOI] causal model + pool: { {k: len(v) for k, v in pool.items()} } "
          f"| {len(io_ids)} IO ids, orders {orders}")
    return causal_model, dict(pool)


def create_ioi_counterfactual_dataset(
    pool: dict[str, list],
    n: int | None = None,
    seed: int = 42,
    categories: tuple[str, ...] = ("context_matched", "context_mismatched"),
    context_var: str = "template_type",
) -> list[CounterfactualExample]:
    """Pair real traces into a flat, category-labelled counterfactual dataset — the
    observational analogue of :func:`create_counterfactual_dataset`. Generic over the
    context axis: IOI passes ``context_var="template_type"`` (ABBA/BABA), the polysemantic
    benchmark passes ``context_var="task"`` (which case's algorithm), so context_matched =
    WITHIN-task and context_mismatched = CROSS-task with no code change.

    IOI sentences are fixed real data (the causal model can't generate them), so we PAIR
    from the pool rather than sampling the model. For each base trace, emit one CF per
    active category, all sharing the base:

        context_matched     source shares the base's TEMPLATE (ABBA->ABBA), different IO
                            -> a WITHIN-context patch.
        context_mismatched  source from the OTHER template (ABBA->BABA), different IO
                            -> an OUT-of-context patch.

    Each example is a ``CounterfactualExample`` (a dict) with two extra keys the
    interchange machinery ignores but the analysis reads: ``category`` and
    ``context={"template_type": base_template}``. The template label also lives on the
    base trace itself, so downstream grouping by ``manifold_cvars=["template_type"]`` fits
    and patches ONE manifold per template — the "two sets per context" you wanted, without
    a second dataset.

    The patch POSITION (subject vs result) is ORTHOGONAL to this dataset: the SAME pairs
    drive every position; where you patch is chosen later in ``build_grid``.
    """
    rng = random.Random(seed)
    contexts = list(pool.keys())
    bases = [t for ts in pool.values() for t in ts]
    if n is not None:
        rng.shuffle(bases)
        bases = bases[:n]

    examples: list[CounterfactualExample] = []
    for base in bases:
        ctx = base[context_var]
        other = [x for x in contexts if x != ctx]
        for cat in categories:
            if cat == "context_matched":
                src_pool = pool[ctx]
            elif cat == "context_mismatched":
                if not other:
                    continue  # only one context present -> no cross-context pairs
                src_pool = pool[rng.choice(other)]
            else:
                raise ValueError(f"unknown category {cat!r}")
            # source with a DIFFERENT result so the patch actually flips the answer
            cands = [s for s in src_pool if s["result"] != base["result"]]
            if not cands:
                continue
            src = rng.choice(cands)
            ex = CounterfactualExample(input=base, counterfactual_inputs=[src])
            ex["category"] = cat
            ex["context"] = {context_var: ctx}
            examples.append(ex)

    matched = sum(e["category"] == "context_matched" for e in examples)
    print(f"[cf {context_var}] {len(examples)} pairs | matched {matched} | "
          f"mismatched {len(examples) - matched} | contexts {contexts}")
    return examples


def create_directed_pairs(pool, base_task, target_task, n=None, seed=42, context_var="task"):
    """DIRECTED counterfactual pairs: base ∈ pool[base_task], source ∈ pool[target_task],
    source result != base result. Same inner pairing as create_ioi_counterfactual_dataset but
    with BOTH ends pinned (vs source = a random OTHER task), so every (base_task -> target_task)
    direction is a clean, separately-labelled scan — the axis the interaction regression needs.

    base_task == target_task -> within-task (the control/diagonal). Each example carries
    base_task / target_task / category / context so downstream rows are self-describing.
    """
    rng = random.Random(seed)
    bases = list(pool[base_task])
    if n is not None:
        rng.shuffle(bases)
        bases = bases[:n]
    src_pool = pool[target_task]
    category = "context_matched" if base_task == target_task else "context_mismatched"
    examples: list[CounterfactualExample] = []
    for base in bases:
        cands = [s for s in src_pool if s["result"] != base["result"] and s is not base]
        if not cands:
            continue
        src = rng.choice(cands)
        ex = CounterfactualExample(input=base, counterfactual_inputs=[src])
        ex["category"] = category
        ex["context"] = {context_var: base[context_var]}
        ex["base_task"] = base_task
        ex["target_task"] = target_task
        examples.append(ex)
    return examples


# --------------------------------------------------------------------------- #
# Polysemantic benchmark: observational, task-keyed (structurally == IOI)      #
# --------------------------------------------------------------------------- #
# A poly SIIT model routes on the TASK token at position 0; raw_input is
# [task, BOS, tok_0…tok_{L-1}], output is a per-position class, and the studied `result` is
# the class at target_pos (last non-PAD by default). Inputs are FIXED dataset sequences
# (can't be sampled — the union vocab causes OOV replacement), so — exactly like IOI — we
# build a real-trace POOL keyed by `task` and PAIR from it. context_matched = within-task,
# context_mismatched = CROSS-task (patch a component with the OTHER task's activation, the
# dormancy probe for shared/overlapping components).
def poly_records(inputs, labels, target_pos: int = -1) -> list[dict[str, Any]]:
    """Per-example poly facts from the dataset tensors.
    inputs : LongTensor [N, n_ctx] — task id at pos 0, BOS(=0) at pos 1, tokens after; PAD=1
             fills the tail. labels : per-position class [N, n_ctx] (argmax'd if one-hot).
    Returns [{task, raw_input (token-id list), result (class at target_pos), target_pos}].
    target_pos=-1 -> the last NON-PAD content position (>=2, token != PAD)."""
    import torch
    lab = labels.argmax(-1) if getattr(labels, "ndim", labels.dim()) == 3 else labels
    records: list[dict[str, Any]] = []
    for i in range(len(inputs)):
        toks = inputs[i].tolist() if torch.is_tensor(inputs[i]) else list(inputs[i])
        content = [p for p, t in enumerate(toks) if p >= 2 and t != 1]  # skip task, BOS, PAD
        tp = content[-1] if (target_pos == -1 and content) else (target_pos % len(toks))
        row_lab = lab[i].tolist() if torch.is_tensor(lab[i]) else list(lab[i])
        records.append({"task": int(toks[0]), "raw_input": toks,
                        "result": int(row_lab[tp]), "target_pos": tp})
    return records


def create_poly_causal_model(inputs, labels, target_pos: int = -1
                             ) -> tuple[CausalModel, dict[int, list]]:
    """Build the polysemantic ``CausalModel`` + its per-task trace pool in one pass — the
    poly counterpart to ``create_ioi_causal_model``.

    Variables
        task        which case's algorithm runs (position-0 router; the CONTEXT axis)
        raw_input   the token ids [task, BOS, tok…]   (an INPUT, provided per example)
        result      the studied per-position output class at target_pos   (fitted parameter)
        raw_output  = result

    Returns ``(causal_model, pool)`` with ``pool = {task: [CausalTrace]}`` for
    :func:`create_ioi_counterfactual_dataset(..., context_var="task")``."""
    from collections import defaultdict

    records = poly_records(inputs, labels, target_pos)
    tasks = sorted({r["task"] for r in records})
    results = sorted({r["result"] for r in records})

    mechanisms = {
        "task": input_var(tasks),
        "raw_input": Mechanism(parents=[], compute=lambda t: None),   # provided per example
        "result": input_var(results),
        "raw_output": Mechanism(parents=["result"], compute=lambda t: t["result"]),
    }
    values = {"task": tasks, "raw_input": None, "result": results, "raw_output": results}
    idx = {v: i for i, v in enumerate(results)}
    embeddings = {"result": (lambda v, _m=idx: [float(_m[v])])}
    causal_model = CausalModel(mechanisms, values, id="poly", embeddings=embeddings)

    pool: dict[int, list] = defaultdict(list)
    for r in records:
        trace = causal_model.new_trace({"task": r["task"], "result": r["result"]})
        trace._values["raw_input"] = r["raw_input"]
        pool[r["task"]].append(trace)
    print(f"[poly] causal model + pool: { {k: len(v) for k, v in pool.items()} } "
          f"| tasks {tasks}, {len(results)} result classes")
    return causal_model, dict(pool)


def ioi_subject_position_spec(name_ids: set[int]) -> Callable[[dict], dict]:
    """A DYNAMIC token-position spec for the IOI subject (S2), for injection via
    ``Task.template``. IOI templates differ in length, so S2 is NOT a fixed absolute
    index — this returns a per-example spec that scans ``raw_input`` for the 3rd name
    occurrence (= the repeated subject). Pair with ``build_grid(positions=["subject"])``
    to patch the s-inhibition / duplicate-token inputs instead of the last token.
    """
    def spec_func(input_sample: dict) -> dict:
        toks = input_sample["raw_input"]
        toks = toks.tolist() if hasattr(toks, "tolist") else list(toks)
        occ = [i for i, t in enumerate(toks) if t in name_ids]
        if len(occ) < 3:
            raise ValueError(f"IOI subject position: <3 name occurrences in {toks}")
        return {"type": "index", "position": occ[2]}  # S2 = 3rd name occurrence
    return spec_func




 
 


