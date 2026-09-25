"""
gpt2_ioi.py
===========
GPT-2 small + IOI (Indirect Object Identification) integration for the
dormancy instrument pipeline.

Loads GPT-2-small via TransformerLens, builds IOI counterfactual pairs from
the IOIDataset, and provides Wang et al. (2022) circuit ground truth in the
same format the interaction-effects pipeline consumes.

The model is a HookedTransformer, so TracPipeline wraps it directly —
encode/decode are the only GPT-2-specific pieces.  The "tasks" axis uses
template type (ABBA / BABA) so the directed sweep produces within-template
and cross-template scans, matching the poly_bench base_task x target_task
design.

Pod-only: needs TransformerLens + GPU for any real sweep.  The model download
(~500 MB) runs once and is cached by HuggingFace.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import torch

# ---------------------------------------------------------------------------
#  Wang et al. 2022 IOI circuit — (layer, head) for attention heads
# ---------------------------------------------------------------------------
# From "Interpretability in the Wild" Table 1 / Figure 3.
# Three functional groups; every head listed is in the circuit.
#   Name Movers (positive):  9.9, 10.0, 9.6
#   Negative Name Movers:    10.7, 11.10
#   S-Inhibition Heads:      7.3, 7.9, 8.6, 8.10
#   Induction Heads:         5.5, 6.9
#   Duplicate Token Heads:   0.1, 3.0
#   Previous Token Heads:    2.2, 4.11
# Backup Name Movers (9.0, 9.7, 10.1, 10.2, 10.6, 10.10, 11.2, 11.9)
# are included in the "extended" circuit; we include them for a 26-head
# ground truth that matches the standard used by Zhang et al. 2023.

IOI_CIRCUIT_CORE: set[tuple[int, int]] = {
    # Name Movers
    (9, 9), (10, 0), (9, 6),
    # Negative Name Movers
    (10, 7), (11, 10),
    # S-Inhibition Heads
    (7, 3), (7, 9), (8, 6), (8, 10),
    # Induction Heads
    (5, 5), (6, 9),
    # Duplicate Token Heads
    (0, 1), (3, 0),
    # Previous Token Heads
    (2, 2), (4, 11),
}

IOI_CIRCUIT_BACKUP: set[tuple[int, int]] = {
    (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9),
}

IOI_CIRCUIT_EXTENDED = IOI_CIRCUIT_CORE | IOI_CIRCUIT_BACKUP

# MLPs are not part of the identified IOI circuit in Wang et al.
# but we include MLP0 as a candidate push site (it processes the
# token embeddings before any attention head acts).
IOI_CIRCUIT_MLPS: set[int] = set()  # no MLPs in the Wang et al. circuit


# ---------------------------------------------------------------------------
#  Model loading
# ---------------------------------------------------------------------------
def load_gpt2_ioi_model(device: str = "cuda"):
    """Load GPT-2 small via TransformerLens with attn_result hooks enabled."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2",
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=False,
        device=device,
    )
    model.set_use_attn_result(True)
    return model


# ---------------------------------------------------------------------------
#  IOI sentence generation (self-contained, no external IOIDataset needed)
# ---------------------------------------------------------------------------
# Wang et al. 2022 sentence templates. {IO} = indirect object, {S} = subject.
# ABBA: IO mentioned first, then S appears, then S repeats.
# BABA: S mentioned first, then IO, then S repeats.
_ABBA_TEMPLATES = [
    "Then, {IO} and {S} went to the store. {S} gave a drink to",
    "Then, {IO} and {S} had a long argument. {S} said sorry to",
    "Then, {IO} and {S} were in the room. {S} gave a present to",
    "Then, {IO} and {S} went to the park. {S} told a story to",
    "Then, {IO} and {S} had a meeting. {S} handed a paper to",
]
_BABA_TEMPLATES = [
    "Then, {S} and {IO} went to the store. {S} gave a drink to",
    "Then, {S} and {IO} had a long argument. {S} said sorry to",
    "Then, {S} and {IO} were in the room. {S} gave a present to",
    "Then, {S} and {IO} went to the park. {S} told a story to",
    "Then, {S} and {IO} had a meeting. {S} handed a paper to",
]

# Common first names that tokenize to a single token in GPT-2
_NAMES = [
    "Michael", "James", "Robert", "John", "David", "William", "Richard",
    "Joseph", "Thomas", "Charles", "Christopher", "Daniel", "Matthew",
    "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
    "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
    "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna",
]


def _generate_ioi_records(
    tokenizer, n: int = 400, seed: int = 42, prepend_bos: bool = True,
) -> list[dict[str, Any]]:
    """Generate IOI sentence records with template-based construction.

    Each record has: template_type, IO (token id), S (token id),
    result (IO token id), raw_input (token id list), pos_IO, pos_S1,
    pos_S2, pos_end.
    """
    rng = random.Random(seed)
    records = []

    for _ in range(n):
        template_type = rng.choice(["ABBA", "BABA"])
        templates = _ABBA_TEMPLATES if template_type == "ABBA" else _BABA_TEMPLATES
        template = rng.choice(templates)

        io_name, s_name = rng.sample(_NAMES, 2)
        sentence = template.format(IO=io_name, S=s_name)

        toks = tokenizer.encode(sentence)
        if prepend_bos:
            bos = tokenizer.bos_token_id
            if bos is not None and (not toks or toks[0] != bos):
                toks = [bos] + toks

        io_tok = tokenizer.encode(" " + io_name)
        s_tok = tokenizer.encode(" " + s_name)
        if not io_tok or not s_tok:
            continue
        io_id = io_tok[0]
        s_id = s_tok[0]

        # locate positions: find each name's first occurrence in toks
        # For ABBA: IO appears first, then S (first mention = S1), then S again (S2)
        # For BABA: S appears first (S1), then IO, then S again (S2)
        io_positions = [i for i, t in enumerate(toks) if t == io_id]
        s_positions = [i for i, t in enumerate(toks) if t == s_id]
        if not io_positions or len(s_positions) < 2:
            continue

        pos_IO = io_positions[0]
        pos_S1 = s_positions[0]
        pos_S2 = s_positions[1]
        pos_end = len(toks) - 1

        records.append({
            "template_type": template_type,
            "IO": io_id,
            "S": s_id,
            "result": io_id,
            "raw_input": toks,
            "pos_IO": pos_IO,
            "pos_S1": pos_S1,
            "pos_S2": pos_S2,
            "pos_end": pos_end,
        })

    return records


def build_gpt2_ioi_pool(model, n: int = 400, seed: int = 42):
    """Build the CausalModel + template-keyed pool of CausalTraces.

    Returns (causal_model, pool) where pool = {"ABBA": [...], "BABA": [...]},
    matching the poly_bench pattern so create_directed_pairs / build_task_clouds
    work unchanged.
    """
    from causalab.causal.causal_model import CausalModel, Mechanism

    def input_var(vals):
        return Mechanism(parents=[], compute=lambda t: None)

    records = _generate_ioi_records(
        model.tokenizer, n=n, seed=seed,
        prepend_bos=model.cfg.default_prepend_bos,
    )

    io_ids = sorted({r["IO"] for r in records})
    s_ids = sorted({r["S"] for r in records})
    templates = sorted({r["template_type"] for r in records})

    mechanisms = {
        "IO": input_var(io_ids),
        "S": input_var(s_ids),
        "template_type": input_var(templates),
        "raw_input": Mechanism(parents=[], compute=lambda t: None),
        "result": Mechanism(parents=["IO"], compute=lambda t: t["IO"]),
        "raw_output": Mechanism(parents=["result"], compute=lambda t: t["result"]),
    }
    values = {
        "IO": io_ids, "S": s_ids, "template_type": templates,
        "raw_input": None, "result": io_ids, "raw_output": io_ids,
    }
    idx = {v: i for i, v in enumerate(io_ids)}
    embeddings = {"result": (lambda v, _m=idx: [float(_m[v])])}
    causal_model = CausalModel(mechanisms, values, id="gpt2-ioi", embeddings=embeddings)

    pool: dict[str, list] = defaultdict(list)
    for r in records:
        trace = causal_model.new_trace({
            "IO": r["IO"], "S": r["S"],
            "template_type": r["template_type"],
            "raw_input": r["raw_input"],
        })
        # Stash position metadata on the trace for downstream position handling
        trace["pos_IO"] = r["pos_IO"]
        trace["pos_S1"] = r["pos_S1"]
        trace["pos_S2"] = r["pos_S2"]
        trace["pos_end"] = r["pos_end"]
        pool[r["template_type"]].append(trace)

    print(f"[gpt2-ioi] pool: {dict((k, len(v)) for k, v in pool.items())} "
          f"| {len(io_ids)} IO ids, templates {templates}")
    return causal_model, dict(pool)


# ---------------------------------------------------------------------------
#  Pipeline construction
# ---------------------------------------------------------------------------
def build_gpt2_ioi_pipeline(model, device: str = "cuda"):
    """Wrap GPT-2-small as a TracrPipeline.

    encode: raw_input is already a token-id list (from the IOIDataset).
    decode: token ids → token ids (the readout compares logit columns directly).
    """
    from tracr_pipeline import TracrPipeline

    def encode(raw_input):
        return raw_input.tolist() if hasattr(raw_input, "tolist") else list(raw_input)

    def decode(ids):
        return [int(i) for i in ids]

    return TracrPipeline(
        model,
        encode=encode,
        decode=decode,
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        device=device,
    )


# ---------------------------------------------------------------------------
#  Circuit ground truth in the pipeline's format
# ---------------------------------------------------------------------------
def gpt2_ioi_hl_ll_corr(circuit: set[tuple[int, int]] | None = None):
    """Build a synthetic hl_ll_corr that circuit_specs() can consume.

    Returns {hl_node: [ll_node, ...]} where each node is a SimpleNamespace
    with .name (the TL hook string) and .index (head index or None for MLPs).
    circuit_specs iterates the LL nodes, calls _hook_to_spec on node.name +
    node.index, and collects the resulting (layer, comp, head) tuples.
    """
    from types import SimpleNamespace

    if circuit is None:
        circuit = IOI_CIRCUIT_EXTENDED

    corr = {}
    for layer, head in circuit:
        hl_key = f"hl_L{layer}H{head}"
        hook = f"blocks.{layer}.attn.hook_result"
        idx = [slice(None), slice(None), head, slice(None)]
        ll_node = SimpleNamespace(name=hook, index=idx)
        corr[hl_key] = [ll_node]

    return corr


def gpt2_ioi_home_tasks(
    model,
    circuit: set[tuple[int, int]] | None = None,
    templates: list[str] | None = None,
) -> dict:
    """Per-component home_tasks mapping.

    For GPT-2 IOI, all circuit heads serve both templates (ABBA and BABA) —
    there are no template-specific sub-circuits.  Non-circuit components get
    an empty list.

    Returns {(layer, comp, head|None): {"tasks": [...], "label": ...}}
    keyed by ComponentSpec, same shape as poly_membership.
    """
    if circuit is None:
        circuit = IOI_CIRCUIT_EXTENDED
    if templates is None:
        templates = ["ABBA", "BABA"]

    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    out = {}
    for layer in range(n_layers):
        # MLP
        spec = (layer, "mlp_activation", None)
        in_circuit = layer in IOI_CIRCUIT_MLPS
        out[spec] = {
            "tasks": list(templates) if in_circuit else [],
            "label": "shared" if in_circuit else "none",
        }
        # Attention heads
        for head in range(n_heads):
            spec = (layer, "attn_result", head)
            in_circuit = (layer, head) in circuit
            out[spec] = {
                "tasks": list(templates) if in_circuit else [],
                "label": "shared" if in_circuit else "none",
            }
    return out


# ---------------------------------------------------------------------------
#  Target position helpers
# ---------------------------------------------------------------------------
def ioi_end_positions(cf_pairs: list[dict]) -> torch.Tensor:
    """Per-example END token position (where the model predicts the IO name).

    Falls back to the last non-padding position if pos_end is not stored.
    """
    out = []
    for ex in cf_pairs:
        inp = ex["input"] if "input" in ex else ex
        pos = inp.get("pos_end")
        if pos is not None:
            out.append(int(pos))
        else:
            # fallback: last token position
            toks = inp.get("raw_input", [])
            out.append(len(toks) - 1)
    return torch.tensor(out, dtype=torch.long)


def pick_ioi_push_site(grid, layer: int = 0):
    """Select a push site for the IOI case.

    Default: MLP at layer 0 (MLP0 processes token embeddings before any
    attention head acts, making it the broadest upstream perturbation point).
    """
    units = {t.flatten()[0].id: t.flatten()[0] for t in grid.values()}
    mlp_candidates = [
        (uid, u) for uid, u in units.items()
        if "mlp" in u.component_type and int(u.layer) == layer
    ]
    if mlp_candidates:
        return mlp_candidates[0][1]
    # fallback: any MLP
    mlp_any = [(uid, u) for uid, u in units.items() if "mlp" in u.component_type]
    if mlp_any:
        return mlp_any[0][1]
    raise RuntimeError("No MLP site found in grid for IOI push")
