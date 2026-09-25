"""
run_dormancy_pipeline.py
========================
InterpBench dormancy pipeline — the WRITER. Produces every artifact the dormancy
correlation needs, into the results_layout tree (results/{case}/...):

    PrepareData(case).generate()      -> cf_dataset (+ category/context labels),
                                          causal_model, manifold_cvars, hl_ll_corr, task, pipeline
    build_full_grid(...)              -> ONE grid over the WHOLE universe (all mlp + all
                                          heads + both residual streams), keyed by site.id
    circuit_specs(hl_ll_corr)         -> circuit.json (true membership per site + top-k cutoff)
    split cf by (category, context)
    generate_activation_manifold(global)      + per-context      -> activation_manifolds/{site}__{ctx}/
    get_intervened_representations(per mode)                     -> intervened_reps/{case}__{cat}__{ctx}/
    compute_patching_scores(per mode, per ctx) -> apparent_scores.json (the continuous dormancy signal)

The CORRELATION is a separate READER step (off-distribution-analyses.dormancy_analysis),
which reuses load_manifold / load_intervened_reps / projection+EMD and does the top-k
FP/FN + off-manifold-vs-apparent correlations per (mode, context). Run per case, then
optionally pool across cases (same full-grid granularity).

Run in the ib-venv (needs circuits_benchmark + forked causalab); one --position per run:
    python run_dormancy_pipeline.py --case 18  --patch-at corr --position last_token
    python run_dormancy_pipeline.py --case ioi --patch-at corr --position subject
    OFFDIST_RESULTS_DIR=results/18 python off-distribution-analyses.py --case 18
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

# --- make sibling modules importable (off_manifold here, interp_bench alongside) ---
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                      # off_manifold/*
sys.path.insert(0, str(_HERE.parent / "interp_bench"))  # interp_bench/*


# per-case results tree. results_layout freezes its paths from OFFDIST_RESULTS_DIR AT
# IMPORT, and site_ids repeat across models (two tracr models both have "L0_mlp..."), so we
# scope every artifact to results/{case}. This both stops manifolds colliding AND keeps each
# model's outputs — hence its dormancy correlation — separate. Must run BEFORE the
# activation_manifold/results_layout imports below.
def _prearg(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


_case_arg = str(_prearg("--case", "18"))
# poly runs share the case name "poly" but differ by which HF model -> namespace by poly_comb
# so results/poly_cases_0+1 and results/poly_cases_1+2 don't collide.
_tree = (f"poly_{_prearg('--poly-comb', 'cases_0+1')}" if _case_arg.lower() == "poly" else _case_arg)
os.environ.setdefault("OFFDIST_RESULTS_DIR", os.path.join("results", _tree))

# reused middle (off_manifold)
from activation_manifold import (
    generate_activation_manifold,
    get_intervened_representations,
    context_key,
)
from path_finder import build_full_grid, compute_patching_scores
from results_layout import INTERVENED_REPS, label_category_dir, dormancy_dir

# dataset builders (one place, all cases)
from context_datasets import (
    create_global_prompts,
    create_counterfactual_dataset,
    create_ioi_causal_model,
    create_ioi_counterfactual_dataset,
    create_poly_causal_model,
    ioi_name_ids,
    ioi_subject_position_spec,
)

# InterpBench glue (interp_bench)
import importlib
from interpbench_loading import build_pipelines, build_poly_pipeline, circuit_specs, poly_membership
from causalab.tasks.loader import load_task, Task


def load_case(case: str, device: str):
    """Load the case's task, correspondence, and low-level HookedTransformer.

    The case class name isn't uniform (numeric -> Case18, ioi -> CaseIOI), so we
    find the Case* class defined in the module rather than format it."""
    # lazy: circuits_benchmark only imports on the pod's ib-venv (torch<2.4); keeps
    # this module importable elsewhere (iit breaks on torch>=2.4 at import time).
    from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
        get_ll_model_loader,
    )

    modname = f"circuits_benchmark.benchmark.cases.case_{str(case).lower()}"
    mod = importlib.import_module(modname)
    cls = next(
        obj
        for n in dir(mod)
        if n.startswith("Case")
        for obj in [getattr(mod, n)]
        if isinstance(obj, type) and getattr(obj, "__module__", "") == modname
    )
    task = cls()
    loader = get_ll_model_loader(task, interp_bench=True)
    hl_ll_corr, model = loader.load_ll_model_and_correspondence(device=device)
    return task, hl_ll_corr, model


def load_poly_case(poly_comb: str, device: str):
    """Load a polysemantic-benchmark SIIT model + its correspondence DIRECTLY from HF (the
    trained weights — no training). ``poly_comb`` is an HF folder name, e.g. 'cases_0+1'.

    Returns ``(inputs, labels, hl_ll_corr, corr_mapping, ll_model, hl_model, n_tasks)``:
      inputs/labels  the dataset tensors used to build the trace pool (observational);
      hl_ll_corr     PolyHLModel.get_correspondence() — same LLNode(name,index) shape as
                     InterpBench's hl_ll_corr, so circuit_specs() works unchanged;
      corr_mapping   per-component per-task membership (shared vs disjoint circuits).
    """
    # lazy: poly_bench + evanhanders/iit only exist in the poly-venv (transformers 4.44,
    # a DIFFERENT iit fork than circuits_benchmark's) — keeps this module importable elsewhere.
    from poly_bench.io import load_from_hf
    from poly_bench.poly_hl_model import PolyModelDataset  # noqa: F401  (dataset builder)
    import torch

    ll_model, hl_model, dset_classes = load_from_hf(model_name=poly_comb, benchmarks_file="cases.json")
    ll_model = ll_model.to(device)
    hl_ll_corr = hl_model.get_correspondence()
    corr_mapping = hl_model.corr_mapping  
    n_tasks = len(getattr(hl_model, "n_ctx_list", [None]))  # one entry per combined case

    # dataset tensors (task at pos 0, per-position labels). ADAPT this accessor to the exact
    # poly_bench API in your notebook — this is the observational source for the trace pool.
    inputs, labels = _poly_dataset_tensors(hl_model, dset_classes, device)
    return inputs, labels, hl_ll_corr, corr_mapping, ll_model, hl_model, n_tasks

 
def _poly_dataset_tensors(hl_model, dset_classes, device, n_samples=800, seed=0):
    """Best-effort extraction of (inputs, labels) LongTensors for the loaded poly combo.
    VERIFY against the exact poly_bench dataset API you used in the load notebook — the base
    inputs and per-position labels are what feed create_poly_causal_model."""
    from poly_bench.poly_hl_model import PolyModelDataset
    import torch
    n_ctx = int(hl_model.cfg.n_ctx)
    # PolyModelDataset prepends the task-id token at pos 0 and REQUIRES each component
    # dataset's own n_ctx to be strictly LESS than the combined n_ctx (it pads
    # `n_ctx - dataset.n_ctx - 1` positions, which goes negative if they're equal). So the
    # component datasets are built at n_ctx-1 and the FULL n_ctx goes to the wrapper. The
    # colab's bare `n_ctx` passed to the dataset classes is cfg.n_ctx-1, not cfg.n_ctx —
    # that's why the colab writes `dsetcase(n_ctx=n_ctx)` but `PolyModelDataset(n_ctx=cfg.n_ctx)`.
    dsets = [C(N_samples=n_samples, n_ctx=n_ctx - 1, seed=seed) for C in dset_classes]
    poly_dataset = PolyModelDataset(dsets, n_ctx=n_ctx)
    dataset = poly_dataset.dataset
    inputs = torch.tensor(dataset.inputs, dtype=torch.int, device=device)   # [batch, seq]
    labels = torch.tensor(dataset.targets, dtype=torch.int, device=device) # [batch, seq, vocab] or [batch, seq]
    return inputs, labels 


# --------------------------------------------------------------------------- #
#  tracr shims
# --------------------------------------------------------------------------- #
def _attach_hf_config(model) -> None:
    """targets.py reads pipeline.model.config (HF-style); a HookedTransformer only
    has .cfg. Attach a scalar-only shim so build_*_targets work unchanged."""
    if getattr(model, "config", None) is not None:
        return
    c = model.cfg
    model.config = SimpleNamespace(
        hidden_size=c.d_model,
        num_attention_heads=c.n_heads,
        intermediate_size=c.d_mlp,
        head_dim=c.d_head,
        n_head=c.n_heads,
        name_or_path=f"interpbench",
    )


def _result_embeddings(out_vocab: list):
    """Minimal embedding map for the output variable: label -> [index]. Needed
    because generate_activation_manifold reads causal_model.embeddings for the
    intervention variable, and the interpbench CausalModel ships none."""
    idx = {v: i for i, v in enumerate(out_vocab)}
    return {"result": (lambda v, _m=idx: [float(_m[v])])}


# --------------------------------------------------------------------------- #
#  PrepareData: case -> counterfactual dataset (+ context labels) + causal model
# --------------------------------------------------------------------------- #
class PrepareData:
    """Turn an InterpBench case into everything the middle needs, in the exact
    CounterfactualExample shape (ex["input"] carries raw_input + context keys).

    Both cases build their counterfactuals through context_datasets so the analysis
    always sees category-labelled pairs: tracr via create_counterfactual_dataset
    (samples the CausalModel), IOI via create_ioi_counterfactual_dataset (pairs an
    existing IOIDataset pool by template)."""

    def __init__(self, case: str, n: int = 400, seed: int = 42, device: str = "cpu",
                 position: str = "last_token", poly_comb: str = "cases_0+1",
                 target_pos: int = -1):
        self.case = case
        self.n = n
        self.seed = seed
        self.device = device
        # WHERE the study patches. For IOI this selects a named token position
        # (last_token | subject | pos_k); it does NOT change the counterfactual dataset
        # (the same pairs drive every position — only build_grid's site moves).
        self.position = position
        self.poly_comb = poly_comb      # which HF poly model (cases_0+1, cases_1+2, …)
        self.target_pos = target_pos    # per-position output slot studied (-1 = last non-PAD)

    # -- load model + correspondence + pipeline (shared by all cases) --
    # ib_task is the circuits_benchmark task (loads model / get_clean_data). The
    # causalab Task (with create_token_positions + causal_model) is built per case.
    def _load(self):
        ib_task, hl_ll_corr, model = load_case(self.case, self.device)
        ll_pipeline, hl_pipeline, bos, hl_fn, vocab, out_vocab = build_pipelines(
            model, ib_task, device=self.device)
        _attach_hf_config(ll_pipeline.model)
        return ib_task, hl_ll_corr, model, ll_pipeline, bos, hl_fn, vocab, out_vocab

    def generate(self):
        c = str(self.case).lower()
        if c == "poly":
            return self._generate_poly()
        if c == "ioi":
            return self._generate_ioi()
        if c in ("gpt2-ioi", "gpt2_ioi"):
            return self._generate_gpt2_ioi()
        return self._generate_tracr()

    # ----------------------- tracr cases (18, etc.) ----------------------- #
    def _generate_tracr(self):
        """No natural contexts -> one global manifold (manifold_cvars=[]). Paired,
        category-labelled counterfactuals via create_counterfactual_dataset."""
        ib_task, hl_ll_corr, model, ll_pipeline, bos, hl_fn, vocab, out_vocab = self._load()
        n_ctx = int(model.cfg.n_ctx)
        seq_len = n_ctx - (1 if bos is not None else 0)
        # causalab Task: create_token_positions works, and causal_model (with the result
        # embedding) is built by CREATE_CAUSAL_MODEL. The ENTIRE config is model-derived
        # and passed via task_cfg (no CASES registry entry needed for a new case):
        # vocab/out_vocab from the tracr encoders, seq_len from n_ctx, hl_fn from the HL
        # program. target_pos defaults to -1 (add it here to study another position).
        task = load_task("interpbench", task_cfg={
            "case": self.case, "vocab": vocab, "out_vocab": out_vocab,
            "seq_len": seq_len, "bos": bos, "hl_fn": hl_fn,
        })
        causal_model = task.causal_model
        base_prompts = create_global_prompts(causal_model, n=self.n, seed=self.seed)
        cf_dataset = create_counterfactual_dataset(
            causal_model, base_prompts, conditioning_vars=[], n=self.n, seed=self.seed,
            categories=("context_matched",), target_variable="result",
            # tracr new_trace runs an HL-transformer FORWARD, so enumerating the input
            # space (default 50k) is millions of forwards (the case-13 hang). Force cheap
            # rejection sampling — "target differs" is easily satisfied by a random input.
            max_enumerate=64,
        )
        return dict(
            task=task, pipeline=ll_pipeline, causal_model=causal_model, hl_ll_corr=hl_ll_corr,
            cf_dataset=cf_dataset, manifold_cvars=[], intervention_variable="result",
            n_ctx=n_ctx, n_layers=int(model.cfg.n_layers), n_heads=int(model.cfg.n_heads),
        )

    # ------------------------------ IOI ---------------------------------- #
    def _generate_ioi(self):
        """Context axis = template_type (ABBA/BABA). The HL model is position/template
        dependent, so this is a real context. create_ioi_causal_model builds the schema +
        the per-template POOL of real CausalTraces in one pass; create_ioi_counterfactual_dataset
        pairs it into ONE flat, context-labelled cf_dataset (context_matched = same template,
        context_mismatched = cross-template). Downstream, manifold_cvars=["template_type"]
        splits that ONE dataset into a manifold per template — no second dataset needed."""
        ib_task, hl_ll_corr, model, ll_pipeline, bos, hl_fn, _vocab, _out_vocab = self._load()
        clean = ib_task.get_clean_data(max_samples=self.n)
        # ONE call: the IOI CausalModel (schema + result embedding) AND the per-template
        # pool of real CausalTraces, built straight from the fixed IOIDataset sentences.
        causal_model, pool = create_ioi_causal_model(clean)    # {template: [CausalTrace]}
        cf_dataset = create_ioi_counterfactual_dataset(pool, seed=self.seed)
        # IOI adds the subject (S2) as a DYNAMIC token position (its index varies per
        # sentence), injected via template so build_grid(positions=["subject"]) can patch
        # it; last_token / pos_k are built in. The cf_dataset is position-INDEPENDENT —
        # self.position only chooses build_grid's site downstream, not the pairs.
        extra_positions = {"subject": ioi_subject_position_spec(ioi_name_ids(clean))}
        if self.position not in ("last_token", "subject") and not self.position.startswith("pos_"):
            print(f"   [warn] IOI position {self.position!r} unknown (last_token|subject|pos_k)")
        task = Task(name="interpbench", causal_model=causal_model,
                    intervention_variable="result", template=extra_positions)
        return dict(
            task=task, pipeline=ll_pipeline, causal_model=causal_model, hl_ll_corr=hl_ll_corr,
            cf_dataset=cf_dataset, manifold_cvars=["template_type"],  # add "category" for axis 2
            intervention_variable="result",
            n_ctx=int(model.cfg.n_ctx), n_layers=int(model.cfg.n_layers), n_heads=int(model.cfg.n_heads),
        )

    # -------------------- POLYSEMANTIC benchmark (poly) ------------------- #
    def _generate_poly(self):
        """Context axis = task (which case's algorithm runs; the position-0 router). Loaded
        DIRECTLY from HF (trained weights, no training). Structurally == IOI: build a
        per-task pool of real CausalTraces from the dataset, then pair within-task
        (context_matched) vs CROSS-task (context_mismatched) via the SAME
        create_ioi_counterfactual_dataset, keyed on `task`. Downstream,
        manifold_cvars=["task"] fits one manifold per task. The correspondence comes from
        PolyHLModel (== hl_ll_corr shape) plus corr_mapping for shared-vs-disjoint labels."""
        inp, lab, hl_ll_corr, corr_mapping, ll_model, hl_model, n_tasks = load_poly_case(
            self.poly_comb, self.device)
        _attach_hf_config(ll_model)
        ll_pipeline = build_poly_pipeline(ll_model, device=self.device)

        causal_model, pool = create_poly_causal_model(inp, lab, target_pos=self.target_pos)
        cf_dataset = create_ioi_counterfactual_dataset(pool, n=self.n, seed=self.seed,
                                                       context_var="task")  # within vs cross task
        task = Task(name="interpbench", causal_model=causal_model,
                    intervention_variable="result", template=None)
        return dict(
            task=task, pipeline=ll_pipeline, causal_model=causal_model, hl_ll_corr=hl_ll_corr,
            cf_dataset=cf_dataset, manifold_cvars=["task"], intervention_variable="result",
            n_ctx=int(ll_model.cfg.n_ctx), n_layers=int(ll_model.cfg.n_layers),
            n_heads=int(ll_model.cfg.n_heads),
            # poly-only extras: per-component shared/disjoint membership (with the per-task
            # 'tasks' list) for the richer analysis, and the task-keyed trace POOL + task ids
            # so interaction_effects can rebuild DIRECTED (base_task -> target_task) pairs and
            # fit a clean manifold per task (create_directed_pairs / per-task clouds).
            poly_membership=(poly_membership(corr_mapping, n_tasks)),
            pool=pool, tasks=sorted(pool.keys()),
        )

    # ------------------- GPT-2 small + IOI (Wang et al.) -------------------- #
    def _generate_gpt2_ioi(self):
        """GPT-2-small on the Indirect Object Identification task. The model is
        loaded directly via TransformerLens (no InterpBench), and the Wang et al.
        2022 circuit provides ground-truth labels. Templates (ABBA/BABA) serve as
        the task axis, so the directed sweep produces within-template and
        cross-template scans matching the poly_bench design."""
        from gpt2_ioi import (
            load_gpt2_ioi_model, build_gpt2_ioi_pool, build_gpt2_ioi_pipeline,
            gpt2_ioi_hl_ll_corr, gpt2_ioi_home_tasks,
        )

        model = load_gpt2_ioi_model(device=self.device)
        _attach_hf_config(model)
        pipeline = build_gpt2_ioi_pipeline(model, device=self.device)
        causal_model, pool = build_gpt2_ioi_pool(model, n=self.n, seed=self.seed)
        hl_ll_corr = gpt2_ioi_hl_ll_corr()
        home = gpt2_ioi_home_tasks(model)

        cf_dataset = create_ioi_counterfactual_dataset(
            pool, n=self.n, seed=self.seed, context_var="template_type")

        task = Task(name="interpbench", causal_model=causal_model,
                    intervention_variable="result", template=None)

        return dict(
            task=task, pipeline=pipeline, causal_model=causal_model,
            hl_ll_corr=hl_ll_corr, cf_dataset=cf_dataset,
            manifold_cvars=["template_type"], intervention_variable="result",
            n_ctx=int(model.cfg.n_ctx), n_layers=int(model.cfg.n_layers),
            n_heads=int(model.cfg.n_heads),
            poly_membership=home, pool=pool, tasks=sorted(pool.keys()),
        )


# --------------------------------------------------------------------------- #
#  writer helpers (dormancy is CORRELATED downstream in off-distribution-analyses)
# --------------------------------------------------------------------------- #
def _site_to_spec(site):
    """AtomicModelUnit -> (layer, comp, head|None) matching circuit_specs' keys."""
    ct = site.component_type
    if ct in ("attn_result", "head_attention_value_output"):
        return (site.layer, "attn_result", getattr(site, "head", None))
    if ct in ("mlp_output", "mlp_activation"):
        return (site.layer, "mlp_activation", None)
    return (site.layer, "block_output" if "output" in ct else "block_input", None)


def _split_by_category_context(cf_dataset, manifold_cvars):
    """{(category, context): [examples]}. Two orthogonal axes:
      category = ex['category'] — the SOURCE relationship (context_matched / mismatched).
      context  = context_key(ex, manifold_cvars) — the MANIFOLD axis ('global' for a tracr
                 case with no context; the ABBA/BABA tuple for IOI).
    Each (category, context) gets its own reps dir + apparent scores, so dormancy is
    reported per (mode, context) exactly as the analysis expects."""
    groups: dict = {}
    for ex in cf_dataset:
        category = ex.get("category", "context_matched")
        context = context_key(ex, manifold_cvars)
        groups.setdefault((category, context), []).append(ex)
    return groups


def _save_circuit(is_circuit, in_circuit, k):
    """circuit.json (per case, context-independent): true membership per grid site + the
    top-k cutoff k = |circuit ∩ grid|. dormancy_analysis reads this to label FP/FN."""
    os.makedirs(dormancy_dir(), exist_ok=True)
    with open(os.path.join(dormancy_dir(), "circuit.json"), "w") as f:
        json.dump({"k": k, "in_circuit": in_circuit, "is_circuit": is_circuit}, f, indent=2)


def _save_apparent(label, category, context, scores):
    """apparent_scores.json next to the reps: {mode: {site_id: apparent_score}} — the
    CONTINUOUS dormancy signal for this (category, context)."""
    rep_dir = label_category_dir(INTERVENED_REPS, label, category, context)
    os.makedirs(rep_dir, exist_ok=True)
    with open(os.path.join(rep_dir, "apparent_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)


# --------------------------------------------------------------------------- #
#  orchestration — WRITER. Emits reps + manifolds + apparent/circuit json into the
#  results_layout tree; the dormancy correlation runs afterwards in
#  off-distribution-analyses.dormancy_analysis(case=...).
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="18",
                    help="tracr case id (e.g. 18), 'ioi', or 'poly' (polysemantic benchmark)")
    ap.add_argument("--poly-comb", default="cases_0+1",
                    help="poly only: which HF model to load (cases_0+1, cases_1+2, cases_0+1+2+3, …)")
    ap.add_argument("--target-pos", type=int, default=-1,
                    help="poly only: which per-position output slot to study (-1 = last non-PAD)")
    ap.add_argument("--patch-at", default="corr", choices=["corr", "write", "internal"])
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian"],
                    help="STR/Gaussian only — DAS has no ground-truth circuit to score")
    ap.add_argument("--position", default="last_token",
                    help="run-level: WHERE to patch (last_token|subject|pos_k). One per run "
                         "— sweep positions across runs, each into its own results tree.")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--k-features", type=int, default=64)
    ap.add_argument("--metric", default="flip", choices=["flip", "logit_diff"],
                    help="apparent-circuit readout: flip=argmax flip (coarse, tracr-friendly); "
                         "logit_diff=continuous logit[base]-logit[src] causal effect (IOI-style, "
                         "sensitive to backup/graded effects; needs integer result indices)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seeds counterfactual sampling + Gaussian noise + manifold fit; "
                         "sweep it (into separate result trees) for seed-variance error bars")
    ap.add_argument("--no-residual", dest="include_residual", action="store_false",
                    help="drop residual sites (default: INCLUDE, for the full hl_ll_corr "
                         "correspondence — resid_pre/resid_post are real circuit nodes). For "
                         "TRACR cases pass this: residual is the carrier bus, not a labelled node")
    ap.set_defaults(include_residual=True)
    args = ap.parse_args()
    if any(m.upper() == "DAS" for m in args.modes):
        raise SystemExit("DAS excluded: no ground-truth circuit for DAS. Use --modes STR Gaussian.")

    from activation_manifold import set_seed
    set_seed(args.seed)  # python/np/torch — covers Gaussian noise + any fit randomness

    label = str(args.case)
    print(f"== 1. prepare data (case {label}, seed {args.seed}) | results -> {os.environ['OFFDIST_RESULTS_DIR']} ==")
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.poly_comb,
                    target_pos=args.target_pos).generate()
    task, pipeline, causal_model = d["task"], d["pipeline"], d["causal_model"]
    m_cvars, ivar = d["manifold_cvars"], d["intervention_variable"]

    print(f"== 2. build FULL grid (all mlp + heads"
          f"{' + residual' if args.include_residual else ''}, patch_at={args.patch_at}, "
          f"position={args.position}) ==")
    layers = args.layers if args.layers is not None else list(range(d["n_layers"]))
    grid = build_full_grid(pipeline, task, layers, positions=[args.position],
                           patch_at=args.patch_at, include_residual=args.include_residual)
    union_sites = [t.flatten()[0] for t in grid.values()]
    print(f"   {len(union_sites)} sites")

    # true circuit membership per site (context-INDEPENDENT) + top-k cutoff
    circuit = circuit_specs(d["hl_ll_corr"])
    is_circuit = {s.id: (_site_to_spec(s) in circuit) for s in union_sites}
    in_circuit = [sid for sid, yes in is_circuit.items() if yes]
    k = len(in_circuit)
    print(f"   circuit sites in grid (k={k}): {in_circuit}")
    _save_circuit(is_circuit, in_circuit, k)

    print("== 3. split cf by (category, context) ==")
    groups = _split_by_category_context(d["cf_dataset"], m_cvars)
    print(f"   groups: { {f'{cat}/{ctx}': len(v) for (cat, ctx), v in groups.items()} }")



    print("== 4. fit GLOBAL manifold (all bases, one per site) ==")
    generate_activation_manifold(
        union_sites, d["cf_dataset"], [], pipeline, causal_model,
        batch_size=args.batch_size, k_features=args.k_features,
        intervention_variable=ivar, task=task, seed=args.seed)

    print("== 5. per (category, context): contextual manifold + reps + apparent scores ==")
    for (category, context), cf_ctx in groups.items():
        print(f"  -- {category} / {context}: {len(cf_ctx)} examples --")
        if m_cvars:  # contextual manifold (IOI ABBA/BABA); a global-only case skips this
            generate_activation_manifold(
                union_sites, cf_ctx, m_cvars, pipeline, causal_model,
                batch_size=args.batch_size, k_features=args.k_features,
                intervention_variable=ivar, task=task, seed=args.seed)
        rep_dir = label_category_dir(INTERVENED_REPS, label, category, context)
        os.makedirs(rep_dir, exist_ok=True)
        for mode in args.modes:
            get_intervened_representations(
                union_sites, cf_ctx, pipeline, mode, causal_model=causal_model,
                k_features=args.k_features, batch_size=args.batch_size, output_dir=rep_dir)
        # apparent-circuit scores (the continuous dormancy signal), per mode, per context
        scores = compute_patching_scores(grid, cf_ctx, pipeline, args.modes, args.batch_size,
                                          metric=args.metric)
        _save_apparent(label, category, context, scores)

    print(f"DONE (writer). Correlate with:\n"
          f"  OFFDIST_RESULTS_DIR={os.environ['OFFDIST_RESULTS_DIR']} "
          f"python off-distribution-analyses.py --case {label}")


if __name__ == "__main__":
    main()
