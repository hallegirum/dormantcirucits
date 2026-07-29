"""
run_dormancy_pipeline.py
========================
End-to-end InterpBench dormancy pipeline, one file:

    PrepareData(case).generate()      -> cf_dataset (+ context labels), causal_model,
                                          manifold_cvars, hl_ll_corr, task, pipeline
    build_grid(...)                   -> grid of interchange targets (residual|mlp|attention)
    for mode in modes:
        generate_activation_manifold(...)     [reused] fit manifold per (site, context)
        get_intervened_representations(...)   [reused] h_intervened per site
        collect_patched_distributions(...)    [reused, optional] behaviour side
    correlate_divergences(...)        -> per-site off-manifold divergence x circuit membership

Only the two ends are new (data in / correlation out) + a couple of tracr shims;
the manifold/rep middle is your existing code, unchanged.

Run in the ib-venv (needs circuits_benchmark + forked causalab):
    python run_dormancy_pipeline.py --case 18 --component mlp --patch-at write
    python run_dormancy_pipeline.py --case ioi --component attention --patch-at write
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
        get_ll_model_loader,
    )
import torch
from path_finder import compute_patching_scores
# --- make sibling modules importable (off_manifold here, interp_bench alongside) ---
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                      # off_manifold/*
sys.path.insert(0, str(_HERE.parent / "interp_bench"))  # interp_bench/*

# reused middle (off_manifold)
from activation_manifold import (
    generate_activation_manifold,
    get_intervened_representations,
)
from collect_behaviour import collect_patched_distributions
from path_finder import build_grid
from results_layout import site_context_dir

# dataset builders (one place, all cases)
from context_datasets import (
    create_global_prompts,
    create_counterfactual_dataset,
    create_ioi_counterfactual_dataset,
)

# InterpBench glue (interp_bench)
import importlib
from interpbench_loading import build_pipelines, circuit_specs
from causalab.tasks.interpbench.causal_models import CASES, create_interpbench_causal_model


def load_case(case: str, device: str):
    """Load the case's task, correspondence, and low-level HookedTransformer."""
    mod = importlib.import_module(f"circuits_benchmark.benchmark.cases.case_{case}")
    task = getattr(mod, f"Case{case}")()
    loader = get_ll_model_loader(task, interp_bench=True)
    hl_ll_corr, model = loader.load_ll_model_and_correspondence(device=device)
    return task, hl_ll_corr, model


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

    def __init__(self, case: str, n: int = 400, seed: int = 42, device: str = "cpu"):
        self.case = case
        self.n = n
        self.seed = seed
        self.device = device

    # -- load model + correspondence + pipeline (shared by all cases) --
    def _load(self):
        task, hl_ll_corr, model = load_case(self.case, self.device)
        ll_pipeline, hl_pipeline, bos, hl_fn = build_pipelines(model, task, device=self.device)
        _attach_hf_config(ll_pipeline.model)
        return task, hl_ll_corr, model, ll_pipeline, bos, hl_fn

    def generate(self):
        return self._generate_ioi() if self.case == "ioi" else self._generate_tracr()

    # ----------------------- tracr cases (18, etc.) ----------------------- #
    def _generate_tracr(self):
        """No natural contexts -> one global manifold (manifold_cvars=[]). Paired,
        category-labelled counterfactuals via create_counterfactual_dataset."""
        task, hl_ll_corr, model, ll_pipeline, bos, hl_fn = self._load()
        n_ctx = int(model.cfg.n_ctx)
        cfg = dataclasses.replace(
            CASES[self.case], bos=bos, seq_len=n_ctx - (1 if bos is not None else 0), hl_fn=hl_fn,
        )
        causal_model = create_interpbench_causal_model(cfg)
        # embeddings for the output variable (label -> index) so manifolds can fit
        causal_model.embeddings = {**(causal_model.embeddings or {}), **_result_embeddings(cfg.out_vocab)}
        base_prompts = create_global_prompts(causal_model, n=self.n, seed=self.seed)
        cf_dataset = create_counterfactual_dataset(
            causal_model, base_prompts, conditioning_vars=[], n=self.n, seed=self.seed,
            categories=("context_matched",), target_variable="result",
        )
        return dict(
            task=task, pipeline=ll_pipeline, causal_model=causal_model, hl_ll_corr=hl_ll_corr,
            cf_dataset=cf_dataset, manifold_cvars=[], intervention_variable="result",
            n_ctx=n_ctx, n_layers=int(model.cfg.n_layers), n_heads=int(model.cfg.n_heads),
        )

    # ------------------------------ IOI ---------------------------------- #
    def _generate_ioi(self):
        """Two context axes: template_type (ABBA/BABA) and category. The HL model
        is position/template dependent, so these are real contexts. Paired,
        category-labelled counterfactuals via create_ioi_counterfactual_dataset.

        NOTE: the IOIDataset fields (template_type, category, IO name) must be read
        off the real object on the pod — the accessors below are the intended shape."""
        task, hl_ll_corr, model, ll_pipeline, bos, hl_fn = self._load()
        clean = task.get_clean_data(max_samples=self.n)
        inputs = clean.get_inputs()
        traces = []
        for i in range(len(inputs)):
            meta = _ioi_meta(clean, i)  # {"template_type","category","result"}
            traces.append({
                "raw_input": _ioi_raw_input(inputs, i),
                "template_type": meta["template_type"],
                "category": meta["category"],
                "result": meta["result"],
            })
        cf_dataset = create_ioi_counterfactual_dataset(traces, seed=self.seed)
        causal_model = _ioi_causal_model_shim(traces)  # .embeddings/.periods/.values
        return dict(
            task=task, pipeline=ll_pipeline, causal_model=causal_model, hl_ll_corr=hl_ll_corr,
            cf_dataset=cf_dataset, manifold_cvars=["template_type"],  # add "category" for axis 2
            intervention_variable="result",
            n_ctx=int(model.cfg.n_ctx), n_layers=int(model.cfg.n_layers), n_heads=int(model.cfg.n_heads),
        )


# ---- IOI helpers (confirm field names on the pod) ------------------------- #
# The accessors below are guesses at the IOIDataset / TracrEncodedDataset API.
# They print the real structure once (i == 0) so you can read off the true field
# names on the pod and fix the accessors, rather than guessing blind.
def _ioi_meta(clean, i):  # pragma: no cover - pod-only
    md = getattr(clean, "metadata", None)
    if i == 0:
        print("\n[IOI introspect] clean =", type(clean).__name__)
        print("[IOI introspect]   public attrs:",
              [a for a in dir(clean) if not a.startswith("_")])
        print("[IOI introspect]   metadata type:", type(md).__name__ if md is not None else None)
        if md is not None:
            print("[IOI introspect]   metadata[0]:", md[0])
            if hasattr(md[0], "keys"):
                print("[IOI introspect]   metadata[0] keys:", list(md[0].keys()))
    tmpl = (md[i].get("template_type") if md is not None else None) or "ABBA"
    return {"template_type": tmpl, "category": "names", "result": _ioi_io_name(clean, i)}


def _ioi_raw_input(inputs, i):  # pragma: no cover
    x = inputs[i]
    if i == 0:
        print("[IOI introspect] inputs =", type(inputs).__name__, "len", len(inputs))
        print("[IOI introspect]   inputs[0] type:", type(x).__name__, "value:", repr(x)[:160])
    return x.tolist() if torch.is_tensor(x) else list(x)


def _ioi_io_name(clean, i):  # pragma: no cover
    tgt = clean.get_targets()
    if i == 0:
        print("[IOI introspect] targets =", type(tgt).__name__,
              "| targets[0]:", repr(tgt[0])[:120], "\n")
    return int(tgt[i]) if torch.is_tensor(tgt) else tgt[i]


def _ioi_causal_model_shim(traces):  # pragma: no cover
    names = sorted({t["result"] for t in traces})
    idx = {v: i for i, v in enumerate(names)}
    return SimpleNamespace(
        embeddings={"result": (lambda v, _m=idx: [float(_m[v])])},
        periods={},
        values={"result": names},
    )


# --------------------------------------------------------------------------- #
#  correlate: off-manifold divergence x apparent circuit x true circuit
# --------------------------------------------------------------------------- #
def correlate_divergences(grid, reps_by_mode, patching_scores, causal_model, cf_dataset,
                          manifold_cvars, hl_ll_corr, out_path, apparent_thresh=0.5):
    """The dormancy table: one row per site joining THREE things —
      is_circuit      : true membership (hl_ll_corr)
      apparent_{mode} : the patching method's own score (compute_patching_scores)
      offmanifold_{mode}: the intervention's off-manifold magnitude
    and the derived `dormancy_{mode}` = (apparent high) AND (NOT is_circuit) — a
    component the patching method calls in-circuit that actually isn't. The
    off-manifold characteristics are what should PREDICT that flag."""
    circuit = circuit_specs(hl_ll_corr)
    rows = []
    for key, target in grid.items():
        site = target.flatten()[0]
        spec = _site_to_spec(site)
        row = {"site": site.id, "key": list(key), "layer": spec[0], "component": spec[1],
               "head": spec[2], "is_circuit": spec in circuit}
        for mode, reps in reps_by_mode.items():
            row[f"offmanifold_{mode}"] = _offmanifold(site, reps[site.id], cf_dataset, manifold_cvars)
            score = float(patching_scores.get(mode, {}).get(key, float("nan")))
            row[f"apparent_{mode}"] = score
            row[f"dormancy_{mode}"] = bool(score >= apparent_thresh and not row["is_circuit"])
        rows.append(row)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    n_circ = sum(r["is_circuit"] for r in rows)
    n_dorm = sum(any(r.get(f"dormancy_{m}") for m in reps_by_mode) for r in rows)
    print(f"correlate: {len(rows)} sites ({n_circ} true-circuit) | "
          f"{n_dorm} dormancy false-positives -> {out_path}")
    return rows


def _offmanifold(site, rep, cf_dataset, manifold_cvars) -> float:
    """Off-manifold magnitude of h_intervened for one site.

    TODO(pod): replace the norm placeholder with the residual against the fitted
    manifold. The pieces exist: import load_manifold + the projection from
    off-distribution-analyses.py (hyphenated -> importlib), group reps by
    context_key(ex, manifold_cvars), project through that context's manifold, and
    return the mean residual norm. Until then this returns ||rep|| so the table and
    membership join are exercised end-to-end."""
    return float(rep.float().norm(dim=-1).mean())


def _site_to_spec(site):
    """AtomicModelUnit -> (layer, comp, head|None) matching circuit_specs' keys."""
    ct = site.component_type
    if ct in ("attn_result", "head_attention_value_output"):
        return (site.layer, "attn_result", getattr(site, "head", None))
    if ct in ("mlp_output", "mlp_activation"):
        return (site.layer, "mlp_activation", None)
    return (site.layer, "block_output" if "output" in ct else "block_input", None)


# --------------------------------------------------------------------------- #
#  orchestration
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="18")
    ap.add_argument("--component", default="mlp", choices=["residual", "mlp", "attention"])
    ap.add_argument("--patch-at", default="corr", choices=["corr", "write", "internal"])
    ap.add_argument("--modes", nargs="+", default=["STR", "Gaussian"])
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="results_v2/dormancy")
    ap.add_argument("--behaviour", action="store_true", help="also run the behaviour side")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"== 1. prepare data (case {args.case}) ==")
    d = PrepareData(args.case, n=args.n, device=args.device).generate()
    task, pipeline, causal_model = d["task"], d["pipeline"], d["causal_model"]

    print(f"== 2. build {args.component} grid (patch_at={args.patch_at}) ==")
    layers = args.layers if args.layers is not None else list(range(d["n_layers"]))
    grid = build_grid(pipeline, task, args.component, layers,
                      positions=[args.position], patch_at=args.patch_at)
    target_sites = [t.flatten()[0] for t in grid.values()]  # flat list[AtomicModelUnit]
    print(f"   {len(target_sites)} sites")

    print("== 3. fit manifolds (once, mode-independent) ==")
    generate_activation_manifold(
        target_sites, d["cf_dataset"], d["manifold_cvars"], pipeline, causal_model,
        batch_size=args.batch_size, intervention_variable=d["intervention_variable"], task=task,
    )

    print("== 4. intervened reps + (opt) behaviour, per mode ==")
    reps_by_mode = {}
    for mode in args.modes:
        reps_by_mode[mode] = get_intervened_representations(
            target_sites, d["cf_dataset"], pipeline, mode,
            causal_model=causal_model, batch_size=args.batch_size, output_dir=args.out,
        )
        if args.behaviour:
            collect_patched_distributions(
                target_sites, d["cf_dataset"], pipeline, task, mode,
                causal_model=causal_model, batch_size=args.batch_size, output_dir=args.out,
            )

    print("== 5. apparent-circuit scan (IIT interchange score, full grid) ==")
    try:
        patching_scores = compute_patching_scores(
            grid, d["cf_dataset"], pipeline, causal_model, args.modes, args.batch_size,
        )
    except Exception as e:  # IOI shim can't score yet; keep the divergence table
        print(f"   [warn] patching scan skipped ({type(e).__name__}: {e}); "
              f"apparent_/dormancy_ columns will be NaN. Needs an HL-model-based "
              f"metric for this case.")
        patching_scores = {}

    print("== 6. correlate: divergence x apparent x true circuit ==")
    correlate_divergences(
        grid, reps_by_mode, patching_scores, causal_model, d["cf_dataset"],
        d["manifold_cvars"], d["hl_ll_corr"], os.path.join(args.out, "dormancy_table.json"),
    )
    print("DONE.")


if __name__ == "__main__":
    main()
