"""
FEASIBILITY SPIKE — can causalab's off-manifold pipeline run on an InterpBench model?

Uses InterpBench's own `circuits_benchmark` API (per DEMO_InterpBench.ipynb) rather
than hand-loading checkpoints. Answers the make-or-break question:

    (A) load the case's HookedTransformer + its clean/corrupted data + the ground-
        truth correspondence, and read residual-stream activations, and
    (B) INTERVENE on it — via pyvene (causalab works unchanged) or, failing that,
        via TransformerLens native hooks (a bounded adapter is needed).

Deps (installs transformer_lens, iit, etc. transitively):
    uv pip install "git+https://github.com/FlyingPumba/InterpBench.git"
Run:
    python spike_load_interpbench.py --case 8 --layer 0
Read the VERDICT block at the end.
"""

import argparse
import importlib
import pickle

import torch


def section(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


# --------------------------------------------------------------------------- #
# A. load case, model, correspondence, data — the demo's API
# --------------------------------------------------------------------------- #
def load_case(case, device):
    section(f"A. LOAD case {case} via circuits_benchmark")
    # import circuits_benchmark.benchmark.cases.case_8 as case_8; task = case_8.Case8()
    mod = importlib.import_module(f"circuits_benchmark.benchmark.cases.case_{case}")
    task = getattr(mod, f"Case{case}")()
    task_idx = task.get_name()
    print("task:", task_idx)

    from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader
    loader = get_ll_model_loader(task, interp_bench=True)
    hl_ll_corr, model = loader.load_ll_model_and_correspondence(device=device)
    print(f"HookedTransformer: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model} "
          f"n_ctx={model.cfg.n_ctx} d_vocab={model.cfg.d_vocab}")
    return task, task_idx, model, hl_ll_corr


# --------------------------------------------------------------------------- #
# GROUND TRUTH — the correspondence (hl node -> ll components) + edges.pkl
# --------------------------------------------------------------------------- #
def inspect_ground_truth(task_idx, hl_ll_corr):
    section("GROUND TRUTH — correspondence + circuit edges (your dormancy labels)")
    try:
        items = list(hl_ll_corr.items()) if hasattr(hl_ll_corr, "items") else list(hl_ll_corr)
        print(f"correspondence entries: {len(items)}")
        for k, v in items[:6]:
            print("   ", k, "->", v)
    except Exception as e:
        print("could not enumerate correspondence:", type(e).__name__, e, "| repr:", repr(hl_ll_corr)[:200])
    try:
        from huggingface_hub import hf_hub_download
        edges_file = hf_hub_download("cybershiptrooper/InterpBench", subfolder=task_idx, filename="edges.pkl")
        edges = pickle.load(open(edges_file, "rb"))
        print(f"edges.pkl loaded: {type(edges).__name__}, len={len(edges) if hasattr(edges,'__len__') else '?'}")
    except Exception as e:
        print("edges.pkl load failed:", type(e).__name__, e)


# --------------------------------------------------------------------------- #
# B. read activations on a REAL clean input
# --------------------------------------------------------------------------- #
def read_activations(task, model, layer, device):
    section(f"B. FORWARD on a real clean input + READ resid at layer {layer}")
    clean = task.get_clean_data(max_samples=4)
    inputs = clean.get_inputs()
    print("clean inputs type:", type(inputs).__name__, "| first:", repr(inputs[0])[:120])
    toks = inputs[:1] if torch.is_tensor(inputs) else torch.as_tensor(inputs[:1])
    toks = toks.to(device)
    hook = f"blocks.{layer}.hook_resid_post"
    logits, cache = model.run_with_cache(toks)
    act = cache[hook]
    print(f"tokens {tuple(toks.shape)} -> logits {tuple(logits.shape)}")
    print(f"{hook}: {tuple(act.shape)}   <- (batch, pos, d_model) is what off-manifold consumes")
    return toks, hook


# --------------------------------------------------------------------------- #
# C1. pyvene (best case: causalab unchanged)  /  C2. TL hooks (fallback)
# --------------------------------------------------------------------------- #
def test_pyvene(model, layer):
    section("C1. pyvene intervention (best case: causalab works UNCHANGED)")
    try:
        import pyvene as pv
        cfg = pv.IntervenableConfig([{
            "component": f"blocks[{layer}].hook_resid_post",
            "intervention_type": pv.VanillaIntervention,
        }])
        iv = pv.IntervenableModel(cfg, model)
        base = torch.randint(0, model.cfg.d_vocab, (1, 6))
        src = torch.randint(0, model.cfg.d_vocab, (1, 6))
        iv(base, [src], {"sources->base": ([[[3]]], [[[3]]])})
        print("PYVENE OK — wrapped the HookedTransformer and ran an interchange.")
        return True
    except Exception as e:
        print(f"PYVENE FAILED: {type(e).__name__}: {e}")
        return False


def test_tl_hooks(model, hook, device):
    section("C2. TransformerLens native-hook intervention (fallback adapter proof)")
    L = 6
    base = torch.randint(0, model.cfg.d_vocab, (1, L), device=device)
    src = torch.randint(0, model.cfg.d_vocab, (1, L), device=device)
    _, src_cache = model.run_with_cache(src)
    src_act, pos = src_cache[hook], 3
    clean = model(base)
    patched = model.run_with_hooks(
        base, fwd_hooks=[(hook, lambda v, h: v.index_copy(1, torch.tensor([pos], device=device), src_act[:, pos:pos+1, :]))])
    changed = not torch.allclose(clean, patched)
    print(f"clean vs patched logits differ: {changed}  (True = swap took effect)")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="8")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    task, task_idx, model, corr = load_case(args.case, args.device)
    inspect_ground_truth(task_idx, corr)
    _, hook = read_activations(task, model, args.layer, args.device)
    pyvene_ok = test_pyvene(model, args.layer)
    tl_ok = test_tl_hooks(model, hook, args.device)

    section("VERDICT")
    print(f"  load + read activations : YES")
    print(f"  ground-truth available  : correspondence + edges.pkl")
    print(f"  pyvene intervention     : {'YES' if pyvene_ok else 'NO'}")
    print(f"  TL-hook intervention    : {'YES' if tl_ok else 'NO'}")
    if pyvene_ok:
        print("\n=> EASY PATH: causalab's intervention code works unchanged. Build only the")
        print("   CausalModel (labels from task.get_clean_data targets) + a thin TracrPipeline.")
    elif tl_ok:
        print("\n=> FALLBACK PATH: pyvene won't hook TL, but native hooks work. Write")
        print("   TracrPipeline.intervenable_generate on model.run_with_hooks. Bounded work.")
    else:
        print("\n=> BLOCKED: neither intervention path worked — paste this output before investing.")


if __name__ == "__main__":
    main()
