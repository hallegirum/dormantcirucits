"""
MINIMAL feasibility spike — loads an InterpBench model DIRECTLY from its HF
checkpoint, with NO circuits_benchmark / tracr / iit / jax / haiku / flax.

Those packages exist only to *compile* the RASP programs; the trained model is
just a TransformerLens HookedTransformer (ll_model_cfg.pkl + ll_model.pth), and
the ground-truth circuit is a plain pickle (edges.pkl). So the make-or-break
question — can pyvene hook the model? — needs only:

    transformer_lens  huggingface_hub  torch  pyvene   (all already in ib-venv)

Run (in the ib-venv):
    python spike_minimal.py --case 8 --layer 0
"""

import argparse
import pickle

import torch
from huggingface_hub import hf_hub_download

REPO = "cybershiptrooper/InterpBench"


def section(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def load_model(case, device):
    section(f"A. LOAD case {case} from HF checkpoint (no tracr/jax/haiku)")
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg_path = hf_hub_download(REPO, subfolder=str(case), filename="ll_model_cfg.pkl")
    w_path = hf_hub_download(REPO, subfolder=str(case), filename="ll_model.pth")
    with open(cfg_path, "rb") as fh:
        raw = pickle.load(fh)
    cfg = raw if isinstance(raw, HookedTransformerConfig) else HookedTransformerConfig(**raw)
    model = HookedTransformer(cfg).to(device)
    state = torch.load(w_path, map_location=device)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"HookedTransformer: n_layers={cfg.n_layers} d_model={cfg.d_model} "
          f"n_ctx={cfg.n_ctx} d_vocab={cfg.d_vocab}")
    if missing or unexpected:
        print(f"  (partial load: {len(missing)} missing / {len(unexpected)} unexpected keys)")
    return model


def load_ground_truth(case):
    section("GROUND TRUTH — edges.pkl (the circuit; your dormancy labels)")
    try:
        edges = pickle.load(open(hf_hub_download(REPO, subfolder=str(case), filename="edges.pkl"), "rb"))
        print(f"edges.pkl: {type(edges).__name__}, len={len(edges) if hasattr(edges,'__len__') else '?'}")
        for e in (list(edges)[:5] if hasattr(edges, "__iter__") else []):
            print("   ", e)
    except Exception as e:
        print("edges.pkl load failed:", type(e).__name__, e)


def read_activations(model, layer, device):
    section(f"B. FORWARD + READ resid at layer {layer}")
    toks = torch.randint(0, model.cfg.d_vocab, (1, min(model.cfg.n_ctx, 8)), device=device)
    hook = f"blocks.{layer}.hook_resid_post"
    logits, cache = model.run_with_cache(toks)
    print(f"tokens {tuple(toks.shape)} -> logits {tuple(logits.shape)}")
    print(f"{hook}: {tuple(cache[hook].shape)}   <- (batch, pos, d_model), what off-manifold consumes")
    return hook


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
    base = torch.randint(0, model.cfg.d_vocab, (1, 6), device=device)
    src = torch.randint(0, model.cfg.d_vocab, (1, 6), device=device)
    _, src_cache = model.run_with_cache(src)
    src_act, pos = src_cache[hook], 3
    clean = model(base)
    # TL hooks are called as fn(value, hook=<HookPoint>) — the 2nd arg is the
    # keyword `hook`, so accept **kw (naming it `h` positionally is why it errored).
    def patch(v, **kw):
        return v.index_copy(1, torch.tensor([pos], device=device), src_act[:, pos:pos+1, :])
    patched = model.run_with_hooks(base, fwd_hooks=[(hook, patch)])
    changed = not torch.allclose(clean, patched)
    print(f"clean vs patched logits differ: {changed}  (True = swap took effect)")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="8")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model = load_model(args.case, args.device)
    load_ground_truth(args.case)
    hook = read_activations(model, args.layer, args.device)
    pyvene_ok = test_pyvene(model, args.layer)
    tl_ok = test_tl_hooks(model, hook, args.device)

    section("VERDICT")
    print(f"  load + read activations : YES")
    print(f"  pyvene intervention     : {'YES' if pyvene_ok else 'NO'}")
    print(f"  TL-hook intervention    : {'YES' if tl_ok else 'NO'}")
    if pyvene_ok:
        print("\n=> EASY PATH: causalab's intervention code works unchanged.")
    elif tl_ok:
        print("\n=> FALLBACK PATH: write TracrPipeline.intervenable_generate on run_with_hooks.")
    else:
        print("\n=> BLOCKED: paste this output.")


if __name__ == "__main__":
    main()
