"""
verify_poly_hooks.py — pod verification for local_gap_frac prerequisites
=========================================================================
Run on the pod BEFORE launching run_interaction_effect.py --local-gap to confirm
that poly_bench models have the required hooks (mlp.hook_pre, attn.hook_attn_scores)
and to check model.cfg.act_fn (relu vs gelu).

    python verify_poly_hooks.py --case poly --poly-comb cases_0+1 --seed 0 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="poly")
    ap.add_argument("--poly-comb", default="cases_0+1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    args = ap.parse_args()

    from run_dormancy_pipeline import PrepareData
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.poly_comb).generate()
    model = d["pipeline"].model

    print("=" * 60)
    print(f"Model type: {type(model).__name__}")
    print(f"n_layers:   {model.cfg.n_layers}")
    print(f"n_heads:    {model.cfg.n_heads}")
    print(f"act_fn:     {model.cfg.act_fn}")
    print(f"d_model:    {model.cfg.d_model}")
    print(f"d_mlp:      {model.cfg.d_mlp}")
    print("=" * 60)

    # run a single forward with cache to see all available hooks
    import torch
    tokens = d["pipeline"].load(d["pool"][d["tasks"][0]][:1])["tokens"]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    hook_names = sorted(cache.keys())
    print(f"\nTotal cached hooks: {len(hook_names)}")

    # check required hooks per layer
    required_mlp = "blocks.{layer}.mlp.hook_pre"
    required_attn = "blocks.{layer}.attn.hook_attn_scores"
    ok = True

    for layer in range(model.cfg.n_layers):
        mlp_pre = required_mlp.format(layer=layer)
        attn_scores = required_attn.format(layer=layer)
        mlp_ok = mlp_pre in cache
        attn_ok = attn_scores in cache

        if mlp_ok:
            shape_mlp = tuple(cache[mlp_pre].shape)
            print(f"  [OK]   {mlp_pre}  shape={shape_mlp}")
        else:
            print(f"  [MISS] {mlp_pre}")
            ok = False

        if attn_ok:
            shape_attn = tuple(cache[attn_scores].shape)
            print(f"  [OK]   {attn_scores}  shape={shape_attn}")
        else:
            print(f"  [MISS] {attn_scores}")
            ok = False

    # also check post-activation hooks for reference
    post_hooks = [f"blocks.{ly}.mlp.hook_post" for ly in range(model.cfg.n_layers)]
    for h in post_hooks:
        if h not in cache:
            print(f"  [MISS] {h} (post-activation, required by existing pipeline)")
            ok = False

    print("\n" + "=" * 60)
    if ok:
        print("PASS: all required hooks present. "
              f"act_fn={model.cfg.act_fn} -> use local_{model.cfg.act_fn}_gap")
    else:
        print("FAIL: some hooks missing. Check model configuration.")
        missing = []
        for layer in range(model.cfg.n_layers):
            if required_mlp.format(layer=layer) not in cache:
                missing.append(required_mlp.format(layer=layer))
            if required_attn.format(layer=layer) not in cache:
                missing.append(required_attn.format(layer=layer))
        print(f"Missing: {missing}")
    print("=" * 60)

    # print all hook names for reference
    print("\nFull hook list:")
    for h in hook_names:
        print(f"  {h}  shape={tuple(cache[h].shape)}")


if __name__ == "__main__":
    main()
