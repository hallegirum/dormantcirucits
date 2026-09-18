"""
diag_softmax_nan.py — why is local_softmax_gap NaN on the 3-layer poly models?
=============================================================================
Loads ONE combo, runs the clean forward with the attn-score hooks, gathers each
head's query row at the per-example target position (exactly as run_layer_scans
does) and reports, per layer/head, the examples whose row is degenerate:
all -inf (softmax 0/0), contains +inf (fp overflow), or contains NaN.

    python off_manifold/diag_softmax_nan.py --combo cases_0+1+2 --seed 0 --device cuda
    python off_manifold/diag_softmax_nan.py --combo cases_0+1   --seed 0 --device cuda   # control
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from interaction_effects import make_counterfactual, _head_query_row  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from run_dormancy_pipeline import PrepareData
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData("poly", n=args.n, seed=args.seed, device=args.device,
                    position="last_token", poly_comb=args.combo).generate()
    pipeline = d["pipeline"]
    model = pipeline.model
    cfg = model.cfg
    print(f"model: n_layers={cfg.n_layers} n_heads={cfg.n_heads} n_ctx={cfg.n_ctx} "
          f"dtype={cfg.dtype} attn_dir={cfg.attention_dir} "
          f"tokenizer={'yes' if getattr(model, 'tokenizer', None) is not None else 'none'} "
          f"pad_token_id={pipeline.pad_token_id}")

    spec = make_counterfactual(d, n=args.n, seed=args.seed, strategy="STR",
                               base_task=0, target_task=0, alpha=1.0)
    bases = [ex["input"] for ex in spec.cf_noise]
    pos = torch.as_tensor(spec.positions, dtype=torch.long)
    tokens = pipeline.load(bases)["tokens"]
    N, S = tokens.shape
    print(f"tokens: N={N} S={S}   pos: min={int(pos.min())} max={int(pos.max())} "
          f"(per-example={pos.dim() > 0})   pos>=S: {int((pos >= S).sum())}")
    raw_lens = [len(ex["raw_input"]) for ex in bases]
    print(f"raw_input len: min={min(raw_lens)} max={max(raw_lens)}")

    hooks = [f"blocks.{ly}.attn.hook_attn_scores" for ly in range(cfg.n_layers)]
    _, slabs = pipeline.forward_cache(bases, [], batch_size=args.batch_size, extra_hooks=hooks)

    worst = None
    for ly, hk in enumerate(hooks):
        if hk not in slabs:
            print(f"[L{ly}] hook {hk} NOT in cache")
            continue
        sc = slabs[hk]                                     # (N, H, Q, K)
        print(f"[L{ly}] scores {tuple(sc.shape)} dtype={sc.dtype}  "
              f"+inf={int(torch.isposinf(sc).sum())}  nan={int(torch.isnan(sc).sum())}")
        for h in range(sc.shape[1]):
            row = _head_query_row(sc, h, pos).float()      # (N, K)
            fin = torch.isfinite(row)
            all_neg_inf = (~fin).all(-1) & torch.isneginf(row).all(-1)
            has_pos_inf = torch.isposinf(row).any(-1)
            has_nan = torch.isnan(row).any(-1)
            p = torch.softmax(row, dim=-1)
            p_nan = torch.isnan(p).any(-1)
            bad = all_neg_inf | has_pos_inf | has_nan
            fin_max = row[fin].abs().max().item() if fin.any() else float("nan")
            print(f"   H{h}: softmax-NaN rows={int(p_nan.sum()):3d}  all(-inf)={int(all_neg_inf.sum()):3d}  "
                  f"+inf={int(has_pos_inf.sum()):3d}  nan={int(has_nan.sum()):3d}  "
                  f"finite keys/row min={int(fin.sum(-1).min())}  max|finite score|={fin_max:.3g}")
            if bad.any() and worst is None:
                worst = (ly, h, int(bad.nonzero()[0]), sc, row)

    if worst is None:
        print("\nno degenerate rows found: NaN must come from the patched side, not the clean scores")
        return
    ly, h, i, sc, row = worst
    print(f"\n=== first degenerate example: L{ly} H{h} example {i}  pos={int(pos[i])} ===")
    print(f"tokens[{i}]     : {tokens[i].tolist()}")
    print(f"raw_input[{i}]  : {list(bases[i]['raw_input'])}")
    print(f"gathered row    : {row[i].tolist()}")
    print(f"full (Q,K) block for this head/example (rows=query, cols=key):")
    torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
    print(sc[i, h])


if __name__ == "__main__":
    main()
