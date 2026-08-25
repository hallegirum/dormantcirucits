"""
run_interpbench_verify.py
=========================
End-to-end driver for one InterpBench case. Exercises the pieces that have no
main() of their own: build_tracr_pipeline, dump_correspondence,
verify_against_model, partition_targets.

Run in the ib-venv (needs circuits_benchmark + your forked causalab):
    source ~/ib-venv/bin/activate
    python dormant-circuit-analysis/interp_bench/run_interpbench_verify.py --case 18

Steps:
  1. load the case's HookedTracrTransformer + hl_ll_corr (circuits_benchmark)
  2. wrap it in a TracrPipeline using the model's OWN encoders (build_tracr_pipeline)
  3. dump_correspondence  — confirm the hl_ll_corr field names
  4. verify_against_model — hl_fn matches the real model? seq_len/bos correct?
  5. partition_targets    — circuit vs non-circuit units (the patch axes)
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_case(case: str, device: str):
    """Load the case's task, correspondence, and HookedTracrTransformer."""
    mod = importlib.import_module(f"circuits_benchmark.benchmark.cases.case_{case}")
    task = getattr(mod, f"Case{case}")()
    from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import (
        get_ll_model_loader,
    )

    loader = get_ll_model_loader(task, interp_bench=True)
    hl_ll_corr, model = loader.load_ll_model_and_correspondence(device=device)
    return task, hl_ll_corr, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="18")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200, help="verify sample count")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from interpbench_loading import build_pipelines, dump_correspondence, partition_targets
    from causalab.tasks.interpbench.causal_models import CASES, verify_against_model
    from causalab.tasks.interpbench.token_positions import create_token_positions

    # if args.case not in CASES:
    #     raise SystemExit(f"Case {args.case} not registered in CASES: {sorted(CASES)}")

    print(f"\n=== 1. load case {args.case} (circuits_benchmark) ===")
    task, hl_ll_corr, model = load_case(args.case, args.device)

    print("\n=== 2. build_pipelines (LL + HL, hl_fn from the HL program) ===")
    ll_pipeline, hl_pipeline, bos, hl_fn, vocab, out_vocab = build_pipelines(
        model, task, device=args.device)
    n_ctx = int(model.cfg.n_ctx)
    print(f"bos={bos!r} | n_ctx={n_ctx} | n_layers={model.cfg.n_layers} n_heads={model.cfg.n_heads}")
    print(f"vocab={vocab} | out_vocab={out_vocab}")

    print("\n=== 3. dump_correspondence (confirm field names) ===")
    dump_correspondence(hl_ll_corr)

    # reconcile the registered config with THIS model (seq_len from n_ctx, real bos),
    # and inject the model-derived ground-truth hl_fn.
    seq_len = n_ctx - (1 if bos is not None else 0)
    cfg = dataclasses.replace(CASES[args.case], bos=bos, seq_len=seq_len, hl_fn=hl_fn)
    print(f"\n(using seq_len={seq_len}, bos={bos!r}, hl_fn=HL-program for case {args.case})")

    if not args.skip_verify:
        print("\n=== 4. verify_against_model (LL model vs HL program) ===")
        verify_against_model(cfg, ll_pipeline, n=args.n)

    print("\n=== 5. partition_targets (circuit vs non-circuit) ===")
    token_positions = create_token_positions(ll_pipeline)
    circuit, noncircuit = partition_targets(hl_ll_corr, model, token_positions["last_token"])
    print(f"circuit units: {len(circuit)} | non-circuit units: {len(noncircuit)}")
    for u in circuit[:12]:
        print("   circuit:", u.id)

    print("\nDONE.")


if __name__ == "__main__":
    main()
