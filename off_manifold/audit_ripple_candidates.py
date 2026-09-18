"""
audit_ripple_candidates.py — count how many components pass the ripple gate
==========================================================================
Scans existing int.json files (no model loading) and reports, per combo x seed,
how many off-circuit and in-circuit components pass --gate-by orth --tau T.
Run BEFORE run_ripple_tp_sweep.sh to estimate cost.

    python off_manifold/audit_ripple_candidates.py --tau 0.05
    python off_manifold/audit_ripple_candidates.py --tau 0.0   # no gate = all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="poly_results_new/results")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--gate-by", default="orth", choices=["orth", "int", "none"])
    ap.add_argument("--kind", default="swap", choices=["swap", "push", "all"])
    args = ap.parse_args()

    combos = sorted(set(
        m.split("/")[-2] for m in
        glob.glob(os.path.join(args.results_root, "poly_*", "int_seed*"))
        if os.path.isdir(m)
    ))

    print(f"gate_by={args.gate_by}  tau={args.tau}  kind={args.kind}")
    print(f"{'combo':<25} {'seed':>4}  {'off_pass':>8} {'off_total':>9}  "
          f"{'tp_pass':>7} {'tp_total':>8}  {'candidates':>10}  {'fwd_passes':>10}")
    print("-" * 100)

    grand_cand = 0
    grand_fwd = 0

    for combo_dir in sorted(glob.glob(os.path.join(args.results_root, "poly_*"))):
        combo = os.path.basename(combo_dir).replace("poly_", "")
        for seed_dir in sorted(glob.glob(os.path.join(combo_dir, "int_seed*"))):
            seed = os.path.basename(seed_dir).replace("int_seed", "")
            int_dir = os.path.join(seed_dir, "interaction")
            if not os.path.isdir(int_dir):
                continue

            # scan direction: count unique (strategy, push, direction, alpha) regimes
            # and unique sites per regime
            regimes = defaultdict(lambda: {"off_pass": set(), "off_total": set(),
                                           "tp_pass": set(), "tp_total": set()})
            for path in glob.glob(os.path.join(int_dir, "**", "int.json"), recursive=True):
                try:
                    r = json.load(open(path))
                except Exception:
                    continue
                if r.get("degenerate"):
                    continue

                strat = r.get("strategy", "?")
                push = r.get("push_site", "?")
                kind = "swap" if push == "input" else ("self" if push == "self" else "push")
                if kind == "self":
                    continue
                if args.kind != "all" and kind != args.kind:
                    continue

                regime_key = (strat, push, r.get("direction", "?"), r.get("alpha", "?"))
                sid = r.get("site", "?")
                in_circ = r.get("in_base_circuit")

                if in_circ is False:
                    regimes[regime_key]["off_total"].add(sid)
                    if args.gate_by == "orth":
                        ob = r.get("orth_base")
                        if ob is not None and abs(ob) >= args.tau:
                            regimes[regime_key]["off_pass"].add(sid)
                    elif args.gate_by == "int":
                        iv = r.get("INT")
                        if iv is not None and abs(iv) >= args.tau:
                            regimes[regime_key]["off_pass"].add(sid)
                    elif args.gate_by == "none":
                        regimes[regime_key]["off_pass"].add(sid)
                elif in_circ is True:
                    regimes[regime_key]["tp_total"].add(sid)
                    regimes[regime_key]["tp_pass"].add(sid)  # TP always pass (no gate)

            off_pass = sum(len(v["off_pass"]) for v in regimes.values())
            off_total = sum(len(v["off_total"]) for v in regimes.values())
            tp_pass = sum(len(v["tp_pass"]) for v in regimes.values())
            tp_total = sum(len(v["tp_total"]) for v in regimes.values())
            cand = off_pass + tp_pass
            # each candidate = 1 forward_cache (ripple push) + 1 component_xint (N partner patches)
            n_regimes = len(regimes)

            grand_cand += cand
            grand_fwd += cand  # ~1 fwd per candidate (plus xINT partner patches)

            print(f"{combo:<25} {seed:>4}  {off_pass:>8} {off_total:>9}  "
                  f"{tp_pass:>7} {tp_total:>8}  {cand:>10}  ~{cand:>9}")

    print("-" * 100)
    print(f"{'TOTAL':<25} {'':>4}  {'':>8} {'':>9}  "
          f"{'':>7} {'':>8}  {grand_cand:>10}  ~{grand_fwd:>9}")
    print(f"\nEach candidate requires 1 forward_cache + 1 component_xint pass (G partner patches).")
    print(f"Total candidates across all combos x seeds x regimes: {grand_cand}")


if __name__ == "__main__":
    main()
