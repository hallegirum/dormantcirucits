"""
run_gap.py  —  post-hoc AtP gap (Type-G / region-crossing) without re-running the scan
======================================================================================
`run_layer_scans` computes gap at scan time; this recovers it for runs that predate that,
WITHOUT redoing any intervention patches. Per scan it does exactly one extra forward+
backward (the AtP gradient) and reuses the SAVED intervened reps (h'_c) + SAVED NIE:

    delta_c = h'_c (saved) - h_c (clean, one forward_cache)
    gap(c)  = |NIE (saved)| - |<delta_c, dPhi/dh_c>|           (AtP* Eq.4 saturation term)

and APPENDS atp_linear / gap / gap_frac onto each int.json row (which IS the row that
collect_interaction reads), so the fields flow straight into the regressions. Same output
as the integrated path; use this to backfill existing results. One (combo, seed) per call.

    python run_gap.py --results-root poly_results_new --combo cases_0+2 --seed 0 \
        --case poly --graddrop 0 --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from interaction_effects import (make_counterfactual, make_readout, atp_gradients, _atp_gap,
                                  _units_of, _gather)


def parse_reps_dir(name: str):
    m = re.match(r"int-([^_]+)__(.+)-a([0-9.]+)__(.+)$", name)
    if not m:
        return None
    strat, direction, alpha, push = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    parts = direction.split("-")
    base = parts[0]
    target = parts[1] if len(parts) == 2 else None
    return {"strategy": strat, "direction": direction, "alpha": alpha, "push_site": push,
            "base_task": int(base) if base.isdigit() else None,
            "target_task": (int(target) if (target and target.isdigit()) else None)}


def load_scan_rows(results_dir):
    """int.json under {results_dir}/interaction/ keyed by (strategy,push,direction,alpha,site)
    -> (row, path), so gap can be written back onto the row collect_interaction reads."""
    idx = {}
    for path in glob.glob(os.path.join(results_dir, "interaction", "**", "int.json"), recursive=True):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if "strategy" in r:
            idx[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = (r, path)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", default="poly")
    ap.add_argument("--n", type=int, default=200,
                    help="MUST match the original run_interaction_effect --n: it fixes the trace "
                         "pool, so a wrong value rebuilds different pairs than the saved reps")
    ap.add_argument("--graddrop", type=int, default=0, help="AtP* GradDrop passes (0 = raw AtP)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--patch-at", default="internal")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    hits = glob.glob(os.path.join(args.results_root, "**", f"poly_{args.combo}",
                                  f"int_seed{args.seed}", "intervened_reps"), recursive=True)
    if not hits:
        raise SystemExit(f"no intervened_reps for poly_{args.combo}/int_seed{args.seed}")
    reps_root = hits[0]
    results_dir = os.path.dirname(reps_root)

    from run_dormancy_pipeline import PrepareData
    from path_finder import build_full_grid
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.combo).generate()
    pipeline = d["pipeline"]
    grid = build_full_grid(pipeline, d["task"], list(range(d["n_layers"])),
                           positions=[args.position], patch_at=args.patch_at, include_residual=False)
    all_units = [t.flatten()[0] for t in grid.values()]
    unit_by_id = {u.id: u for u in all_units}
    rows_idx = load_scan_rows(results_dir)

    n_written = 0
    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None or info["base_task"] is None:
            continue
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])                                    # {site_id: (N,D)} corrupted reps
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        n_scan = int(next(iter(hp.values())).shape[0])           # EXACT example count of this scan

        # rebuild bases + readout with the scan's own N -> same examples/order as saved reps
        mk = dict(strategy=strat, base_task=info["base_task"], alpha=alpha)
        if push not in ("input", "self"):
            mk["push_site"] = unit_by_id.get(push)
        if strat == "STR":
            mk["target_task"] = info["target_task"]
        spec = make_counterfactual(d, n=n_scan, seed=args.seed, **mk)
        if len(spec.cf_noise) != n_scan:                         # guard: pairs didn't reproduce
            print(f"[gap][skip] {strat} {direc} a{alpha}: rebuilt {len(spec.cf_noise)} != saved {n_scan}")
            continue
        readout = make_readout(spec.cf_noise, spec.positions)
        bases = [ex["input"] for ex in spec.cf_noise]
        pos = spec.positions

        # clean reps h_c + Ya (one forward_cache) + AtP gradients (one forward+backward)
        base_scores, base_slabs = pipeline.forward_cache(bases, all_units, batch_size=args.batch_size)
        h = {uid: _gather(base_slabs[uid], pos) for uid in base_slabs}
        Ya = readout(base_scores)                                # per-example clean readout

        # alignment safety net (swap): recompute h'_c = h + alpha*(h_src - h) and confirm it
        # matches the SAVED reps — a wrong --n rebuilds different pairs and this catches it.
        if strat == "STR" and push == "input":
            srcs = [ex["counterfactual_inputs"][0] for ex in spec.cf_noise]
            _, src_slabs = pipeline.forward_cache(srcs, all_units, batch_size=args.batch_size)
            s0 = next(iter(hp))
            hchk = h[s0] + alpha * (_gather(src_slabs[s0], pos) - h[s0])
            if float((hchk - hp[s0]).abs().max()) > 1e-3:
                print(f"[gap][skip] {strat} {direc} a{alpha}: saved reps != rebuild — pass --n "
                      f"matching the ORIGINAL run (reps N={n_scan})")
                continue

        grad = atp_gradients(pipeline, bases, readout, all_units, pos,
                             batch_size=args.batch_size, graddrop_k=args.graddrop)

        for sid, hp_c in hp.items():
            hit = rows_idx.get((strat, push, direc, alpha, sid))
            if hit is None or sid not in h:
                continue
            row, path = hit
            unit = unit_by_id.get(sid)
            if unit is None:
                continue
            delta_c = hp_c - h[sid]                              # divergence at the site
            # CLEAN abs-inside full effect, recomputed self-contained on the rebuilt examples:
            # one noising patch (c <- h'_c) -> Phi_n per example -> mean|Phi_n - Ya|. No reliance
            # on saved NIE, so full-effect and AtP are on the SAME examples (E1/E2 mismatch moot).
            phi_n = readout(pipeline.patch(bases, [(unit, hp_c, pos)], push=None,
                                           batch_size=args.batch_size))
            full_abs = float((phi_n - Ya).abs().mean())
            row["NIE_absmean"] = full_abs
            row.update(_atp_gap(delta_c, grad.get(sid), full_abs))
            json.dump(row, open(path, "w"), indent=2)
            n_written += 1
        print(f"[gap] {strat:>8} {direc:>7} a{alpha:<4g} push={push:<32} sites {len(hp)}")

    print(f"\nwrote atp_linear/gap/gap_frac onto {n_written} int.json rows "
          f"({'GradDrop x'+str(args.graddrop) if args.graddrop else 'raw AtP'})")


if __name__ == "__main__":
    main()
