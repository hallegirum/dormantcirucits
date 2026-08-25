"""
run_gap_int.py  —  post-hoc gap_INT (emergent / bypass-gated interaction), like run_gap
========================================================================================
gap_INT isolates the EMERGENT part of the interaction term. Where gap(c) is c's own
region-crossing (diagonal Hessian), gap_INT is the bypass-CONDITIONED curvature (off-
diagonal): how much c's nonlinearity is switched on by the corrupted bypass — the sharp
signature of "the clean/live context recruits an otherwise-dead mediator" (Type-G,
bypass-gated). Per component:

    INT_linear = <delta_c, grad_clean - grad_corrupt>          (smooth interaction)
      grad_clean   = dPhi/dh_c at (clean bypass,  clean mediator)   -> AtP for NIE
      grad_corrupt = dPhi/dh_c at (corrupt bypass, clean mediator)  -> AtP for PIE
    gap_INT    = INT (saved) - INT_linear                     (emergent interaction)

Reuses the SAVED INT and reps; recomputes two gradient fields (grad_clean once for all
sites; grad_corrupt per component via a differentiable write in the corrupt-bypass forward).
Appends int_linear / gap_INT / gap_INT_frac onto each int.json row. Skips self-push (INT
degenerate) and the push site itself. One (combo, seed) per call. --n MUST match the
original run (fixes the pool), same as run_gap.

    python run_gap_int.py --results-root poly_results_new --combo cases_0+2 --seed 0 \
        --case poly --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from interaction_effects import (make_counterfactual, make_readout, atp_gradients,
                                 atp_grad_corrupt, _int_gap, _gather)


def parse_reps_dir(name):
    m = re.match(r"int-([^_]+)__(.+)-a([0-9.]+)__(.+)$", name)
    if not m:
        return None
    strat, direction, alpha, push = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    parts = direction.split("-")
    return {"strategy": strat, "direction": direction, "alpha": alpha, "push_site": push,
            "base_task": int(parts[0]) if parts[0].isdigit() else None,
            "target_task": (int(parts[1]) if (len(parts) == 2 and parts[1].isdigit()) else None)}


def load_scan_rows(results_dir):
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
    ap.add_argument("--n", type=int, default=200, help="MUST match the original run_interaction_effect --n")
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
        if info is None or info["push_site"] == "self" or info["base_task"] is None:
            continue                                           # self-push: INT degenerate
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        n_scan = int(next(iter(hp.values())).shape[0])

        mk = dict(strategy=strat, base_task=info["base_task"], alpha=alpha)
        if push not in ("input", "self"):
            mk["push_site"] = unit_by_id.get(push)
        if strat == "STR":
            mk["target_task"] = info["target_task"]
        spec = make_counterfactual(d, n=n_scan, seed=args.seed, **mk)
        if len(spec.cf_noise) != n_scan:
            print(f"[gapINT][skip] {strat} {direc} a{alpha}: rebuilt {len(spec.cf_noise)} != saved {n_scan}")
            continue
        readout = make_readout(spec.cf_noise, spec.positions)
        bases = [ex["input"] for ex in spec.cf_noise]
        sources = [ex["counterfactual_inputs"][0] for ex in spec.cf_noise]
        pos = spec.positions

        # clean reps + grad_clean (all sites, one backward)
        _, base_slabs = pipeline.forward_cache(bases, all_units, batch_size=args.batch_size)
        h = {uid: _gather(base_slabs[uid], pos) for uid in base_slabs}
        # alignment net (swap): confirm saved reps match the rebuild
        if push == "input":
            _, src_slabs = pipeline.forward_cache(sources, all_units, batch_size=args.batch_size)
            s0 = next(iter(hp))
            hchk = h[s0] + alpha * (_gather(src_slabs[s0], pos) - h[s0])
            if float((hchk - hp[s0]).abs().max()) > 1e-3:
                print(f"[gapINT][skip] {strat} {direc} a{alpha}: saved reps != rebuild — pass --n matching original")
                continue
        grad_clean = atp_gradients(pipeline, bases, readout, all_units, pos, batch_size=args.batch_size)

        # corrupt-bypass input + push, per kind
        if push == "input":                                    # swap: corrupt bypass = source input
            corrupt_input, push_tuple, skip_site = sources, None, None
        else:                                                  # two-site push: base input + push at X
            X = unit_by_id[push]
            x_base = _gather(pipeline.forward_cache(bases, [X], batch_size=args.batch_size)[1][X.id], pos)
            if strat == "STR":
                x_src = _gather(pipeline.forward_cache(sources, [X], batch_size=args.batch_size)[1][X.id], pos)
                delta = alpha * (x_src - x_base)
            else:                                              # Gaussian
                g = torch.Generator().manual_seed(spec.seed)
                delta = alpha * 3.0 * x_base.std() * torch.randn(x_base.shape, generator=g)
            corrupt_input, push_tuple, skip_site = bases, (X, delta, pos), X.id

        for sid, hp_c in hp.items():
            if sid == skip_site or sid not in unit_by_id or sid not in h:
                continue
            hit = rows_idx.get((strat, push, direc, alpha, sid))
            if hit is None:
                continue
            row, path = hit
            if row.get("degenerate") or row.get("INT") is None:
                continue
            grad_corr = atp_grad_corrupt(pipeline, corrupt_input, readout, unit_by_id[sid],
                                         h[sid], pos, push=push_tuple, batch_size=args.batch_size)
            delta_c = hp_c - h[sid]
            row.update(_int_gap(delta_c, grad_clean.get(sid), grad_corr, row["INT"]))
            json.dump(row, open(path, "w"), indent=2)
            n_written += 1
        print(f"[gapINT] {strat:>8} {direc:>7} a{alpha:<4g} push={push:<32} sites {len(hp)}")

    print(f"\nwrote int_linear/gap_INT/gap_INT_frac onto {n_written} int.json rows")


if __name__ == "__main__":
    main()
