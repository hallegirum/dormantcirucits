"""
run_xint.py  —  cross-interaction (xINT) partner maps + healthy/borrowed/ectopic verdict
=========================================================================================
Post-hoc over a finished interaction run. For each SCAN (strategy x direction x alpha) it
selects the interaction-carried CANDIDATES (|INT| >= tau, non-degenerate) and, for each,
computes xINT(c, N) against the WHOLE grid (Curse of Multiple Mediators sec.4):

    xINT(c,N) = NIE(c,N) - NIE(c) - NIE(N)      NIE(c,N) = joint noising patch of c AND N

then buckets |xINT| by the partner's GROUND-TRUTH task membership (home_tasks) into
on-task / off-task / shared / none, and reads off a verdict:

    off-task mass dominant & concentrated   -> BORROWED   (recruited another task's circuit)
    on-task mass dominant                    -> HEALTHY    (relational within its own task)
    diffuse / no coherent bucket             -> ECTOPIC    (interaction with noise)

Reuses the SAVED intervened reps (h'_c, gathered) and the SAVED int.json (NIE/INT per
site, home_tasks) — only the JOINT term is recomputed, via pipeline.patch's multi-write.
Marginals NIE(c), NIE(N) are read from int.json. One (combo, seed) per call (rebuilds the
model). Skips self-push (degenerate). Portability note: home_tasks is the ground-truth
partition used to VALIDATE the signal here; the transferable version replaces it with an
on-inputs activity map (silent-on-distribution) — build that after this validates.

    python run_xint.py --results-root poly_results_new --combo cases_0+2 --seed 0 \
        --case poly --tau 0.05 --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from interaction_effects import make_counterfactual, component_xint, _units_of, _gather


# --------------------------------------------------------------------------- #
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
    """Every int.json under {results_dir}/interaction/, keyed by
    (strategy, push_site, direction, alpha, site) -> (row, file_path) so we can WRITE the
    xINT summary back onto the row (int.json IS the row that collect_interaction reads)."""
    idx = {}
    for path in glob.glob(os.path.join(results_dir, "interaction", "**", "int.json"), recursive=True):
        try:
            r = json.load(open(path))
        except Exception:
            continue
        if "strategy" not in r:
            continue
        idx[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = (r, path)
    return idx


def bucket(xint_map, unit_by_id, home_by_id, poly_mem, base_task):
    """Split |xINT| mass by partner ground-truth membership relative to base_task."""
    b = {"on_task": 0.0, "off_task": 0.0, "shared": 0.0, "none": 0.0}
    contrib = []
    for pid, x in xint_map.items():
        ax = abs(x)
        home = home_by_id.get(pid, [])
        is_shared = (poly_mem.get(pid) == "shared") or (len(home) > 1)
        if is_shared:
            b["shared"] += ax; tag = "shared"
        elif not home:
            b["none"] += ax; tag = "none"
        elif base_task in home:
            b["on_task"] += ax; tag = "on_task"
        else:
            b["off_task"] += ax; tag = "off_task"
        contrib.append((pid, x, tag))
    tot_task = b["on_task"] + b["off_task"]
    off_frac = b["off_task"] / (tot_task + 1e-9)
    # coherence = share of total |xINT| held by the top-3 partners (concentrated vs diffuse)
    mags = sorted((abs(x) for _p, x, _t in contrib), reverse=True)
    coherence = (sum(mags[:3]) / (sum(mags) + 1e-9)) if mags else 0.0
    if tot_task < 1e-6:
        verdict = "ectopic"
    elif off_frac >= 0.6 and coherence >= 0.5:
        verdict = "borrowed"
    elif off_frac <= 0.4:
        verdict = "healthy"
    else:
        verdict = "mixed"
    return b, off_frac, coherence, verdict, contrib


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", default="poly")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.05, help="|INT| gate for candidates")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--patch-at", default="internal")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default="xint_analysis")
    ap.add_argument("--no-writeback", action="store_true",
                    help="don't append xint_* fields onto the int.json rows (default: writeback)")
    args = ap.parse_args()

    hits = glob.glob(os.path.join(args.results_root, "**", f"poly_{args.combo}",
                                  f"int_seed{args.seed}", "intervened_reps"), recursive=True)
    if not hits:
        raise SystemExit(f"no intervened_reps for poly_{args.combo}/int_seed{args.seed}")
    reps_root = hits[0]

    from run_dormancy_pipeline import PrepareData
    from path_finder import build_full_grid
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.combo).generate()
    pipeline = d["pipeline"]
    grid = build_full_grid(pipeline, d["task"], list(range(d["n_layers"])),
                           positions=[args.position], patch_at=args.patch_at, include_residual=False)
    unit_lists = [_units_of(t) for t in grid.values()]          # partners = every grid site
    unit_by_id = {ul[0].id: ul for ul in unit_lists}
    from run_interaction_effect import site_labels
    _gt, mem, home = site_labels(grid, d)

    results_dir = os.path.dirname(reps_root)                     # {..}/int_seed{S}
    rows_idx = load_scan_rows(results_dir)                        # (row, int.json path) by key
    os.makedirs(args.out, exist_ok=True)
    summary = []

    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None or info["push_site"] == "self" or info["base_task"] is None:
            continue
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])                                    # {site_id: (N,D)} corrupted reps
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        n_scan = int(next(iter(hp.values())).shape[0])           # EXACT example count of this scan

        # marginals NIE + INT per site from the saved rows for THIS scan
        nie_by_site, cand_ids = {}, []
        for sid in hp:
            hit = rows_idx.get((strat, push, direc, alpha, sid))
            if hit is None:
                continue
            r, _p = hit
            nie_by_site[sid] = r.get("NIE", 0.0)
            if (not r.get("degenerate")) and r.get("INT") is not None and abs(r["INT"]) >= args.tau:
                cand_ids.append(sid)
        if not cand_ids:
            continue

        # rebuild bases + readout (same n/seed -> same examples as the saved reps)
        mk = dict(strategy=strat, base_task=info["base_task"], alpha=alpha)
        if push not in ("input", "self"):
            mk["push_site"] = unit_by_id[push][0] if push in unit_by_id else None
        if strat == "STR":
            mk["target_task"] = info["target_task"]
        spec = make_counterfactual(d, n=n_scan, seed=args.seed, **mk)
        if len(spec.cf_noise) != n_scan:                         # guard: pairs didn't reproduce
            print(f"[xint][skip] {strat} {direc} a{alpha}: rebuilt {len(spec.cf_noise)} != saved {n_scan}")
            continue
        from interaction_effects import make_readout
        readout = make_readout(spec.cf_noise, spec.positions)
        bases = [ex["input"] for ex in spec.cf_noise]
        base_scores, _ = pipeline.forward_cache(bases, [], batch_size=args.batch_size)
        import torch
        Ya = readout(base_scores)
        pos = spec.positions
        partners = [unit_by_id[sid] for sid in hp if sid in unit_by_id]

        # alignment safety net (swap): a wrong --n rebuilds different pairs than the saved reps
        if strat == "STR" and push == "input" and partners:
            s0u = partners[0]
            sid0 = s0u[0].id
            _, bsl = pipeline.forward_cache(bases, s0u, batch_size=args.batch_size)
            srcs = [ex["counterfactual_inputs"][0] for ex in spec.cf_noise]
            _, ssl = pipeline.forward_cache(srcs, s0u, batch_size=args.batch_size)
            h0 = _gather(bsl[sid0], pos)
            hchk = h0 + alpha * (_gather(ssl[sid0], pos) - h0)
            if sid0 in hp and float((hchk - hp[sid0]).abs().max()) > 1e-3:
                print(f"[xint][skip] {strat} {direc} a{alpha}: saved reps != rebuild — pass --n "
                      f"matching the ORIGINAL run (reps N={n_scan})")
                continue

        for cid in cand_ids:
            if cid not in unit_by_id:
                continue
            xmap = component_xint(pipeline, bases, readout, Ya, unit_by_id[cid], partners,
                                  hp, nie_by_site, pos, batch_size=args.batch_size)
            b, off_frac, coh, verdict, contrib = bucket(xmap, unit_by_id, home, mem, info["base_task"])
            cand_row, cand_path = rows_idx[(strat, push, direc, alpha, cid)]
            # xINT-weighted OFF-MANIFOLDNESS of c's partners: is c's interaction carried by
            # partners whose OWN activations are off-manifold (dormant recruitment) vs on-
            # manifold live components (backup/relational)? Look up each partner's orth/gap
            # from its row in THIS scan and weight by |xINT(c,N)|.
            tot_x = sum(abs(x) for _p, x, _t in contrib) + 1e-9

            def _pweight(field):
                s = 0.0
                for pid, x, _t in contrib:
                    hit = rows_idx.get((strat, push, direc, alpha, pid))
                    v = hit[0].get(field) if hit else None
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        s += abs(x) * abs(v)
                return s / tot_x
            xint_fields = {"xint_off_frac": round(off_frac, 4), "xint_coherence": round(coh, 4),
                           "xint_on_task": round(b["on_task"], 4), "xint_off_task": round(b["off_task"], 4),
                           "xint_shared": round(b["shared"], 4), "xint_none": round(b["none"], 4),
                           "xint_partner_orth": round(_pweight("orth_base"), 4),
                           "xint_partner_gap": round(_pweight("gap_frac"), 4),
                           "xint_verdict": verdict}
            rec = {"site": cid, "strategy": strat, "push_site": push, "direction": direc,
                   "alpha": alpha, "base_task": info["base_task"], "INT": cand_row.get("INT"),
                   "in_base_circuit": cand_row.get("in_base_circuit"), **xint_fields}
            summary.append(rec)
            if not args.no_writeback:                            # APPEND xINT onto the row (int.json)
                cand_row.update(xint_fields)
                json.dump(cand_row, open(cand_path, "w"), indent=2)
            # full partner map to a sidecar
            safe = lambda s: str(s).replace(os.sep, "-")
            xd = os.path.join(args.out, "maps", safe(cid), strat, safe(push), safe(direc), f"a{alpha:g}")
            os.makedirs(xd, exist_ok=True)
            json.dump({"summary": rec,
                       "xint": {p: round(x, 5) for p, x, _t in contrib},
                       "bucket": {p: t for p, _x, t in contrib}},
                      open(os.path.join(xd, "xint.json"), "w"), indent=2)
        print(f"[xint] {strat:>8} {direc:>7} a{alpha:<4g} push={push:<32} "
              f"candidates {len(cand_ids)}")

    import csv
    tag = f"{args.combo}_seed{args.seed}"
    with open(os.path.join(args.out, f"xint_summary_{tag}.csv"), "w", newline="") as f:
        cols = ["site", "strategy", "push_site", "direction", "alpha", "base_task",
                "in_base_circuit", "INT", "xint_verdict", "xint_off_frac", "xint_coherence",
                "xint_on_task", "xint_off_task", "xint_shared", "xint_none"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)

    # verdict breakdown by strategy x circuit
    agg = defaultdict(lambda: defaultdict(int))
    for r in summary:
        circ = ("in" if r["in_base_circuit"] is True else "off" if r["in_base_circuit"] is False else "?")
        kind = "push" if r["push_site"] not in ("input", "self") else "swap"
        agg[f"{r['strategy']}_{kind} | {circ}"][r["xint_verdict"]] += 1
    print("\n== xINT verdict by strategy x circuit ==")
    for name, vc in sorted(agg.items()):
        tot = sum(vc.values())
        print(f"  {name:<28} n={tot:<4} " + "  ".join(f"{k}:{v}" for k, v in sorted(vc.items())))
    print(f"\nwrote xint_summary_{tag}.csv + per-candidate maps under {args.out}/")


if __name__ == "__main__":
    main()
