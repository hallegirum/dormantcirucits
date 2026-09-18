"""
run_ripple.py  —  off-manifold RIPPLE that c's single patch induces on the live circuit
========================================================================================
Your dormancy theory, operationalised: patching a component c (noising: clean bypass,
corrupted c) sends its divergence through the residual stream and pushes the CLEAN IN-
CIRCUIT components off THEIR manifold — and that off-manifold displacement of the live
circuit is what breaks the output and fakes c's importance. So for each candidate c, in
c's OWN single patch, we measure how far every other component is pushed off-manifold:

    push c by delta_c (= h'_c - h_c) -> forward, cache all sites
    delta_N|c   = N's activation under c's patch  -  N's clean activation
    orth_N|c    = off-(base-task)-manifold part of delta_N|c   (manifold_decompose)
    ripple_in   = mean orth_N|c over IN-circuit N   (c knocks the LIVE circuit off-manifold)
    ripple_off  = mean orth_N|c over OFF-circuit N
    ripple_frac = ripple_in / (ripple_in + ripple_off)         (is the ripple ON the circuit?)

One cached forward per candidate. Appends ripple_in_orth / ripple_off_orth / ripple_frac
onto each candidate's int.json so collect_interaction picks them up. --n MUST match the
original run. One (combo, seed) per call.

Works for BOTH swap (--kind swap, default) and two-site PUSH (--kind push) — STR or
Gaussian at the shared MLP. The ripple forward is identical: push ONLY c by delta_c on a
CLEAN bypass (the MLP is not re-pushed); only the source of hp[c] differs (c's swapped rep
vs c's rep downstream of the MLP push). Use --strategy Gaussian to run only Gaussian-push.

    # swap (default):
    python run_ripple.py --results-root poly_results_gapInt --combo cases_0+2 --seed 0 \
        --case poly --tau 0.05 --n 150 --device cuda
    # Gaussian-push at the MLP (INT is strong here -> tests interaction contagion):
    python run_ripple.py --results-root poly_results_gapInt --combo cases_0+2 --seed 0 \
        --case poly --kind push --strategy Gaussian --tau 0.5 --n 150 --device cuda
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

from interaction_effects import (make_counterfactual, make_readout, manifold_decompose,
                                 component_xint, _gather, _units_of)


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
    ap.add_argument("--n", type=int, default=200, help="MUST match the original run's --n")
    ap.add_argument("--gate-by", default="orth", choices=["orth", "int", "none"],
                    help="candidate gate: 'orth' = off-manifold divergence (the harmless/pernicious "
                         "population, INT-agnostic); 'int' = |INT| (biases to interaction-carried, "
                         "NOT recommended); 'none' = all off-circuit (costly: xINT weighting is O(G) each)")
    ap.add_argument("--tau", type=float, default=0.0, help="threshold for --gate-by (0 = keep all)")
    ap.add_argument("--only", default="all", choices=["all", "within", "cross"],
                    help="restrict to within-task (base==target) or cross-task (base!=target). "
                         "within-task orth is small (~<0.6) so pair with a LOW --tau (e.g. 0.26)")
    ap.add_argument("--kind", default="all", choices=["swap", "push", "all"],
                    help="swap (input single-patch, default) | push (two-site @ MLP: STR or Gaussian) | all")
    ap.add_argument("--strategy", default="all", choices=["all", "STR", "Gaussian"],
                    help="restrict to one strategy, e.g. Gaussian to run ONLY Gaussian-push")
    ap.add_argument("--include-tp", action="store_true",
                    help="also instrument in-circuit (TP) components so ripple_xint_in_orth "
                         "values exist for FP-vs-TP comparison")
    ap.add_argument("--device", default="cuda")
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
    from interaction_effects import build_task_clouds
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.combo).generate()
    pipeline = d["pipeline"]
    grid = build_full_grid(pipeline, d["task"], list(range(d["n_layers"])),
                           positions=[args.position], patch_at=args.patch_at, include_residual=False)
    unit_lists = [_units_of(t) for t in grid.values()]             # each site = list of units
    unit_by_id = {ul[0].id: ul for ul in unit_lists}               # site_id -> unit LIST (component_xint contract)
    all_units = [u for ul in unit_lists for u in ul]               # flat, for forward_cache/clouds
    clouds = build_task_clouds(pipeline, d["pool"], all_units)      # {task: {site: (M,D)}}
    rows_idx = load_scan_rows(results_dir)

    n_written = 0
    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None or info["base_task"] is None:
            continue
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        kind = "swap" if push == "input" else ("self" if push == "self" else "push")
        if kind == "self":
            continue                                           # self-push: INT degenerate, no bypass to ripple
        if (args.kind != "all" and kind != args.kind) or \
           (args.strategy != "all" and strat != args.strategy):
            continue
        if kind == "push" and push not in unit_by_id:
            continue                                           # can't locate the push site in the grid
        is_within = (info["base_task"] == info["target_task"])
        if (args.only == "within" and not is_within) or (args.only == "cross" and is_within):
            continue                                           # within/cross-task filter
        base = info["base_task"]
        if base not in clouds:
            continue
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])
        n_scan = int(next(iter(hp.values())).shape[0])

        # rebuild the counterfactual: swap -> (base,target); push -> add push_site @ MLP;
        # Gaussian has no task direction (drop target_task). hp[c] is c's rep UNDER the
        # intervention (swapped, or downstream of the MLP push); delta_c/ripple are identical.
        mk = dict(strategy=strat, base_task=base, alpha=alpha)
        if kind == "push":
            mk["push_site"] = unit_by_id[push][0]              # two-site push at the shared MLP
        if strat == "STR":
            mk["target_task"] = info["target_task"]            # Gaussian: no target direction
        spec = make_counterfactual(d, n=n_scan, seed=args.seed, **mk)
        if len(spec.cf_noise) != n_scan:
            print(f"[ripple][skip] {direc} a{alpha}: rebuilt {len(spec.cf_noise)} != saved {n_scan}")
            continue
        bases = [ex["input"] for ex in spec.cf_noise]
        pos = spec.positions
        readout = make_readout(spec.cf_noise, spec.positions)
        base_scores, base_slabs = pipeline.forward_cache(bases, all_units, batch_size=args.batch_size)
        h = {uid: _gather(base_slabs[uid], pos) for uid in base_slabs}
        Ya = readout(base_scores)                               # for the xINT map

        # in/off-circuit label + NIE per site (from the saved rows of this scan)
        incirc, nie_by_site = {}, {}
        for sid in hp:
            hit = rows_idx.get((strat, push, direc, alpha, sid))
            incirc[sid] = (hit[0].get("in_base_circuit") if hit else None)
            nie_by_site[sid] = (hit[0].get("NIE", 0.0) if hit else 0.0)
        partners = [unit_by_id[sid] for sid in hp if sid in unit_by_id]

        # candidates = OFF-circuit sites, gated by divergence (default) not interaction — the
        # FP mechanism is geometric, so an |INT| gate would drop the actual false positives.
        # With --include-tp, also instrument in-circuit sites (for FP-vs-TP ripple comparison).
        def _keep_off(r):
            if r.get("degenerate") or r.get("in_base_circuit") is not False:
                return False
            if args.gate_by == "int":
                return r.get("INT") is not None and abs(r["INT"]) >= args.tau
            if args.gate_by == "orth":
                return r.get("orth_base") is not None and abs(r["orth_base"]) >= args.tau
            return True                                          # 'none' -> all off-circuit

        def _keep_tp(r):
            return not r.get("degenerate") and r.get("in_base_circuit") is True

        def _keep(r):
            if _keep_off(r):
                return True
            if args.include_tp and _keep_tp(r):
                return True
            return False

        cand = [sid for sid in hp if sid != push                     # never the push site itself
                and (hit := rows_idx.get((strat, push, direc, alpha, sid))) and _keep(hit[0])]

        for cid in cand:
            if cid not in unit_by_id or cid not in h:
                continue
            delta_c = hp[cid] - h[cid]
            # push c by its own divergence -> the noising forward; cache every site
            _, rippled = pipeline.forward_cache(bases, all_units,
                                                push=(unit_by_id[cid][0], delta_c, pos),
                                                batch_size=args.batch_size)
            orthN = {}                                          # off-base-manifold ripple per site
            for uid in rippled:
                if uid == cid or uid not in clouds[base]:
                    continue
                dN = _gather(rippled[uid], pos) - h[uid]
                orthN[uid] = float(manifold_decompose(dN, clouds[base][uid])[1].mean())
            # xINT map (c vs every partner) -> WEIGHT the ripple by how much c interacts with N
            xmap = component_xint(pipeline, bases, readout, Ya, unit_by_id[cid], partners,
                                  hp, nie_by_site, pos, batch_size=args.batch_size)

            def _wavg(keep):                                    # |xINT|-weighted mean orth over kept N
                num = den = 0.0
                for uid, o in orthN.items():
                    if keep(uid):
                        w = abs(xmap.get(uid, 0.0)); num += w * o; den += w
                return num / (den + 1e-9)
            orth_in = [o for u, o in orthN.items() if incirc.get(u) is True]
            orth_off = [o for u, o in orthN.items() if incirc.get(u) is False]
            ri = float(sum(orth_in) / len(orth_in)) if orth_in else 0.0
            ro = float(sum(orth_off) / len(orth_off)) if orth_off else 0.0
            fields = {"ripple_in_orth": round(ri, 4), "ripple_off_orth": round(ro, 4),
                      "ripple_frac": round(ri / (ri + ro + 1e-9), 4),
                      # xINT-WEIGHTED ripple: off-manifold push on the components c INTERACTS with
                      "ripple_xint_orth": round(_wavg(lambda u: True), 4),
                      "ripple_xint_in_orth": round(_wavg(lambda u: incirc.get(u) is True), 4)}
            row, path = rows_idx[(strat, push, direc, alpha, cid)]
            row.update(fields)
            json.dump(row, open(path, "w"), indent=2)
            n_written += 1
        print(f"[ripple] {strat:>8} {kind:>4} {direc:>7} a{alpha:<4g} candidates {len(cand)}")

    print(f"\nwrote ripple_* (incl. xINT-weighted) onto {n_written} candidate rows")


if __name__ == "__main__":
    main()
