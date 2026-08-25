"""
test_makelov_decomp.py  —  TEST 2: the Makelov decomposition (needs the model)
==============================================================================
For every saved intervened-reps set, decomposes the divergence delta_c = h'_c - mu_base
TWICE, against the two structures that define Makelov-style (Type-R) dormancy:

  (1) manifold:  tangent(base-task cloud)  vs  normal   -> ON-manifold vs OFF-manifold
  (2) readout :  rowspace(W_out/W_O)        vs  ker      -> READABLE vs DISCONNECTED

Makelov dormancy = the OUTPUT-carrying part of the divergence lives in a direction that
is (readable by the down-projection) AND (off the on-manifold tangent). So the key signal
per component is

    readable_off  = || proj_rowspace( delta_normal ) ||        (orth  ∩  rowspace)

i.e. the OFF-manifold excursion RESTRICTED to what the layer actually reads. A pure
ker(W_out) excursion is off-manifold but output-null (Makelov's "disconnected" part —
harmless); readable_off strips that out and is the sharpened version of orth_base.

Reps are stored per (strategy, direction, alpha, push_site) as safetensors {site_id: h'_c}
(reps_out = hp in run_layer_scans). Weights: MLP down-proj W_out [d_mlp, d_model] at
mlp.hook_post; head W_O[head] [d_head, d_model] at attn.hook_z. delta dim D matches.

Runs ONE (combo, seed) per invocation (it rebuilds that exact model). Strategy axis: every
statistic is aggregated by strategy so you can see which interventions produce readable-
off-manifold (Type-R) divergence vs which do not.

    python test_makelov_decomp.py --results-root poly_results_new \
        --combo cases_0+2 --seed 0 --case poly --device cpu
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "interp_bench"))

from collect_interaction import load_rows          # for the in_base_circuit / NIE lookup


# --------------------------------------------------------------------------- #
#  linear-algebra primitives
# --------------------------------------------------------------------------- #
def tangent_basis(cloud: torch.Tensor, d: int = 8, var_keep: float = 0.99):
    """Top-d_eff right singular vectors of the CENTERED cloud (M,D) -> (d_eff, D) rows,
    plus the cloud mean (D,). Effective-rank capped exactly like manifold_decompose."""
    X = cloud.double()
    mu = X.mean(0)
    Xc = X - mu
    _u, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    if float(S[0]) <= 1e-8:
        return torch.zeros((0, X.shape[1]), dtype=torch.double), mu
    energy = (S ** 2).cumsum(0) / (S ** 2).sum().clamp_min(1e-12)
    d_eff = int(min(d, int((energy < var_keep).sum().item()) + 1, S.shape[0]))
    return Vt[:d_eff], mu


def rowspace_basis(W_proj: torch.Tensor, tol_ratio: float = 1e-4):
    """Readable directions in R^D for the map x -> x @ W_proj. W_proj is (D, d_model);
    the map's rowspace = span of the left singular vectors of W_proj with S>tol -> (D, r)."""
    U, S, _Vt = torch.linalg.svd(W_proj.double(), full_matrices=False)
    if S.numel() == 0 or float(S[0]) <= 0:
        return torch.zeros((W_proj.shape[0], 0), dtype=torch.double)
    keep = S > (tol_ratio * float(S[0]))
    return U[:, keep]                                    # (D, r), orthonormal columns


def _proj(delta: torch.Tensor, basis_rows_or_cols, rows: bool):
    """Project rows of delta (N,D) onto the span of `basis`. If rows=True basis is
    (k,D) row-vectors; else (D,r) column-vectors. Returns the projected (N,D)."""
    if rows:
        B = basis_rows_or_cols                           # (k, D)
        if B.shape[0] == 0:
            return torch.zeros_like(delta)
        return (delta @ B.T) @ B
    B = basis_rows_or_cols                                # (D, r)
    if B.shape[1] == 0:
        return torch.zeros_like(delta)
    return (delta @ B) @ B.T


def decompose(delta: torch.Tensor, tan_rows: torch.Tensor, row_cols: torch.Tensor) -> dict:
    """delta (N,D) displacement. tan_rows (k,D) tangent basis. row_cols (D,r) readout
    rowspace. Returns per-example-averaged norms + fractions (the Makelov split)."""
    d = delta.double()
    n = d.norm(dim=-1).clamp_min(1e-12)                   # (N,)
    d_tan = _proj(d, tan_rows, rows=True)
    d_norm = d - d_tan                                    # OFF-manifold part
    d_norm_read = _proj(d_norm, row_cols, rows=False)     # off-manifold ∩ readable
    d_read = _proj(d, row_cols, rows=False)               # whole delta, readable part

    tan_n = d_tan.norm(dim=-1)
    orth_n = d_norm.norm(dim=-1)
    read_off_n = d_norm_read.norm(dim=-1)
    read_n = d_read.norm(dim=-1)
    return {
        "tangent_norm":   float(tan_n.mean()),
        "orth_norm":      float(orth_n.mean()),                        # == orth_base (recomputed)
        "readable_off":   float(read_off_n.mean()),                    # orth ∩ rowspace  (Type-R signal)
        "readable_off_frac": float((read_off_n / orth_n.clamp_min(1e-12)).mean()),
        "ker_frac":       float((1.0 - read_n / n).mean()),            # disconnected fraction of whole delta
        "n_examples":     int(delta.shape[0]),
    }


# --------------------------------------------------------------------------- #
#  site_id -> (layer, head, kind) and the matching output projection
# --------------------------------------------------------------------------- #
def parse_site(site_id: str):
    layer = int(m.group(1)) if (m := re.search(r"Layer-(\d+)", site_id)) else None
    head = int(m.group(1)) if (m := re.search(r"Head-(\d+)", site_id)) else None
    kind = "mlp" if "mlp" in site_id.lower() else ("head" if head is not None else "other")
    return layer, head, kind


def output_proj(model, site_id: str):
    """W_out (MLP, [d_mlp,d_model]) or W_O[head] (attn, [d_head,d_model]) for this site."""
    layer, head, kind = parse_site(site_id)
    if layer is None:
        return None
    blk = model.blocks[layer]
    if kind == "mlp":
        return blk.mlp.W_out.detach()
    if kind == "head":
        return blk.attn.W_O[head].detach()
    return None


# --------------------------------------------------------------------------- #
#  reps-dir name  ->  (strategy, direction, alpha, push_site)
# --------------------------------------------------------------------------- #
def parse_reps_dir(name: str):
    # "int-{strategy}__{direction}-a{alpha}__{push_site}"
    m = re.match(r"int-([^_]+)__(.+)-a([0-9.]+)__(.+)$", name)
    if not m:
        return None
    strat, direction, alpha, push = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    base_task = direction.split("-")[0]
    within = (len(direction.split("-")) == 2 and direction.split("-")[0] == direction.split("-")[1])
    if push == "input":
        bucket = f"STR_swap_{'within' if within else 'cross'}"
    elif push == "self":
        bucket = f"{strat}_self"
    else:
        bucket = f"{strat}_push"
    return {"strategy": strat, "direction": direction, "alpha": alpha,
            "push_site": push, "base_task": base_task, "bucket": bucket}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--combo", required=True, help="e.g. cases_0+2 (poly_comb)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", default="poly")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--patch-at", default="internal")
    ap.add_argument("--out", default="makelov_analysis")
    args = ap.parse_args()

    # locate the intervened_reps dir for this (combo, seed)
    hits = glob.glob(os.path.join(args.results_root, "**", f"poly_{args.combo}",
                                  f"int_seed{args.seed}", "intervened_reps"), recursive=True)
    if not hits:
        raise SystemExit(f"no intervened_reps for poly_{args.combo}/int_seed{args.seed} under {args.results_root}")
    reps_root = hits[0]
    print(f"reps root: {reps_root}")

    # rebuild THIS model + per-task clouds + weights (same recipe as the scan)
    from run_dormancy_pipeline import PrepareData
    from path_finder import build_full_grid
    from interaction_effects import build_task_clouds
    from activation_manifold import set_seed
    set_seed(args.seed)
    d = PrepareData(args.case, n=args.n, seed=args.seed, device=args.device,
                    position=args.position, poly_comb=args.combo).generate()
    pipeline, model = d["pipeline"], d["pipeline"].model
    layers = list(range(d["n_layers"]))
    grid = build_full_grid(pipeline, d["task"], layers, positions=[args.position],
                           patch_at=args.patch_at, include_residual=False)
    units = [t.flatten()[0] for t in grid.values()]
    print(f"grid: {len(units)} sites | fitting per-task clouds ...")
    clouds = build_task_clouds(pipeline, d["pool"], units)          # {task: {site_id: (M,D)}}

    # precompute tangent bases (per task, per site) and rowspace bases (per site)
    tan_cache = {task: {sid: tangent_basis(c) for sid, c in sites.items()}
                 for task, sites in clouds.items()}
    row_cache, mu_cache = {}, {}
    for sid in clouds[next(iter(clouds))].keys():
        Wp = output_proj(model, sid)
        row_cache[sid] = rowspace_basis(Wp) if Wp is not None else None

    # in_base_circuit / NIE lookup so we can split FP vs in-circuit
    rows = load_rows(args.results_root, [args.combo], [args.seed])
    meta = {(r["site"], r["strategy"], r["push_site"], r["direction"], float(r["alpha"])):
            (r.get("in_base_circuit"), r.get("NIE")) for r in rows}

    per_site, skipped = [], 0
    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None:
            continue
        base_task = int(info["base_task"]) if info["base_task"].isdigit() else None
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf or base_task is None or base_task not in clouds:
            skipped += 1
            continue
        reps = load_file(sf[0])                                    # {site_id: (N,D)}
        for sid, hp in reps.items():
            tan = tan_cache.get(base_task, {}).get(sid)
            row = row_cache.get(sid)
            if tan is None or row is None or sid not in clouds[base_task]:
                continue
            mu = clouds[base_task][sid].double().mean(0)
            delta = hp.double() - mu                                # displacement from base manifold
            try:
                dec = decompose(delta, tan, row)
            except Exception:
                continue
            ibc, nie = meta.get((sid, info["strategy"], info["push_site"], info["direction"], info["alpha"]),
                                (None, None))
            per_site.append({"site": sid, "bucket": info["bucket"], "strategy": info["strategy"],
                             "push_site": info["push_site"], "direction": info["direction"],
                             "alpha": info["alpha"], "in_base_circuit": ibc, "NIE": nie, **dec})
    print(f"decomposed {len(per_site)} site-intervention reps ({skipped} dirs skipped)")

    # aggregate over the strategy axis (x FP/in-circuit)
    groups = defaultdict(list)
    for s in per_site:
        circ = ("in_circuit" if s["in_base_circuit"] is True
                else "off_circuit" if s["in_base_circuit"] is False else "unknown")
        groups[(s["bucket"], circ)].append(s)
    agg = {}
    for (bucket, circ), items in sorted(groups.items()):
        agg[f"{bucket} | {circ}"] = {
            "n": len(items),
            "mean_orth_norm":       round(float(np.mean([x["orth_norm"] for x in items])), 4),
            "mean_readable_off":    round(float(np.mean([x["readable_off"] for x in items])), 4),
            "mean_readable_off_frac": round(float(np.mean([x["readable_off_frac"] for x in items])), 4),
            "mean_ker_frac":        round(float(np.mean([x["ker_frac"] for x in items])), 4),
            "mean_tangent_norm":    round(float(np.mean([x["tangent_norm"] for x in items])), 4),
        }

    os.makedirs(args.out, exist_ok=True)
    tag = f"{args.combo}_seed{args.seed}"
    import csv
    with open(os.path.join(args.out, f"decomp_{tag}.csv"), "w", newline="") as f:
        cols = ["site", "bucket", "strategy", "push_site", "direction", "alpha", "in_base_circuit",
                "NIE", "orth_norm", "readable_off", "readable_off_frac", "ker_frac", "tangent_norm",
                "n_examples"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_site)
    json.dump(agg, open(os.path.join(args.out, f"decomp_by_strategy_{tag}.json"), "w"), indent=2)

    print("\n== Makelov decomposition by strategy (readable_off = orth ∩ rowspace = Type-R signal) ==")
    print(f"  {'strategy | circuit':<30} {'n':>5} {'orth':>8} {'read_off':>9} "
          f"{'read_frac':>10} {'ker_frac':>9}")
    for name, a in agg.items():
        print(f"  {name:<30} {a['n']:>5} {a['mean_orth_norm']:>8} {a['mean_readable_off']:>9} "
              f"{a['mean_readable_off_frac']:>10} {a['mean_ker_frac']:>9}")
    print(f"\nwrote decomp_{tag}.csv + decomp_by_strategy_{tag}.json under {args.out}/")


if __name__ == "__main__":
    main()
