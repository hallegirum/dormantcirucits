"""check_endpoint_density_tp.py — endpoint density for TRUE POSITIVES only.

Companion to check_endpoint_density.py. Skips off-circuit components (already
computed) and processes all in-circuit components. Outputs the same JSONL
record format with is_fp=0 and in_circuit=1 so records can be concatenated
with the original dump. Threshold-based TP filtering can be applied at plot time.

For each TP, computes:
  - off_dist: min normalised endpoint distance to ANY task's data
  - dist_K:   normalised endpoint distance to the most-aligned non-base task K
  - subset:   "borrowed" if K is in the component's home_tasks, else "epotic"

Usage (same interface as the original, just add --dump):
    python check_endpoint_density_tp.py --results-root poly_results_fullgap/poly_results_new/results \
        --combo cases_0+1 --seed 0 --case poly --n 150 --device cuda \
        --dump endpoint_density_tp.jsonl
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from safetensors.torch import load_file

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE.parent / "interp_bench"))
from interaction_effects import build_task_clouds, _units_of


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
    for p in glob.glob(os.path.join(results_dir, "interaction", "**", "int.json"), recursive=True):
        try:
            r = json.load(open(p))
        except Exception:
            continue
        if "strategy" in r:
            idx[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = r
    return idx


def whiten_fit(C, d=8):
    C = np.asarray(C, dtype=np.float64)
    mu = C.mean(0)
    _, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    d = int(min(d, (S > 1e-6).sum()))
    V = Vt[:d]
    std = S[:d] / np.sqrt(max(len(C) - 1, 1))
    return mu, V, std


def wdist(X, fit):
    mu, V, std = fit
    Z = (np.asarray(X, dtype=np.float64) - mu) @ V.T / (std + 1e-6)
    return np.linalg.norm(Z, axis=1)


def anie(r):
    v = r.get("abs_NIE")
    if v is None:
        v = r.get("NIE")
    return abs(float(v)) if isinstance(v, (int, float)) and np.isfinite(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--reps-dir", default=None)
    ap.add_argument("--dump", required=True, help="append TP records to this JSONL file")
    ap.add_argument("--combo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--case", default="poly")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--position", default="last_token")
    ap.add_argument("--patch-at", default="internal")
    ap.add_argument("--pca-dim", type=int, default=8)
    args = ap.parse_args()

    if args.reps_dir:
        reps_root = os.path.normpath(args.reps_dir)
    else:
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
    grid = build_full_grid(d["pipeline"], d["task"], list(range(d["n_layers"])),
                           positions=[args.position], patch_at=args.patch_at, include_residual=False)
    unit_lists = [_units_of(t) for t in grid.values()]
    all_units = [u for ul in unit_lists for u in ul]
    clouds = build_task_clouds(d["pipeline"], d["pool"], all_units)

    # load int.json rows
    rows = load_scan_rows(results_dir)
    if not rows:
        alt = glob.glob(os.path.join(results_dir, "**", "int.json"), recursive=True)
        for p in alt:
            try:
                r = json.load(open(p))
            except Exception:
                continue
            if "strategy" in r:
                rows[(r["strategy"], r["push_site"], r["direction"], float(r["alpha"]), r["site"])] = r

    # whitening fits
    fits = defaultdict(dict)
    insc = defaultdict(dict)
    for K, sites in clouds.items():
        for sid, C in sites.items():
            Cn = np.asarray(C.detach().cpu() if hasattr(C, "detach") else C, dtype=np.float64)
            if len(Cn) < args.pca_dim + 2:
                continue
            fit = whiten_fit(Cn, args.pca_dim)
            fits[K][sid] = fit
            insc[K][sid] = float(np.median(wdist(Cn, fit)) + 1e-9)

    tasks = list(clouds.keys())
    records = []
    n_tp = 0

    for dpath in sorted(glob.glob(os.path.join(reps_root, "int-*"))):
        info = parse_reps_dir(os.path.basename(dpath))
        if info is None or info["push_site"] == "self" or info["base_task"] is None:
            continue
        strat, push, direc, alpha = info["strategy"], info["push_site"], info["direction"], info["alpha"]
        base = info["base_task"]
        if base not in clouds:
            continue
        kind = "swap" if push == "input" else "push"
        within = (info["base_task"] == info["target_task"])
        regime = ("swap" if kind == "swap" else "STR_push") + ("_within" if within else "_cross")
        sf = glob.glob(os.path.join(dpath, "*_intervened_reps.safetensors"))
        if not sf:
            continue
        hp = load_file(sf[0])

        for sid, Hp in hp.items():
            r = rows.get((strat, push, direc, alpha, sid))
            if r is None:
                continue
            if r.get("in_base_circuit") is not True:
                continue

            n_tp += 1
            Hpn = np.asarray(Hp.detach().cpu() if hasattr(Hp, "detach") else Hp, dtype=np.float64)

            # off_dist: min normalised distance to ANY task's data
            dmins = [float(np.median(wdist(Hpn, fits[K2][sid]))) / insc[K2][sid]
                     for K2 in fits if sid in fits[K2]]
            off_dist = min(dmins) if dmins else None

            # dist_K: distance to the most-aligned non-base task
            dist_K = subset = None
            abt = {k: v for k, v in (r.get("along_by_task") or {}).items()
                   if k != str(base) and isinstance(v, (int, float))}
            if abt:
                K = int(max(abt, key=abt.get))
                if K in fits and sid in fits[K]:
                    home = {int(t) for t in (r.get("home_tasks") or [])}
                    subset = "borrowed" if K in home else "epotic"
                    dist_K = float(np.median(wdist(Hpn, fits[K][sid]))) / insc[K][sid]

            records.append({
                "combo": args.combo, "seed": args.seed, "regime": regime,
                "is_fp": 0, "in_circuit": 1, "site": sid,
                "off_dist": off_dist, "dist_K": dist_K, "subset": subset,
                "gap_frac": float(r["gap_frac"]) if isinstance(r.get("gap_frac"), (int, float)) else None,
            })

    with open(args.dump, "a") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    print(f"[{args.combo} seed{args.seed}] {n_tp} TPs processed, {len(records)} records -> {args.dump}")


if __name__ == "__main__":
    main()
